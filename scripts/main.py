import torch
import argparse
from argparse import ArgumentParser
import numpy as np
from torch_geometric.loader import DataLoader
from torch.nn import BCELoss, CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
import time
import os
import pickle
import logging
import json
import pandas as pd
from dataset import MixBench
from stop import EarlyStopping
from model import HeteroGNN, HeteroRGCN
import trainer
from trainer import train as trainer_train
from trainer import evaluate as trainer_eval
torch.cuda.empty_cache()


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GRAPHS = "AST_CFG_PDG_CAST_DEP"
ROOT = "../raw/MixBench/time_root"
TESTROOT = "../raw/MixBench/time_root"
BENCH = ""
PRETRAINED = "../raw/model/time_AST_CFG_PDG_CAST_DEP_checkpoint.pt"
BATCH = 16

if "time" in ROOT:
    LABEL = "runtime"
else:
    LABEL = "error"
LOAD = True


def dataset():
    dataset = MixBench(root=ROOT)
    print("Finish dataset building.\nDataset size is: ", str(len(dataset)) )
    loader = DataLoader(dataset, batch_size=1)
    r0_counter = 0
    e0_counter = 0
    r1_counter = 0
    e1_counter = 0

    nodes = 0
    edges = 0

    for data in tqdm(loader, position=0, leave=True):
        runtime = int(data.y[0])
        error = int(data.y[1])
        nodes += data["node"].x.shape[0]
        edges += data["node", "AST", "node"]['edge_index'].shape[1]
        edges += data["node", "CFG", "node"]['edge_index'].shape[1]
        edges += data["node", "PDG", "node"]['edge_index'].shape[1]
        edges += data["node", "CAST", "node"]['edge_index'].shape[1]
        edges += data["node", "DEP", "node"]['edge_index'].shape[1]

        if runtime == 0: 
            r0_counter += 1
        else:
            r1_counter += 1
        if error == 0:
            e0_counter +=1
        else:
            e1_counter += 1
    print("in all:")
    print("# runtime == 0: ", r0_counter, " # runtime == 1: ", r1_counter)
    print("# error == 0: ", e0_counter, " # error == 1: ", e1_counter)
    avgedge = edges / len(loader)
    avgnode = nodes / len(loader)
    print(f"Average edge number per graph = {avgedge}, average node number per graph = {avgnode}")

def train():
    if not os.path.exists("log"):
        os.makedirs("log")
    if not os.path.exists("model"):
        os.makedirs("model")

    dataset = MixBench(root=ROOT)
    savedir = f"{LABEL}_{GRAPHS}"
    if not os.path.exists(os.path.join("model", savedir)):
        os.makedirs(os.path.join("model", savedir))
    if not os.path.exists(os.path.join("log", savedir)):
        os.makedirs(os.path.join("log", savedir))
    logging.basicConfig(filename=os.path.join("log", savedir, "train.log"), format='%(asctime)s %(message)s', filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    print("Savedir: ", savedir)
    logger.info("Savedir: " + savedir)

    D = dataset.shuffle()
    train_size = int(0.7 * len(D)) 
    train_dataset = D[:train_size]
    val_size = int(0.1 * len(D))
    val_dataset = D[train_size:(train_size + val_size)]
    test_dataset = D[(train_size + val_size):]

    print("Dataset size: " + str(len(dataset)))
    print("Split: \n  train -> {}, val -> {}, test -> {}".format(len(train_dataset), len(val_dataset), len(test_dataset)))
    print("Edges: " + str(GRAPHS.split('_')))
    logger.info("Dataset size: " + str(len(dataset)))
    logger.info("Split: \n  train -> {}, val -> {}, test -> {}".format(len(train_dataset), len(val_dataset), len(test_dataset)))
    logger.info("Edges: " + str(GRAPHS.split('_')))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, pin_memory=True)

    model = HeteroGNN(GRAPHS.split('_'))
    # model = HeteroRGCN(GRAPHS.split('_'))

    loss_function = BCELoss(reduction="sum").to(DEVICE)
    optim = Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    model.to(DEVICE)

    logger.info("============> Training mode start...")

    early_stopping = EarlyStopping(patience=30, verbose=True, path=os.path.join("model", savedir, 'checkpoint.pt'), trace_func=logger.info)
    
    for epoch in tqdm(range(0, 500)):
        start_time = time.monotonic()    
        train_loss, train_acc, train_acc0, train_acc1  = trainer_train(model, train_loader, optim, DEVICE, loss_function, LABEL)
        print(f'\tTrain Loss: {train_loss:.6f} | Train Acc: {train_acc * 100:.3f}% | Acc0: {train_acc0 * 100:.2f}% |  Acc1: {train_acc1 * 100:.2f}% ')
        logger.info(f'\tTrain Loss: {train_loss:.6f} | Train Acc: {train_acc * 100:.3f}% | Acc0: {train_acc0 * 100:.2f}% |  Acc1: {train_acc1 * 100:.2f}% ')
        
        valid_loss, valid_acc, valid_pre, valid_rec, valid_fsc, \
                pre0, pre1, rec0, rec1, \
                train_acc0, train_acc1,\
                = trainer_eval(model, val_loader, DEVICE, loss_function, LABEL)
        print(
                f'\t Val. Loss: {valid_loss:.6f} |  Val. Acc: {valid_acc * 100:.3f}% |  Val. Pre: {valid_pre * 100:.2f}% |  Val. Rec: {valid_rec * 100:.2f}% |  Val. Fsc: {valid_fsc * 100:.2f}%')
        logger.info(
                f'\t Val. Loss: {valid_loss:.6f} |  Val. Acc: {valid_acc * 100:.3f}% |  Val. Pre: {valid_pre * 100:.2f}% |  Val. Rec: {valid_rec * 100:.2f}% |  Val. Fsc: {valid_fsc * 100:.2f}%')
        print(
                f'\t Acc0: {train_acc0 * 100:.2f}% |  Acc1: {train_acc1 * 100:.2f}% |  Pre0: {pre0 * 100:.2f}% |  Pre1: {pre1 * 100:.2f}% |  Rec0: {rec0 * 100:.2f}% | Rec1: {rec1 * 100:.2f}% ')      
        logger.info(
                f'\t Acc0: {train_acc0 * 100:.2f}% |  Acc1: {train_acc1 * 100:.2f}% |  Pre0: {pre0 * 100:.2f}% |  Pre1: {pre1 * 100:.2f}% |  Rec0: {rec0 * 100:.2f}% | Rec1: {rec1 * 100:.2f}% ')
        
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(),
                    os.path.join("model", savedir, 'epoch_{}_'.format(epoch + 1) + "checkpoint.pt"))
            
        
        end_time = time.monotonic()
        epoch_mins, epoch_secs = trainer.epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            logger.info("Early stopping")
            break

def test():
    if not os.path.exists("log"):
        os.makedirs("log")
    if not os.path.exists("model"):
        os.makedirs("model")
    dataset = MixBench(root=TESTROOT)
    savedir = f"{LABEL}_{GRAPHS}"

    if not os.path.exists(os.path.join("log", savedir)):
        os.makedirs(os.path.join("log", savedir))
    logging.basicConfig(filename=os.path.join("log", savedir, "test.log"), format='%(asctime)s %(message)s', filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    print("Savedir: ", savedir)
    logger.info("Savedir: " + savedir)

    # STEP 1 - Prepare dataset
    D = dataset.shuffle()
    train_size = int(0.7 * len(D))
    val_size = int(0.1 * len(D))
    test_dataset = D[(train_size + val_size):]
    test_loader = DataLoader(test_dataset, batch_size=BATCH, pin_memory=True)

    print("Test dataset size: " + str(len(test_dataset)))
    print("Split: \n  test -> {}".format(len(test_dataset)))
    print("Edges: " + str(GRAPHS.split('_')))
    logger.info("Test dataset size: " + str(len(test_dataset)))
    logger.info("Split: \n  test -> {}".format(len(test_dataset)))
    logger.info("Edges: " + str(GRAPHS.split('_')))

    model = HeteroGNN(GRAPHS.split('_'))
    # model = HeteroRGCN(GRAPHS.split('_'))

    loss_function = BCELoss(reduction="sum").to(DEVICE)
    
    if LOAD:
        model.load_state_dict(torch.load(PRETRAINED)) 
        print("Model ({}) is loaded.".format(PRETRAINED.split('/')[-1]))
        logger.info("Model ({}) is loaded.".format(PRETRAINED.split('/')[-1]))

    model.to(DEVICE)

    data = next(iter(test_loader)).to(DEVICE)
    with torch.no_grad():  # Initialize lazy modules.
        model(data, DEVICE)
    

    # STEP 3 - Start evaluation
    logger.info("============> Testing mode start...")

    test_loss, test_acc, test_pre, test_rec, test_fsc, \
                pre0, pre1, rec0, rec1,\
                train_acc0, train_acc1 \
                    = trainer_eval(model, test_loader, DEVICE, loss_function, LABEL)
    print(
            f'\t Test Loss: {test_loss:.6f} |  Test Acc: {test_acc * 100:.3f}% |  Test Pre: {test_pre * 100:.2f}% |  Test Rec: {test_rec * 100:.2f}% |  Test Fsc: {test_fsc * 100:.2f}%')
    logger.info(
            f'\t Test Loss: {test_loss:.6f} |  Test Acc: {test_acc * 100:.3f}% |  Test Pre: {test_pre * 100:.2f}% |  Test Rec: {test_rec * 100:.2f}% |  Test Fsc: {test_fsc * 100:.2f}%')

    print(
                f'\t Acc0: {train_acc0 * 100:.2f}% |  Acc1: {train_acc1 * 100:.2f}% |  Pre0: {pre0 * 100:.2f}% |  Pre1: {pre1 * 100:.2f}% |  Rec0: {rec0 * 100:.2f}% | Rec1: {rec1 * 100:.2f}% ')      
    logger.info(
                f'\t Acc0: {train_acc0 * 100:.2f}% |  Acc1: {train_acc1 * 100:.2f}% |  Pre0: {pre0 * 100:.2f}% |  Pre1: {pre1 * 100:.2f}% |  Rec0: {rec0 * 100:.2f}% | Rec1: {rec1 * 100:.2f}% ')

def transfer():
    dataset = MixBench(root=ROOT)
    if LOAD:
        savedir = f"trl_{BENCH}"
    else:
        savedir = f"scr_{BENCH}"
    if not os.path.exists(os.path.join("model", savedir)):
        os.makedirs(os.path.join("model", savedir))
    if not os.path.exists(os.path.join("log", savedir)):
        os.makedirs(os.path.join("log", savedir))
    logging.basicConfig(filename=os.path.join("log", savedir, "train.log"), format='%(asctime)s %(message)s', filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    print("Savedir: ", savedir)
    logger.info("Savedir: " + savedir)

    D = dataset.shuffle()
    train_size = int(0.8 * len(D)) 
    train_dataset = D[:train_size]
    test_dataset = D[train_size:]

    print("Target dataset size: " + str(len(D)))
    print("Split: \n  train -> {}, test -> {}".format(len(train_dataset), len(test_dataset)))
    print("Edges: " + str(GRAPHS.split('_')))
    logger.info("Target dataset size: " + str(len(D)))
    logger.info("Split: \n  train -> {}, test -> {}".format(len(train_dataset), len(test_dataset)))
    logger.info("Edges: " + str(GRAPHS.split('_')))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, pin_memory=True)

    model = HeteroGNN(GRAPHS.split('_'))
    loss_function = BCELoss(reduction="sum").to(DEVICE)
    if LOAD:
        model.load_state_dict(torch.load(os.path.join("model", PRETRAINED)))
        print("Model ({}) is loaded.".format(PRETRAINED))
        logger.info("Model ({}) is loaded.".format(PRETRAINED))
        # for param in model.parameters():
        #     param.requires_grad = False
        model.classifier = torch.nn.Linear(in_features=100, out_features=1)
        
    
    optim = Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    model.to(DEVICE)

    data = next(iter(train_loader)).to(DEVICE)
    with torch.no_grad():  # Initialize lazy modules.
        model(data, DEVICE)

    logger.info("============> Training mode start...")

    early_stopping = EarlyStopping(patience=30, verbose=True, path=os.path.join("model", savedir, 'checkpoint.pt'), trace_func=logger.info)
    for epoch in tqdm(range(0, 100)):
        start_time = time.monotonic()
        train_loss, train_acc, train_acc0, train_acc1  = trainer_train(model, train_loader, optim, DEVICE, loss_function, LABEL)
        print(f'\tTrain Loss: {train_loss:.6f} | Train Acc: {train_acc * 100:.3f}% | Acc0: {train_acc0 * 100:.2f}% |  Acc1: {train_acc1 * 100:.2f}% ')
        logger.info(f'\tTrain Loss: {train_loss:.6f} | Train Acc: {train_acc * 100:.3f}% | Acc0: {train_acc0 * 100:.2f}% |  Acc1: {train_acc1 * 100:.2f}% ')
        
        valid_loss, valid_acc, valid_pre, valid_rec, valid_fsc, \
                    pre0, pre1, rec0, rec1, \
                    train_acc0, train_acc1 \
                    = trainer_eval(model, test_loader, DEVICE, loss_function, LABEL)
        print(
                    f'\t Val. Loss: {valid_loss:.6f} |  Val. Acc: {valid_acc * 100:.3f}% |  Val. Pre: {valid_pre * 100:.2f}% |  Val. Rec: {valid_rec * 100:.2f}% |  Val. Fsc: {valid_fsc * 100:.2f}%')
        logger.info(
                    f'\t Val. Loss: {valid_loss:.6f} |  Val. Acc: {valid_acc * 100:.3f}% |  Val. Pre: {valid_pre * 100:.2f}% |  Val. Rec: {valid_rec * 100:.2f}% |  Val. Fsc: {valid_fsc * 100:.2f}%')
        print(
                    f'\t Acc0: {train_acc0 * 100:.2f}% |  Acc1: {train_acc1 * 100:.2f}% | Pre0: {pre0 * 100:.2f}% |  Pre1: {pre1 * 100:.2f}% | Rec0: {rec0 * 100:.2f}% | Rec1: {rec1 * 100:.2f}%  ')      
        logger.info(
                    f'\t Acc0: {train_acc0 * 100:.2f}% |  Acc1: {train_acc1 * 100:.2f}% | Pre0: {pre0 * 100:.2f}% |  Pre1: {pre1 * 100:.2f}% | Rec0: {rec0 * 100:.2f}% | Rec1: {rec1 * 100:.2f}%  ')
    
          
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                    os.path.join("model", savedir, 'epoch_{}_'.format(epoch + 1) + "checkpoint.pt"))
        end_time = time.monotonic()

        epoch_mins, epoch_secs = trainer.epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            logger.info("Early stopping")
            break

def fix_transfer():
    dataset = MixBench(root=ROOT)
    if LOAD:
        savedir = f"trl_{BENCH}"
    else:
        savedir = f"scr_{BENCH}"
    if not os.path.exists(os.path.join("model", savedir)):
        os.makedirs(os.path.join("model", savedir))
    if not os.path.exists(os.path.join("log", savedir)):
        os.makedirs(os.path.join("log", savedir))
    logging.basicConfig(filename=os.path.join("log", savedir, "train.log"), format='%(asctime)s %(message)s', filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    print("Savedir: ", savedir)
    logger.info("Savedir: " + savedir)

    D = dataset.shuffle()

    print("Target dataset size: " + str(len(D)))
    print("Split: \n  train -> {}".format(len(D)))
    print("Edges: " + str(GRAPHS.split('_')))
    logger.info("Target dataset size: " + str(len(D)))
    logger.info("Split: \n  train -> {}".format(len(D)))
    logger.info("Edges: " + str(GRAPHS.split('_')))

    train_loader = DataLoader(D, shuffle=True, batch_size=16, pin_memory=True)

    model = HeteroGNN(GRAPHS.split('_'))
    loss_function = BCELoss(reduction="sum").to(DEVICE)
    if LOAD:
        model.load_state_dict(torch.load(os.path.join("model", PRETRAINED)))
        print("Model ({}) is loaded.".format(PRETRAINED))
        logger.info("Model ({}) is loaded.".format(PRETRAINED))
        # for param in model.parameters():
        #     param.requires_grad = False
        model.classifier = torch.nn.Linear(in_features=100, out_features=1)
        
    
    optim = Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    model.to(DEVICE)

    data = next(iter(train_loader)).to(DEVICE)
    with torch.no_grad():  # Initialize lazy modules.
        model(data, DEVICE)

    logger.info("============> Training mode start...")

    for epoch in tqdm(range(0, 200)):
        start_time = time.monotonic()
        train_loss, train_acc, train_acc0, train_acc1  = trainer_train(model, train_loader, optim, DEVICE, loss_function, LABEL)
        print(f'\tTrain Loss: {train_loss:.6f} | Train Acc: {train_acc * 100:.3f}% | Acc0: {train_acc0 * 100:.2f}% |  Acc1: {train_acc1 * 100:.2f}% ')
        logger.info(f'\tTrain Loss: {train_loss:.6f} | Train Acc: {train_acc * 100:.3f}% | Acc0: {train_acc0 * 100:.2f}% |  Acc1: {train_acc1 * 100:.2f}% ')
        
          
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(),
                    os.path.join("model", savedir, 'epoch_{}_'.format(epoch + 1) + "checkpoint.pt"))
        end_time = time.monotonic()

        epoch_mins, epoch_secs = trainer.epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')


def main():
    """
    main function that executes tasks based on command-line options
    """
    torch.manual_seed(12345)
    np.random.seed(12345)
    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-train', '--train', action='store_true')
    parser.add_argument('-test', '--test', action='store_true')
    parser.add_argument('-data', '--data', action='store_true')
    parser.add_argument('-tr', '--transfer', action='store_true')
    parser.add_argument('-fixtr', '--fixtransfer', action='store_true')
    parser.add_argument("-b", "--batch", type=int, default=16,
                    help="The batch size")

    args = parser.parse_args()
    global BATCH
    if args.batch:
        BATCH = args.batch
        print("BATCH SIZE = ", BATCH)
    if args.data:
        dataset()
    if args.train:
        train()
    if args.test:
        test()
    if args.transfer:
        transfer()
    if args.fixtransfer:
        fix_transfer()



if __name__ == "__main__":
    main()

