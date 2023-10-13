import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, device, loss_function, label):
    epoch_loss = 0
    epoch_acc = 0
    matrix_yture = []
    matrix_ypred = []

    model.train()
    iteration = 0

    try:
        for data in tqdm(iterator, position=0, leave=True):
            model.zero_grad()
            optimizer.zero_grad()

            E = data.edge_index_dict
            for e in E:
                for i in range(len(E[e][0])):
                    e1 = E[e][0][i]
                    e2 = E[e][1][i]
                    if e1 >= data.num_nodes or e2 >= data.num_nodes:
                        E[e][0][i] = 0
                        E[e][1][i] = 0
            
            predictions = model(data, device)

            preds = (predictions > 0.5).float()
            if label == "runtime":
                Y = data.y.view(data.num_graphs, 2).T
                targets = Y[0].to(torch.float32)
            elif label == "error":
                Y = data.y.view(data.num_graphs, 2).T
                targets = Y[1].to(torch.float32)
            acc = (preds == targets).sum().float()
            acc = acc / preds.shape[0]
    
            batch_loss = loss_function(predictions, targets)
            for i in targets.cpu().numpy():
                matrix_yture.append(i)
            for i in preds.cpu().numpy():
                matrix_ypred.append(i)
            epoch_acc += acc.item()

            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.detach().cpu().item()
            iteration += 1
    except KeyboardInterrupt:
        print('Training Interrupted by user!')

    matrix = metrics.confusion_matrix(matrix_yture, matrix_ypred)
    print("Confusion_matrix: \n", matrix)
    train_acc0, train_acc1 = matrix.diagonal()/matrix.sum(axis=1)
    return epoch_loss / iteration, epoch_acc / iteration, train_acc0, train_acc1


def evaluate(model, iterator, device, loss_function, label):
    epoch_loss = 0
    epoch_acc = 0
    epoch_pre = 0
    epoch_rec = 0
    epoch_pre0 = 0
    epoch_pre1 = 0
    epoch_rec0 = 0
    epoch_rec1 = 0
    matrix_yture = []
    matrix_ypred = []
    model.eval()

    with torch.no_grad():
        iteration = 0
        try:
            for data in tqdm(iterator, position=0, leave=True):
                model.zero_grad()

                E = data.edge_index_dict
                for e in E:
                    for i in range(len(E[e][0])):
                        e1 = E[e][0][i]
                        e2 = E[e][1][i]
                        if e1 >= data.num_nodes or e2 >= data.num_nodes:
                            E[e][0][i] = 0
                            E[e][1][i] = 0
                            
                predictions = model(data, device)
                preds = (predictions > 0.5).float()

                if label == "runtime":
                        Y = data.y.view(data.num_graphs, 2).T
                        targets = Y[0].to(torch.float32)
                elif label == "error":
                        Y = data.y.view(data.num_graphs, 2).T
                        targets = Y[1].to(torch.float32)
                acc = (preds == targets).sum().float()
                acc = acc / preds.shape[0]

                precision, recall, fscore, support = metrics.precision_recall_fscore_support(targets.cpu().numpy(),
                                                                                             preds.cpu().numpy(),
                                                                                             average="macro"
                                                                                             )
                precision_lisr, recall_list, fscore_list, support = metrics.precision_recall_fscore_support(targets.cpu().numpy(),
                                                                                             preds.cpu().numpy(),
                                                                                             average=None,
                                                                                             labels=[0, 1])
                for i in targets.cpu().numpy():
                    matrix_yture.append(i)
                for i in preds.cpu().numpy():
                    matrix_ypred.append(i)
      
                batch_loss = loss_function(predictions, targets)
                    
                epoch_acc += acc.item()
                epoch_pre += precision.item()
                epoch_rec += recall.item()
                epoch_pre0 += precision_lisr[0]
                epoch_pre1 += precision_lisr[1]
                epoch_rec0 += recall_list[0]
                epoch_rec1 += recall_list[1]

                epoch_loss += batch_loss.detach().cpu().item()
                iteration += 1
        except KeyboardInterrupt:
            print('Evaluation Interrupted by user!')

        matrix = metrics.confusion_matrix(matrix_yture, matrix_ypred)
        print("Confusion_matrix: \n", matrix)
        train_acc0, train_acc1 = matrix.diagonal()/matrix.sum(axis=1)

        epoch_fsc = 2 * (epoch_pre * epoch_rec) / (epoch_pre + epoch_rec)
        return epoch_loss / iteration, epoch_acc / iteration, epoch_pre / iteration, epoch_rec / iteration, epoch_fsc / iteration, \
                epoch_pre0 / iteration, epoch_pre1 / iteration, epoch_rec0 / iteration, epoch_rec1 / iteration, train_acc0, train_acc1
