import sys, os, json, math, logging, time, subprocess
from src import utilities as utilities
import pandas as pd
import numpy as np
import pickle

import torch
from torch_geometric.loader import DataLoader
from src import model

search_counter = 0
HIGHEST =-1
LOWEST = 0
TIME_ERROR = 0 # 0.05
benchmark = sys.argv[1] # in lower case
TIME_OUT = int(sys.argv[4])
EPSILON = int(sys.argv[5])
CLASS = sys.argv[6]
RESULTSDIR = f"results-eps={EPSILON}"

EXE_COUNT = 1
PRE_CHECK_TIME_OUT = 0 ##2
PROB_SIZE1 = "2000"
PROB_SIZE2 = "../scripts/reference.dat"
PROB_SIZE3 = "1"
PROB_SIZE4 = "1"
PROB_SIZE5 = "../scripts/200_200_260_ldc.of"
benchfile = "lbm.c"
TO_RUN = "../tempscripts/lbm_s"


original_runtime = 0
glob_min_score = -1
glob_search_conf = -1
glob_min_idx = 0

# dataframe to store all the results
df = pd.DataFrame()

# model prediction related
GRAPHS = "AST_CFG_PDG_CAST_DEP"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
V = torch.load("../src/vocab-2.pt")
embedding = torch.load("../src/emb-2.pt")
dictionary = V.get_stoi()

time_predictor = model.HeteroGNN(GRAPHS.split('_'))
time_predictor.load_state_dict(torch.load("../src/time.pt"))
print("Runtime time_predictor {} is loaded.".format("../src/time.pt"))
time_predictor.to(DEVICE)
time_predictor.eval()

error_predictor = model.HeteroGNN(GRAPHS.split('_'))
error_predictor.load_state_dict(torch.load("../src/error.pt"))
print("Error error_predictor {} is loaded.".format("../src/error.pt"))
error_predictor.to(DEVICE)
error_predictor.eval()

pd_configs = pd.read_csv('../src/df-configs.csv')
edges_csv = f"../src/edges.csv"
nodes_csv = f"../src/nodes.csv"
with open(f"../src/rootnode.txt", "r") as f:
    ROOTNODE = int(f.readline())
    print(f"Rootnode is {ROOTNODE}.")
print("One time preloading...")
data0 = torch.load(f'../src/data_0.pt')
with open(f"../src/nameLineToIdx.pkl", "rb") as f:
    nameLineToIdx = pickle.load(f)
nodedf = pd.read_csv(nodes_csv)
edgedf = pd.read_csv(edges_csv)

trainset_counter = 0
trainset_size = 80

newedge = torch.tensor([[],[]])
tree = {}
upcasting = {':START_ID': [],
                ':END_ID': [],
                ':TYPE': [],
                ':VARIABLE': []}
downcasting = {':START_ID': [],
                ':END_ID': [],
                ':TYPE': [],
                ':VARIABLE': []}

def createTreeFromEdges(edges):
    tree = {}
    for v1, v2 in edges:
        tree.setdefault(v1, []).append(v2)
    return tree


def postTraverse(node):
    global newedge
    global tree
    if node or (node == 0):
        # if node is not leaf
        if node in tree.keys():
            for child in tree[node]:
                postTraverse(child)

        if (nodedf[':ID'] == node).any():
            row = nodedf.loc[nodedf[':ID'] == node]
            name = str(row[':NAME'].values[0])
            label = str(row[':LABEL'].values[0])
            if 'CALL' in label and ('exp' in name or 'log' in name or 'sin' in name
                                    or 'tan' in name or 'cos' in name or 'sqrt' in name
                                    or 'abs' in name or 'pow' in name or 'floor' in name
                                    or 'ceil' in name or 'mod' in name) and (not name.endswith('f')):
                if not node in tree.keys(): return
                child = tree[node][0]
                if (nodedf[':ID'] == child).any():
                    child_type = str(nodedf.loc[nodedf[':ID'] == child][':DATATYPE'].values[0])
                    if 'float' in child_type:
                        newedge = torch.cat((newedge, torch.tensor([[child, node], [node, child]])), 1).float()
            elif 'CALL' in label and ('exp' in name or 'log' in name or 'sin' in name
                                    or 'tan' in name or 'cos' in name or 'sqrt' in name
                                    or 'abs' in name or 'pow' in name or 'floor' in name
                                    or 'ceil' in name or 'mod' in name) and name.endswith('f'):
                if not node in tree.keys(): return
                child = tree[node][0]
                if (nodedf[':ID'] == child).any():
                    child_type = str(nodedf.loc[nodedf[':ID'] == child][':DATATYPE'].values[0])
                    if 'double' in child_type:
                        newedge = torch.cat((newedge, torch.tensor([[child, node], [node, child]])), 1).float()

            if '<operator>' in name:
                if not node in tree.keys(): return
                if len(tree[node]) == 2:
                    left_child = tree[node][0]
                    right_child = tree[node][1]
                    if not (nodedf[':ID'] == left_child).any() or not (nodedf[':ID'] == right_child).any():
                        return
                    left_child_label = str(nodedf.loc[nodedf[':ID'] == left_child][':LABEL'].values[0])
                    right_child_label = str(nodedf.loc[nodedf[':ID'] == right_child][':LABEL'].values[0])

                    if 'indirectIndexAccess' not in name:
                        if ('IDENTIFIER' in left_child_label or 'CALL' in left_child_label ) \
                                and ('IDENTIFIER' in right_child_label or 'CALL' in right_child_label ):
                            left_child_type = str(nodedf.loc[nodedf[':ID'] == left_child][':DATATYPE'].values[0])
                            right_child_type = str(nodedf.loc[nodedf[':ID'] == right_child][':DATATYPE'].values[0])
                            if 'float' in left_child_type and 'double' in right_child_type:
                                if 'assignment' in name:
                                    newedge = torch.cat((newedge, torch.tensor([[right_child, node], [node, right_child]])), 1).float()
                                else:                        
                                    newedge = torch.cat((newedge, torch.tensor([[left_child, node], [node, left_child]])), 1).float()
                            elif 'double' in left_child_type and 'float' in right_child_type:
                                newedge = torch.cat((newedge, torch.tensor([[right_child, node], [node, right_child]])), 1).float()


def getPrediction(data):
  time_predictor.eval()
  error_predictor.eval()
  with torch.no_grad():
    time_predictor.zero_grad()
    error_predictor.zero_grad()
    E = data.edge_index_dict
    for e in E:
      for i in range(len(E[e][0])):
        e1 = E[e][0][i]
        e2 = E[e][1][i]
        if e1 >= data.num_nodes or e2 >= data.num_nodes:
          E[e][0][i] = 0
          E[e][1][i] = 0
                            
    time_predictions = time_predictor(data, DEVICE)
    time_preds = (time_predictions > 0.5).float() # finetune: 0.5 scratch: 0.55 
    error_predictions = error_predictor(data, DEVICE)
    error_preds = (error_predictions > 0.4).float() # finetune: 0.4 scratch: 0.545
    torch.cuda.empty_cache()

    return time_preds, time_predictions, error_preds, error_predictions


def not_run_orig_config(search_config, original_config, benchmark):
  global search_counter
  global original_runtime
  global df
  global trainset_counter

  config_info_dict = {}

  # to record for each searched config
  config_info_dict["index"] = search_counter
  config_info_dict["benchmark"] = benchmark
  
  row = pd_configs.loc[pd_configs['Unnamed: 0'] == trainset_counter]

  config_info_dict["label"] = "VALID"
  print("cp config.json {}/{}.json".format(RESULTSDIR, "VALID_config_" + benchmark + "_" + str(search_counter)))
  os.system("cp config.json {}/{}.json".format(RESULTSDIR, "VALID_config_" + benchmark + "_" + str(search_counter)))
  
  avg_runtime = row['runtime'].item()
  with open('./time.txt', "w") as f:
     f.write(str(avg_runtime))

  config_info_dict["runtime"] = avg_runtime
  config_info_dict["runtime_label"] = 0
  config_info_dict["error"] = row['error'].item()
  config_info_dict["error_label"] = row['error_label'].item()
  config_info_dict["runtime_pred"] = -1
  config_info_dict["error_pred"] = -1

  df_item = pd.DataFrame(config_info_dict, index=[0])
  df = df.append(df_item, ignore_index = True)
  df.to_csv(RESULTSDIR+'/df-configs.csv')

  search_counter = search_counter + 1
  trainset_counter = trainset_counter + 1
  return 1

def run_orig_config(search_config, original_config, benchmark):
  global search_counter
  global original_runtime
  global df
  global trainset_counter

  config_info_dict = {}

  # to record for each searched config
  config_info_dict["index"] = search_counter
  config_info_dict["benchmark"] = benchmark

  # step 1: generate "config_temp.json" according to search_config
  json_string = json.dumps(search_config, indent=4)
  with open('config_temp.json', 'w') as outfile:
    outfile.write(json_string)
    print("-------- running config {} --------".format(search_counter))
    logging.info("-------- running config {} --------".format(search_counter))

  # (compare search_config with original_config)
  # step 2: transform benchmark(cg.c) with "config_temp.json"

  os.system("rm ../tempscripts/{}".format(benchfile))
  os.system("python3 trans-type.py config_temp.json")
  if not os.path.exists("../tempscripts/{}".format(benchfile)):
    os.system("cp ../scripts/{} ../tempscripts/".format(benchfile))
  # the thing you need to do before compiling
  targetfile = os.path.join("../tempscripts", benchfile)
  replacetxt = "../src/replacement.txt"
  utilities.comment_before_make(targetfile, replacetxt)


  # delete previous log files and executable
  os.system(f"rm -f log.txt time.txt {TO_RUN}")

  # step 3: recompile modified cg.c into executable
  os.system("cd ../tempscripts; specmake clean; specmake")

  # step 4: run executable
  if os.path.exists(TO_RUN):
    print("Round 1:")
    process = subprocess.Popen([TO_RUN, PROB_SIZE1, PROB_SIZE2, PROB_SIZE3, PROB_SIZE4, PROB_SIZE5])
    try:
      process.wait(timeout=TIME_OUT)
    except subprocess.TimeoutExpired:
      print('Timed out - killing', process.pid)
      process.kill()

  # os.system("./temp-NPB3.3-SER-C/bin/{}.A.x".format(benchmark))
  # step 5: check if the computation result is with error threshold
  # step 6: record the configuration, mark valid or invalid
  result = utilities.check_error()
  runtime = utilities.get_dynamic_score()
  print(f"runtime: {runtime}")
  logging.info(f"runtime: {runtime}")
  if result == 1:
    config_info_dict["label"] = "VALID"
    print("mv config_temp.json {}/{}.json".format(RESULTSDIR, "VALID_config_" + benchmark + "_" + str(search_counter)))
    os.system("mv config_temp.json {}/{}.json".format(RESULTSDIR, "VALID_config_" + benchmark + "_" + str(search_counter)))
    # run EXE_COUNT times in total to get average runtime
    avg_runtime = 0
    avg_runtime += runtime
    for i in range(EXE_COUNT - 1):
      if os.path.exists(TO_RUN):
        print("Round {}:".format(i+2))
        process = subprocess.Popen([TO_RUN, PROB_SIZE1, PROB_SIZE2, PROB_SIZE3, PROB_SIZE4, PROB_SIZE5])
        try:
          process.wait(timeout=TIME_OUT)
        except subprocess.TimeoutExpired:
          print('Timed out - killing', process.pid)
          process.kill()
      runtime = utilities.get_dynamic_score()
      print(f"runtime: {runtime}")
      logging.info(f"runtime: {runtime}")
      avg_runtime += runtime
    avg_runtime = avg_runtime / EXE_COUNT
    with open('./time.txt', "w") as f:
      f.write(str(avg_runtime))
    runtime = avg_runtime
  elif result == 0:
    config_info_dict["label"] = "INVALID"
    print("mv config_temp.json {}/{}.json".format(RESULTSDIR, "INVALID_config_" + benchmark + "_" + str(search_counter)))
    os.system("mv config_temp.json {}/{}.json".format(RESULTSDIR, "INVALID_config_" + benchmark + "_" + str(search_counter)))
  elif result == -1:
    config_info_dict["label"] = "FAIL"
    print("mv config_temp.json {}/{}.json".format(RESULTSDIR, "FAIL_config_" + benchmark + "_" + str(search_counter)))
    os.system("mv config_temp.json {}/{}.json".format(RESULTSDIR, "FAIL_config_" + benchmark + "_" + str(search_counter)))

  time_to_log = utilities.get_dynamic_score()
  logging.info("Average runtime = {} s".format(time_to_log))
  print("Average runtime = {} s".format(runtime))

  config_info_dict["runtime"] = runtime
  if runtime >= original_runtime:
    config_info_dict["runtime_label"] = 0
  else:
    config_info_dict["runtime_label"] = 1

  if os.path.isfile("log.txt"):
    with open("log.txt") as f:
      firstline = f.readline().rstrip()
      err = f.readline().rstrip()

    config_info_dict["error"] = err
    config_info_dict["error_label"] = 1 if firstline == "true" else 0
  else:
    config_info_dict["error"] = np.nan
    config_info_dict["error_label"] = 0
  
  config_info_dict["runtime_pred"] = -1
  config_info_dict["error_pred"] = -1

  df_item = pd.DataFrame(config_info_dict, index=[0])
  df = df.append(df_item, ignore_index = True)
  df.to_csv(RESULTSDIR+'/df-configs.csv')

  search_counter = search_counter + 1
  trainset_counter = trainset_counter + 1
  return result


def run_config(search_config, original_config, benchmark):
  global search_counter
  global original_runtime
  global df
  global trainset_counter
  global trainset_size
  global newedge
  global tree
  global upcasting
  global downcasting

  config_info_dict = {}

  # to record for each searched config
  config_info_dict["index"] = search_counter
  config_info_dict["benchmark"] = benchmark

  # step 1: generate "config_temp.json" according to search_config
  json_string = json.dumps(search_config, indent=4)
  with open('config_temp.json', 'w') as outfile:
    outfile.write(json_string)
    print("-------- running config {} --------".format(search_counter))
    logging.info("-------- running config {} --------".format(search_counter))
    # print(json_string)

  if trainset_counter < trainset_size:
        row = pd_configs.loc[pd_configs['Unnamed: 0'] == trainset_counter]
        error = row['error_label']
        runtime = row['runtime_label']
        if error.item() == 1:
              print("mv config_temp.json {}/{}.json".format(RESULTSDIR, "VALID_config_" + benchmark + "_" + str(search_counter)))
              os.system("mv config_temp.json {}/{}.json".format(RESULTSDIR, "VALID_config_" + benchmark + "_" + str(search_counter)))
              avg_runtime = row['runtime'].item()
              with open('./time.txt', "w") as f:
                f.write(str(avg_runtime))

              config_info_dict["label"] = "VALID"
              config_info_dict["runtime"] = avg_runtime
              config_info_dict["runtime_label"] = runtime.item()
              config_info_dict["error"] = row['error'].item()
              config_info_dict["error_label"] = error.item()
              config_info_dict["runtime_pred"] = -1
              config_info_dict["error_pred"] = -1
              config_info_dict["runtime_score"] = -1
              config_info_dict["error_score"] = -1
              df_item = pd.DataFrame(config_info_dict, index=[0])
              df = df.append(df_item, ignore_index = True)
              df.to_csv(RESULTSDIR+'/df-configs.csv')

              search_counter = search_counter + 1
              trainset_counter = trainset_counter + 1
              return 1
        else:
              tmplabel = row['label'].item()
              print("mv config_temp.json {}/{}.json".format(RESULTSDIR, f"{tmplabel}_config_" + benchmark + "_" + str(search_counter)))
              os.system("mv config_temp.json {}/{}.json".format(RESULTSDIR, f"{tmplabel}_config_" + benchmark + "_" + str(search_counter)))
              
              config_info_dict["label"] = tmplabel
              config_info_dict["runtime"] = row['runtime'].item()
              config_info_dict["runtime_label"] = runtime.item()
              config_info_dict["error"] = row['error'].item()
              config_info_dict["error_label"] = error.item()
              config_info_dict["runtime_pred"] = -1
              config_info_dict["error_pred"] = -1
              config_info_dict["runtime_score"] = -1
              config_info_dict["error_score"] = -1
              df_item = pd.DataFrame(config_info_dict, index=[0])
              df = df.append(df_item, ignore_index = True)
              df.to_csv(RESULTSDIR+'/df-configs.csv')
              
              search_counter = search_counter + 1
              trainset_counter = trainset_counter + 1
              return 0

  # (compare search_config with original_config)
  # step 2: transform benchmark(cg.c) with "config_temp.json"
  # os.system("python3 setup.py {}".format(benchmark))
  os.system("rm ../tempscripts/{}".format(benchfile))
  os.system("python3 trans-type.py config_temp.json")
  if not os.path.exists("../tempscripts/{}".format(benchfile)):
    os.system("cp ../scripts/{} ../tempscripts/".format(benchfile))
  # the thing you need to do before compiling
  targetfile = os.path.join("../tempscripts", benchfile)
  replacetxt = "../src/replacement.txt"
  utilities.comment_before_make(targetfile, replacetxt)

  # delete previous log files and executable
  os.system(f"rm -f log.txt time.txt {TO_RUN}")

  # step 3: recompile modified cg.c into executable
  os.system("cd ../tempscripts; specmake clean; specmake")

  # step 4: run executable
  finished_flag = 0
  # if os.path.exists(TO_RUN):
  #   print("Pre check:")
  #   process = subprocess.Popen([TO_RUN, PROB_SIZE1, PROB_SIZE2, PROB_SIZE3, PROB_SIZE4, PROB_SIZE5])
  #   try:
  #     process.wait(timeout=PRE_CHECK_TIME_OUT)
  #   except subprocess.TimeoutExpired:
  #     print('Timed out - killing', process.pid)
  #     process.kill()
  #     finished_flag = 0
  if finished_flag:
    config_info_dict["label"] = "FAIL"
    config_info_dict["runtime"] = 9999
    config_info_dict["runtime_label"] = 0
    config_info_dict["error"] = np.nan
    config_info_dict["error_label"] = 0
    config_info_dict["runtime_pred"] = -1
    config_info_dict["error_pred"] = -1
    config_info_dict["runtime_score"] = -1
    config_info_dict["error_score"] = -1
    result = -1
    print("mv config_temp.json {}/{}.json".format(RESULTSDIR, "FAIL_config_" + benchmark + "_" + str(search_counter)))
    os.system("mv config_temp.json {}/{}.json".format(RESULTSDIR, "FAIL_config_" + benchmark + "_" + str(search_counter)))
    df_item = pd.DataFrame(config_info_dict, index=[0])
    df = df.append(df_item, ignore_index = True)
    df.to_csv(RESULTSDIR+'/df-configs.csv')

    search_counter = search_counter + 1
    return result

  #######################################################
  ####### cerate data
  print("Looking for indices to change...")
  with open("config_temp.json") as f:
      configjson = json.load(f)
  nameline_pair_to_tune = []
  idx_to_tune = []
  for item in configjson:
      if "float" in configjson[item]["type"]:
          name =  configjson[item]["name"]
          for l in configjson[item]["lines"]:
              nameline_pair_to_tune.append((name, l))
              if (name, l) in nameLineToIdx:
                  for idx in nameLineToIdx[(name, l)]:
                      idx_to_tune.append(idx)
  # print("1. Variable's (name, line) pair: \n", nameline_pair_to_tune)
  # print("2. Corresponding index: \n", idx_to_tune)

  print("Changing data object according to configuration...")
  print("1. Updating data types...")
  newdata = data0.clone()

  for idx in idx_to_tune:
      word_vector = []
      encoded_sentence = []
      kind = nodedf.at[idx, ":LABEL"]
      dtype = "float"
      name = nodedf.at[idx, ":NAME"]
      nodedf.at[idx, ":DATATYPE"] = "float"

      if kind in dictionary:
          encoded_sentence.append(dictionary[kind])
      else:
          encoded_sentence.append(dictionary['<pad>'])
      encoded_sentence.append(dictionary[dtype])
      if kind == 'CALL' and name in dictionary:
          encoded_sentence.append(dictionary[name])
            # else:
            #     encoded_sentence.append(dictionary['<pad>'])
      word_vector = embedding(torch.LongTensor(encoded_sentence))
      word_vector = torch.mean(word_vector, 0)

      newdata["node"].x[idx] = word_vector

  print("2. Updating casting edges...")

  newedge = torch.tensor([[],[]])
  tree = {}
  upcasting = {':START_ID': [],
                ':END_ID': [],
                ':TYPE': [],
                ':VARIABLE': []}
  downcasting = {':START_ID': [],
                ':END_ID': [],
                ':TYPE': [],
                ':VARIABLE': []}

  edge_list = edgedf[(edgedf[':TYPE'] == 'AST') | (edgedf[':TYPE'] == 'CONTAINS')]
  edge_list = edge_list[[":START_ID", ":END_ID"]].values.tolist()
  tree = createTreeFromEdges(edge_list)
  postTraverse(ROOTNODE)
  newdata["node", "CAST", "node"]['edge_index'] = newedge

  ####### make prediction
  time_preds, time_predictions, error_preds, error_predictions = getPrediction(newdata)
  print(f"Runtime prediction: {time_preds}")
  logging.info(f"Runtime prediction: {time_preds}")
  print(f"Error prediction: {error_preds}")
  logging.info(f"Error prediction: {error_preds}")
  config_info_dict["runtime_score"] = float(time_predictions)
  config_info_dict["error_score"] = float(error_predictions)
  #######################################################
 
  # if preds == 1:
  #   run
  # else:
  #   save config_info_dict
  
  if time_preds == 0:
    config_info_dict["label"] = "FAIL"
    config_info_dict["runtime"] = -1
    config_info_dict["runtime_label"] = -1
    config_info_dict["error"] = -1
    config_info_dict["error_label"] = -1
    config_info_dict["runtime_pred"] = time_preds.item()
    config_info_dict["error_pred"] = error_preds.item()
    result = -1
    print("mv config_temp.json {}/{}.json".format(RESULTSDIR, "FAIL_config_" + benchmark + "_" + str(search_counter)))
    os.system("mv config_temp.json {}/{}.json".format(RESULTSDIR, "FAIL_config_" + benchmark + "_" + str(search_counter)))

  elif error_preds == 0:
    config_info_dict["label"] = "INVALID"
    config_info_dict["runtime"] = -1
    config_info_dict["runtime_label"] = -1
    config_info_dict["error"] = -1
    config_info_dict["error_label"] = -1
    config_info_dict["runtime_pred"] = time_preds.item()
    config_info_dict["error_pred"] = error_preds.item()
    result = -1
    print("mv config_temp.json {}/{}.json".format(RESULTSDIR, "INVALID_config_" + benchmark + "_" + str(search_counter)))
    os.system("mv config_temp.json {}/{}.json".format(RESULTSDIR, "INVALID_config_" + benchmark + "_" + str(search_counter)))

  elif time_preds == 1 and error_preds == 1:
    # # delete previous log files and executable
    # os.system(f"rm log.txt time.txt {TO_RUN}")

    # # step 3: recompile modified cg.c into executable
    # os.system("cd ../tempscripts; rm -r build; mkdir build; cd build; cmake -DCMAKE_BUILD_TYPE=Release -DMPI_CXX_COMPILER=`which mpicxx` ..; make")

    # step 4: run executable
    if os.path.exists(TO_RUN):
      print("Round 1:")
      process = subprocess.Popen([TO_RUN, PROB_SIZE1, PROB_SIZE2, PROB_SIZE3, PROB_SIZE4, PROB_SIZE5])
      try:
        process.wait(timeout=TIME_OUT)
      except subprocess.TimeoutExpired:
        print('Timed out - killing', process.pid)
        process.kill()

    # os.system("./temp-NPB3.3-SER-C/bin/{}.A.x".format(benchmark))
    # step 5: check if the computation result is with error threshold
    # step 6: record the configuration, mark valid or invalid
    result = utilities.check_error()
    runtime = utilities.get_dynamic_score()
    print(f"runtime: {runtime}")
    logging.info(f"runtime: {runtime}")
    if result == 1:
      config_info_dict["label"] = "VALID"
      print("mv config_temp.json {}/{}.json".format(RESULTSDIR, "VALID_config_" + benchmark + "_" + str(search_counter)))
      os.system("mv config_temp.json {}/{}.json".format(RESULTSDIR, "VALID_config_" + benchmark + "_" + str(search_counter)))
      # run EXE_COUNT times in total to get average runtime
      avg_runtime = 0
      avg_runtime += runtime
      
      if runtime <= 1000: 
        for i in range(EXE_COUNT - 1):
          if os.path.exists(TO_RUN):
            print("Round {}:".format(i+2))
            process = subprocess.Popen([TO_RUN, PROB_SIZE1, PROB_SIZE2, PROB_SIZE3, PROB_SIZE4, PROB_SIZE5])
            try:
              process.wait(timeout=TIME_OUT)
            except subprocess.TimeoutExpired:
              print('Timed out - killing', process.pid)
              process.kill()
          runtime = utilities.get_dynamic_score()
          print(f"runtime: {runtime}")
          logging.info(f"runtime: {runtime}")
          avg_runtime += runtime
        avg_runtime = avg_runtime / EXE_COUNT
      with open('./time.txt', "w") as f:
        f.write(str(avg_runtime))
      runtime = avg_runtime
    elif result == 0:
      config_info_dict["label"] = "INVALID"
      print("mv config_temp.json {}/{}.json".format(RESULTSDIR, "INVALID_config_" + benchmark + "_" + str(search_counter)))
      os.system("mv config_temp.json {}/{}.json".format(RESULTSDIR, "INVALID_config_" + benchmark + "_" + str(search_counter)))
    elif result == -1:
      config_info_dict["label"] = "FAIL"
      print("mv config_temp.json {}/{}.json".format(RESULTSDIR, "FAIL_config_" + benchmark + "_" + str(search_counter)))
      os.system("mv config_temp.json {}/{}.json".format(RESULTSDIR, "FAIL_config_" + benchmark + "_" + str(search_counter)))

    time_to_log = utilities.get_dynamic_score()
    logging.info("Average runtime = {} s".format(time_to_log))
    print("Average runtime = {} s".format(runtime))

    config_info_dict["runtime"] = runtime
    if runtime >= original_runtime:
      config_info_dict["runtime_label"] = 0
    else:
      config_info_dict["runtime_label"] = 1

    if os.path.isfile("log.txt"):
      with open("log.txt") as f:
        firstline = f.readline().rstrip()
        err = f.readline().rstrip()
      config_info_dict["error"] = err
      config_info_dict["error_label"] = 1 if firstline == "true" else 0
    else:
      config_info_dict["error"] = np.nan
      config_info_dict["error_label"] = 0
    
    config_info_dict["runtime_pred"] = time_preds.item()
    config_info_dict["error_pred"] = error_preds.item()


  df_item = pd.DataFrame(config_info_dict, index=[0])
  df = df.append(df_item, ignore_index = True)
  df.to_csv(RESULTSDIR+'/df-configs.csv')

  search_counter = search_counter + 1
  return result


def dd_search_config(change_set, type_set, switch_set, search_config, original_config, benchmark, div, original_score):
  #
  # partition change_set into deltas and delta inverses
  #
  global glob_min_score
  global glob_search_conf
  global glob_min_idx
  
  delta_change_set = []
  delta_type_set = []
  delta_switch_set = []
  delta_inv_change_set = []
  delta_inv_type_set = []
  delta_inv_switch_set = []
  div_size = int(math.ceil(float(len(change_set))/float(div)))
  for i in range(0, len(change_set), div_size):
    delta_change = []
    delta_type = []
    delta_switch = []
    delta_inv_change = []
    delta_inv_type = []
    delta_inv_switch = []
    for j in range(0, len(change_set)):
      if j >= i and j < i+div_size:
        delta_change.append(change_set[j])
        delta_type.append(type_set[j])
        delta_switch.append(switch_set[j])
      else:
        delta_inv_change.append(change_set[j])
        delta_inv_type.append(type_set[j])
        delta_inv_switch.append(switch_set[j])
    delta_change_set.append(delta_change)
    delta_type_set.append(delta_type)
    delta_switch_set.append(delta_switch)
    delta_inv_change_set.append(delta_inv_change)
    delta_inv_type_set.append(delta_inv_type)
    delta_inv_switch_set.append(delta_inv_switch)

  #
  # iterate through all delta and inverse delta set
  # record delta set that passes
  #
  pass_inx = -1
  inv_is_better = False
  min_score = -1

  for i in range(0, len(delta_change_set)):
    delta_change = delta_change_set[i]
    delta_type = delta_type_set[i]
    delta_switch = delta_switch_set[i]
    if len(delta_change) > 0:
      # always reset to lowest precision
      utilities.to_2nd_highest_precision(change_set, type_set, switch_set)
      # apply change for variables in delta
      utilities.to_highest_precision(delta_change, delta_type, switch_set)
      # record i if config passes
      if run_config(search_config, original_config, benchmark) == 1 and utilities.get_dynamic_score() < original_score:
        score = utilities.get_dynamic_score()
        if score < glob_min_score or glob_min_score == -1:
          pass_inx = i
          inv_is_better = False
          # min_score = score
          glob_min_score = score
          glob_search_conf = search_config
          glob_min_idx = search_counter - 1

    delta_inv_change = delta_inv_change_set[i]
    delta_inv_type = delta_inv_type_set[i]
    delta_inv_switch = delta_inv_switch_set[i]
    if len(delta_inv_change) > 0 and div > 2:
      # always reset to lowest precision
      utilities.to_2nd_highest_precision(change_set, type_set, switch_set)
      # apply change for variables in delta inverse
      utilities.to_highest_precision(delta_inv_change, delta_inv_type, delta_inv_switch)
      # record i if config passes
      if run_config(search_config, original_config, benchmark) == 1 and utilities.get_dynamic_score() < original_score:
        score = utilities.get_dynamic_score()
        if score < glob_min_score or glob_min_score == -1:
          pass_inx = i
          inv_is_better = True
          # min_score = score
          glob_min_score = score
          glob_search_conf = search_config
          glob_min_idx = search_counter - 1

  #
  # recursively search in pass delta or pass delta inverse
  # right now keep searching for the first pass delta or
  # pass delta inverse; later on we will integrate cost
  # model here
  #
  if pass_inx != -1:
    pass_change_set = delta_inv_change_set[pass_inx] if inv_is_better else delta_change_set[pass_inx]
    pass_type_set = delta_inv_type_set[pass_inx] if inv_is_better else delta_type_set[pass_inx]
    pass_switch_set = delta_inv_switch_set[pass_inx] if inv_is_better else delta_switch_set[pass_inx]
    # log the configuration if it is faster than original_score
    # to_2nd_highest_precision(change_set, type_set)
    # to_highest_precision(pass_change_set, pass_type_set)
    # run_config(search_config, original_config, bitcode)
    # modified_score = utilities.get_dynamic_score()
    # if modified_score <= original_score:
    # utilities.log_fast_config("fast_configs.cov", search_counter-1, modified_score)

    if len(pass_change_set) > 1:
      # always reset to lowest precision
      utilities.to_2nd_highest_precision(change_set, type_set, switch_set)
      dd_search_config(pass_change_set, pass_type_set, pass_switch_set, search_config, original_config, benchmark, 2, original_score)
    else:
      utilities.to_2nd_highest_precision(change_set, type_set, switch_set)
      utilities.to_highest_precision(pass_change_set, pass_type_set, pass_switch_set)
    return

  #
  # stop searching when division greater than change set size
  #
  if div >= len(change_set):
    utilities.to_highest_precision(change_set, type_set, switch_set)
    # log the configuration if it is faster than original_score
    # run_config(search_config, original_config, bitcode)
    # modified_score = utilities.get_dynamic_score()
    # if modified_score <= original_score:
    # utilities.log_fast_config("fast_configs.cov", search_counter-1, modified_score)
    return
  else:
    dd_search_config(change_set, type_set, switch_set, search_config, original_config, benchmark, 2*div, original_score)
    return


def search_config(change_set, type_set, switch_set, search_config, original_config, benchmark, original_score):
  # search from bottom up
  utilities.to_2nd_highest_precision(change_set, type_set, switch_set)
  if run_config(search_config, original_config, benchmark) != 1 or utilities.get_dynamic_score() > original_score:
    dd_search_config(change_set, type_set, switch_set, search_config, original_config, benchmark, 2, original_score)
  # remove types and switches that cannot be changed
  for i in range(0, len(change_set)):
    if len(type_set[i]) > 0 and change_set[i]["type"] == type_set[i][HIGHEST]:
      del(type_set[i][:])
      if len(switch_set[i]) > 0:
        del(switch_set[i][:])

  # remove highest precision from each type vector
  for i in range(0, len(type_set)):
    type_vector = type_set[i]
    switch_vector = switch_set[i]
    if len(type_vector) > 0:
      type_vector.pop()
    if len(switch_vector) > 0:
      switch_vector.pop()
  return
  

def main():
  global benchmark
  search_conf_file = sys.argv[2]
  original_conf_file = sys.argv[3]

  if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)

  logging.basicConfig(filename='{}/dd2-{}.log'.format(RESULTSDIR, time.strftime("%Y%m%d-%H%M%S")), level=logging.DEBUG)

  #
  # delete log file if exists
  #
  try:
    os.remove("log.dd")
  except OSError:
    pass

  #
  # parsing config files
  #
  search_conf = json.loads(open(search_conf_file, 'r').read())
  search_changes = search_conf
  original_conf = json.loads(open(original_conf_file, 'r').read())
  change_set = []
  type_set = []
  switch_set = []

  #
  # record the change set
  #
  for search_change in search_changes:
    type_vector = search_changes[search_change]["type"]
    if isinstance(type_vector, list):
      type_set.append(type_vector)
      change_set.append(search_changes[search_change])
    if "call" in search_change:
      switch_set.append(search_changes[search_change]["switch"])
    else:
      switch_set.append([])

  # search for valid configuration
  print("** Searching for valid configuration using delta-debugging algorithm")
  logging.info("** Searching for valid configuration using delta-debugging algorithm")


  # get original score
  utilities.to_highest_precision(change_set, type_set, switch_set)
  not_run_orig_config(search_conf, original_conf, benchmark)
  global original_runtime
  original_runtime = utilities.get_dynamic_score()
  original_score = original_runtime * (1 + TIME_ERROR) # allow 5% difference

  # global original_runtime
  # original_runtime = original_score

  # keep searching while the type set is not searched throughout
  while not utilities.is_empty(type_set):
    search_config(change_set, type_set, switch_set, search_conf, original_conf, benchmark, original_score)
  print(glob_search_conf, glob_min_score, glob_min_idx)
  # get the score of modified program
  if glob_min_score != -1:
    logging.info("Check dd2_valid_" + benchmark + ".json for the valid configuration file")
    print("Check dd2_valid_" + benchmark + ".json for the valid configuration file")
    json_string = json.dumps(glob_search_conf, indent=4)
    with open('{}/dd2_valid_{}_{}.json'.format(RESULTSDIR, benchmark, str(glob_min_idx)), 'w') as outfile:
      outfile.write(json_string)
    with open('{}/best_speedup_{}_{}.txt'.format(RESULTSDIR, benchmark, str(glob_min_idx)), 'w') as f:
      speedup = (((original_runtime/(1 + TIME_ERROR)) - glob_min_score) / (original_runtime/(1 + TIME_ERROR))) + 1
      print(speedup, file=f)
  else:
    logging.info("No valid and speedup configuration file!")
    print("No valid and speedup configuration file!")

  df.to_csv(RESULTSDIR+'/df-configs.csv')



if __name__ == "__main__":
  main()

