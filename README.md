# FPLearner

This is the artifact of the paper: **Predicting Performance and Accuracy of Mixed-Precision Programs for Precision Tuning**.
The artifact is intended to provide the users with raw data descriptions and a set of guidelines to reproduce experiment results.

The first part of our artifact, the raw data description, details the `MixBench` dataset used to train and test our prediction models, which were created from the MixBench programs. In addition, we provide a description of the running results from the model's training, testing, and case studies.

In the second part of our artifact, instructions are provided to play with our data, get statistics of our dataset, reproduce the testing scores of our pre-trained models on the dataset, as well as an optional step to train the prediction models from scratch. Besides, we also present the running commands to reconstruct the case studies on the four target benchmarks: `CG`, `MG`, `Lulesh`, and `LBM`.

## Overview of Contents

- [1. Raw Data Description](#1-raw-data-description)
  - [1.1 Dataset for model training](#11-dataset-for-model-training)
  - [1.2 Raw data from model training](#12-raw-data-from-model-training)
    - [1.2.1 Pre-trained model description](#121-pre-trained-model-description)
    - [1.2.2 Training and testing log description](#122-training-and-testing-log-description)
  - [1.3 Raw data from case study](#13-raw-data-from-case-study)
- [2. Experiment Reproduction](#2-experiment-reproduction)
  - [2.1 Environment preparation](#21-environment-preparation)
    - [2.1.1 Setup environment via Docker](#211-setup-environment-via-docker)
  - [2.2 Training and testing reproduction](#22-training-and-testing-reproduction)
    - [2.2.1 Obtain statistics of the dataset](#211-setup-environment-via-docker)
    - [2.2.2 Reproduce testing scores of the model](#222-reproduce-testing-scores-of-the-model)
  - [2.3 Case study reproduction](#23-case-study-reproduction)
    - [2.3.0 Toy example on funarc](#230-toy-example-to-run-precimonious-on-funarc)
    - [2.3.1 Case study on CG](#231-case-study-on-cg)
    - [2.3.2 Case study on MG](#232-case-study-on-mg)
    - [2.3.3 Case study on Lulesh](#233-case-study-on-lulesh)
    - [2.3.4 Case study on LBM](#234-case-study-on-lbm)


<!-- > **Contents Overview**
> 
> **1. Raw Data Description**
>
> ====== 1.1 Dataset for model training
>
> ====== 1.2 Raw data from model training
>
> ============ 1.2.1 Pre-trained model description
>
> ============ 1.2.2 Training and testing log description
>
> ====== 1.3 Raw data from case study
>
> **2. Experiment Reproduction**
>
> ====== 2.1 Environment preparation
>
> ============ 2.1.1 Setup environment via Docker
>
> ====== 2.2 Training and testing reproduction
>
> ============ 2.2.1 Obtain statistics of the dataset
>
> ============ 2.2.2 Reproduce testing scores of the model
>
> ====== 2.3 Case study reproduction
>
> ============ 2.3.0 Toy example on funarc
> 
> ============ 2.3.1 Case study on CG
>
> ============ 2.3.2 Case study on MG
>
> ============ 2.3.3 Case study on Lulesh
> 
> ============ 2.3.4 Case study on LBM

<br/><br/> -->



## 1. Raw Data Description

### 1.1 Dataset for model training

Our MixBench datasets, one for calculation error prediction
([`raw/MixBench/error_root/processed`](https://github.com/ucd-plse/FPLearner/tree/main/raw/MixBench/error_root/processed)),
and the other for execution runtime prediction
on floating-point programs
([`raw/MixBench/time_root/processed`](https://github.com/ucd-plse/FPLearner/tree/main/raw/MixBench/time_root/processed)), are provided to reproduce
the experiment results reported in the paper.
Each data object ("data_idx.pt") is a graph representation
containing nodes, edges, and label information
for a corresponding mixed-precision floating-point program.

The MixBench dataset involves five benchmarks:
[BlackScholes](https://github.com/ucd-plse/FPLearner/blob/main/raw/MixBench/orig_files/blackscholes.cpp), 
[CFD](https://github.com/ucd-plse/FPLearner/blob/main/raw/MixBench/orig_files/euler3d.cpp), 
[Hotspot](https://github.com/ucd-plse/FPLearner/blob/main/raw/MixBench/orig_files/hotspot.cpp), 
[HPCCG](https://github.com/ucd-plse/FPLearner/blob/main/raw/MixBench/orig_files/hpccg.cpp), 
and [LavaMD](https://github.com/ucd-plse/FPLearner/blob/main/raw/MixBench/orig_files/lavaMD.cpp).
The source code of these five benchmarks in their original precision
can be found here ([`raw/MixBench/orig_files`](https://github.com/ucd-plse/FPLearner/tree/main/raw/MixBench/orig_files)).

| Task                       | Total | Label 0 | Label 1 | Avg. node count | Avg. edge count |
| -------------------------- | ----- | ------- | ------- | --------------- | --------------- |
| Error Prediction Dataset   | 600   | 300     | 300     | 3191            | 11597           |
| Runtime Prediction Dataset | 628   | 314     | 314     | 3195            | 11487           |

The table above shows the statistics of the two balanced datasets. Note that for error prediction, *label 0* refers to "program not within error threshold", and *label 1* refers to "program within error threshold"; while for runtime prediction, *label 0* refers to "program with speedup", and *label 1* refers to "program with no speedup".

### 1.2 Raw data from model training

#### 1.2.1 Pre-trained model description

<!-- (raw/model) -->

In [`raw/model`](https://github.com/ucd-plse/FPLearner/tree/main/raw/model), the artifact provides the two well-trained models in FPLearner, [`error_AST_CFG_PDG_CAST_DEP_checkpoint.pt`](https://github.com/ucd-plse/FPLearner/blob/main/raw/model/error_AST_CFG_PDG_CAST_DEP_checkpoint.pt) and [`time_AST_CFG_PDG_CAST_DEP_checkpoint.pt`](https://github.com/ucd-plse/FPLearner/blob/main/raw/model/time_AST_CFG_PDG_CAST_DEP_checkpoint.pt), which were trained and tested on the provided MixBench dataset. Both models were trained
on the composite graph representation (including AST, CFG, PDG, TypeCastingGraph, and VarDependenceGraph).

#### 1.2.2 Training and testing log description

<!-- (raw/log) -->

In [`raw/log`](https://github.com/ucd-plse/FPLearner/tree/main/raw/log), the artifact provides training and testing logs for
both error and runtime prediction models on different combinations
of edges. The logs provide the full set of results shown in the ablation study of edges in the paper.

For example, the testing results including accuracy, precision, recall, and f-1 score for the error prediction model learning on a composite graph can be found in [`raw/log/error_AST_CFG_PDG_CAST_DEP/test.log`](https://github.com/ucd-plse/FPLearner/blob/main/raw/log/error_AST_CFG_PDG_CAST_DEP/test.log) which has the following content:

```
2023-03-01 15:13:31,038 Savedir: MixBench/error_allpurpose_root_AST_CFG_PDG_CAST_DEP
2023-03-01 15:13:31,050 Test dataset size: 120
2023-03-01 15:13:31,050 Split: 
  test -> 120
2023-03-01 15:13:31,051 Edges: ['AST', 'CFG', 'PDG', 'CAST', 'DEP']
2023-03-01 15:13:32,777 Model (model/MixBench/error_allpurpose_root_AST_CFG_PDG_CAST_DEP/checkpoint.pt) is loaded.
2023-03-01 15:13:33,911 ============> Testing mode start...
2023-03-01 15:14:14,466 	 Test Loss: 1.850765 |  Test Acc: 96.875% |  Test Pre: 97.24% |  Test Rec: 96.82% |  Test Fsc: 97.03%
2023-03-01 15:14:14,466 	 Acc0: 92.98% |  Acc1: 100.00% |  Pre0: 100.00% |  Pre1: 94.49% |  Rec0: 93.63% | Rec1: 100.00% 
```

### 1.3 Raw data from case study

The running results from our case study on  the target benchmark CG are provided in [`raw/case-study/cg-results`](https://github.com/ucd-plse/FPLearner/tree/main/raw/case-study/cg-results).

This folder consists of the following files:
-  `*.json`: the precision configuraions for all mixed-precision floating-point programs in the search
-  `dd2-20230301-160059.log`: the log file which contains both the prediction results and the ground truth for each candidate program
-  `df-configs.csv`: the csv file which contains both the prediction results and the ground truth for each candidate program
-  `dd2_valid_cg_441.json`: the final precision configuration found in the search
-  `best_speedup_cg_441.txt`: the corresponding best speedup



## 2. Experiment Reproduction

> Please note that results of our model's training and case studies are non-deterministic, and thus are expected to vary from what is reported.

### 2.1 Environment preparation

The user has two options to run the tool: the **CPU Only** option, and the **GPU** option.

#### 2.1.1 Setup environment via Docker

 
##### 1. Required Prerequisites (for both options)

- Ubuntu 20.04 with kernel version 5.14.0 (The reproduction package has not been tested on other systems.)
- Docker 23.0.1 (The reproduction package has not been tested with other Docker versions.)
- 40 GiB free disk space recommended

- Clone this GitHub repository to your local directory 
```
git clone https://github.com/ucd-plse/FPLearner.git <YOUR LOCAL PATH TO THIS REPO>
```
#####  2. Optional Prerequisites (for the GPU option)

- An NVIDIA GPU with 48GiB memory reccomended (The reproduction package is tested on the Nvidia RTX A6000, with the driver version 525.)

- NVIDIA Container Toolkit installed with the instructions from [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit) in the *Setting up NVIDIA Container Toolkit* section (The toolkit allows you to use GPUs in the Docker container.)

- Make sure the Nvidia driver and library versions match in order to use GPU in the docker container. The tool is tested on the Nvidia Kernel Version and API version `525.89.02`.
  - Check the run-time driver information: 
  `cat /proc/driver/nvidia/version`.
  - Compare against the version for drivers installed: `dpkg -l | grep nvidia-driver`.




##### 3. Start the Docker Container
##### Step 1: Pull the Docker image 

```
docker pull jsid8qihgds3/artifact_cu11.7.1
```

Please note that the docker image size is 28.3GB.

##### Step 2: Run a Docker container (approx. a few secs)

```
docker run -v <YOUR LOCAL PATH TO THIS REPO>:/root/home -ti --gpus all  --name artifact jsid8qihgds3/artifact_cu11.7.1
```

Please replace `<YOUR LOCAL PATH TO THIS REPO>`
to your local path of this github repository.
If necessary, you can also change the container's name.
If the user prefers the CPU only option in the container, then remove the flag `--gpus all`.

##### Step 3: (For the GPU option) Check if CUDA is currently available in your container (approx. a few secs)

When you are inside the container,

```
python3 -c 'import torch; print(torch.cuda.is_available())'
```

The expected terminal output is "*True*" when CUDA is available in your container. If `torch.cuda.is_available()` returns "*False*",
you could first check the PyTorch version and the CUDA version
in the running environment, check [pytorch.org](https://pytorch.org)
to make sure the PyTorch version matches with the CUDA version.

In the subsequent process of experiments reproduction,
the first step is always to make sure you are inside the docker container. If you are not, please run the following command (approx. a few secs):

```
docker start -i artifact
```

### 2.2 Training and testing reproduction

#### 2.2.1 Obtain statistics of the dataset

The artifact provides the instructions to get statistics of our MixBench dataset.

##### ==> Run the following commands (approx. a few secs)

```
cd /root/home/scripts
python3 main.py -data
```

The expected terminal output is:

```
Finish dataset building.
Dataset size is:  628
100%|████████████████████| 628/628 [00:04<00:00, 153.55it/s]
in all:
# runtime == 0:  314  # runtime == 1:  314
# error == 0:  398  # error == 1:  230
Average edge number per graph = 11486.964968152866, average node number per graph = 3195.353503184713
```

The output messages describe the dataset information
for the program performance prediction task.
The information involves:
- Dataset size
- Label distribution
- The average number of edges and nodes per graph in the dataset

#### 2.2.2 Reproduce testing scores of the model

The artifact provides three options to reproduce the testing scores of our model.

##### Option 1: Testing on Pre-trained Model (Default)

By default, you're reccomended to reproduce testing scores reported in the paper useing the pre-trianed models. 

##### ==> Run the following commands (approx. 1min)

```
cd /root/home/scripts
python3 main.py -test -b 16
```

In this command, we set the batch size to be 16 by default.

The expected terminal output is:

```
Savedir:  runtime_AST_CFG_PDG_CAST_DEP
Test dataset size: 127
Split: 
  test -> 127
Edges: ['AST', 'CFG', 'PDG', 'CAST', 'DEP']
Model (time_AST_CFG_PDG_CAST_DEP_checkpoint.pt) is loaded.
100%|████████████████████| 8/8 [00:44<00:00,  5.56s/it]
Confusion_matrix: 
 [[60  3]
 [ 2 62]]
	 Test Loss: 1.957213 |  Test Acc: 96.094% |  Test Pre: 96.72% |  Test Rec: 95.96% |  Test Fsc: 96.34%
	 Acc0: 95.24% |  Acc1: 96.88% |  Pre0: 97.19% |  Pre1: 96.25% |  Rec0: 95.09% | Rec1: 96.83% 
```

The output message contains the following information:

- Testing dataset size
- Edges extracted in the Precision Interaction Graph (PIG)
- The confusion matrix on the testing dataset
- Metric scores to reflect the model's performance (accuracy, precision, recall and f1 score)



##### Option 2: Train the Model from Scratch with Required GPU (Optional)

We trained our models on the MixBench dataset from scratch using Nvidia RTX A6000 with a batch size of 16 data instances.

If you have the GPU with the required memory size 48GB, you could run this step to train from scratch.

##### ==> Run the following commands (approx. 15h)

```
cd /root/home/scripts
python3 main.py -train -b 16
```

The default total number of training epochs is set to be 500. The early-stopping approach is used with a patience of 30 epochs. The training process is tested on our GPU, Nvidia RTX A6000, where each epoch is expected to take around 2 minutes.
The training log and model checkpoints are automatically saved and updated under the `scripts/log` and `scripts/model` folders during the training process.


##### Option 3: Train the Model from Scratch with Smaller GPU (Optional)

If you have a GPU with a smaller memory size, you could train from scratch by decreasing the batch size, e.g. batch size = 1, but this will lead to a longer training time.

##### ==> Run the following commands (approx. more than 15h depending on machine)

```
cd /root/home/scripts
python3 main.py -train -b 1

```


### 2.3 Case study reproduction

In this section, our artifact presents instructions to reproduce case study on four target benchmarks: `CG`, `MG`, `LULESH`, and `LBM`. The user has the option to incorporate the fine-tuned model plugins into two different precision tuners: `Precimonious` and `HiFPTuner`.

#### 2.3.0 Toy example to run Precimonious on `funarc`
```
cd /root/home/case-study/Precimonious
python3 run.py funarc 10
```

Note that the second argument `10` indicates the timeout in seconds to run the benchmark `funarc` once.
To get an idea of how to compile and run `funarc`, check the file `/root/home/case-study/Precimonious/funarc/scripts/Makefile` 
and run `make` under the same directory.


#### 2.3.1 Case study on CG

##### Option 1: Precimonious
##### ==> Run the following commands (approx. 1h)
  
```
cd /root/home/case-study/Precimonious
python3 run.py cg 10
```

##### Option 2: Precimonious + Model Plugins
##### ==> Run the following commands (approx. 1h)
  
```
cd /root/home/case-study/Precimonious-plugin
python3 run.py cg 10
```

##### Option 3: HiFPTuner + Model Plugins
##### ==> Run the following commands (approx. 30min)
  
```
cd /root/home/case-study/HiFPTuner-plugin
python3 run.py cg 10
```

In the command `python3 run.py cg 10`, we execute the script called `run.py` to start dynamic precision tuning with plugin on the benchmark `CG`. The second argument `10` indicates the maximum time in seconds to run the benchmark `CG` once.

<details>
<summary>Click to check the beginning of the sample terminal output:</summary>
<pre><code>
include.json is generated.
rm -f *.out *config_*.json *.txt
Plugin arg size = 6
Output path = ./
Output file name = config.json
Input file name = cg.c
Output file created - ./config.json
Output file created - ./search_config.json
/usr/bin/ld: /usr/bin/../lib/gcc/x86_64-linux-gnu/9/../../../x86_64-linux-gnu/crt1.o: in function `_start':
(.text+0x24): undefined reference to `main'
clang-12: error: linker command failed with exit code 1 (use -v to see invocation)
Runtime time_predictor ../src/time_ggnn_5graphs_trl.pt is loaded.
Error error_predictor ../src/error_ggnn_5graphs_trl.pt is loaded.
Rootnode is 1534.
One time preloading...
** Searching for valid configuration using delta-debugging algorithm
cp config.json results-eps=4-A/VALID_config_cg_0.json
-------- running config 1 --------
mv config_temp.json results-eps=4-A/INVALID_config_cg_1.json
-------- running config 2 --------
mv config_temp.json results-eps=4-A/INVALID_config_cg_2.json
-------- running config 3 --------
mv config_temp.json results-eps=4-A/INVALID_config_cg_3.json
</code></pre>
</details>

> Please ignore the possible error messages during the search which do not have any affect to the run:
>   - "clang-12: error: linker command failed with exit code 1" 
>   - "fatal error: 'npbparams.h' file not found"


After the precision tuning is done, you can find a folder in `case-study/<TunerName>-plugin/cg/run/results-eps==4-A` which contains the following files (`<TunerName>` is either `Precimonious` or `HiFPTuner`):

- `*.json`: all precision configurations in the search
- `.log`: a log file containing model prediction results for each configuration and the corresponding verification results
- `.csv`: a csv file containing model prediction results for each configuration and the corresponding verification results
- `dd2_valid_{BENCH}_{IDX}.json`: the best precision configuration found by our tool
- `best_speedup_{BENCH}_{IDX}.txt`: the corresponding best speed up

#### 2.3.2 Case study on MG

##### Option 1: Precimonious + Model Plugins  
##### ==> Run the following commands (approx. 1h)

```
cd /root/home/case-study/Precimonious-plugin
python3 run.py mg 10
```

##### Option 2: HiFPTuner + Model Plugins  
##### ==> Run the following commands (approx. 40min)

```
cd /root/home/case-study/HiFPTuner-plugin
python3 run.py mg 10
```


For `MG`, the timeout is 10s. The expected terminal output and results are similar to `CG`.

#### 2.3.3 Case study on LULESH

##### Option 1: Precimonious + Model Plugins  
##### ==> Run the following commands (approx. 12h)

```
cd /root/home/case-study/Precimonious-plugin
python3 run.py lulesh 30
```
##### Option 2: HiFPTuner + Model Plugins  
##### ==> Run the following commands (approx. 3h)

```
cd /root/home/case-study/HiFPTuner-plugin
python3 run.py lulesh 30
```

For `LULESH`, the timeout is 30s. The expected terminal output and results are similar to `CG`.

#### 2.3.4 Case study on LBM

The `LBM` benchmark from SPEC CPU 2017 is proprietary and require a license from SPEC to use. We offer the instructions below and scripts for running our tool on LBM, but we don't provide the source code of LBM or any scripts to run SPEC benchmarks. If you have licence to SPEC CPU 2017 Benchmark Suites, please follow the instructions to run our tool:

Step 1: Downlowd and collect the source code of the LBM benchmark.

Step 2: Make sure you are able to compile and run the LBM with `specmake` provided by SPEC. For more information or instructions, please refer to the official website from [here](https://www.spec.org/cpu2017/Docs/).

Step 3: Install SPEC CPU 2017 in the docker container.

Step 4: Copy the source code of LBM to the path `/root/home/case-study/<TunerName>-plugin/scripts` and `/root/home/case-study/<TunerName>-plugin/tempscripts`. (Note that `<TunerName>` is either `Precimonious` or `HiFPTuner`. The following instructions will use `Precimonious` as an example.)

Step 5: Run the case study with the following commands

```
cd /root/home/case-study/Precimonious-plugin
python3 run.py lbm 300
```
For `LBM`, the timeout is 300s. The expected terminal output and results are similar to `CG`.
