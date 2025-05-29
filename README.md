
<div align="center">
  <img src='assets/logo.png'  width="200"/>
<h1>Federated Embodied Learning</h1>

</div>

This repo implement several robot arm manipulation tasks in simulation in a federated manner. It supports several common cases including homogeneous-embodiment collaboration, cross-embodiment collaboration across different tasks and data collectors.

## Installation
- pytorch
Please intall pytorch following the guidence in the pytorch's offsite [link](https://pytorch.org)

- Other
```shell
pip install flgo
pip install robosuite==1.5.1
# install robomimic
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
pip install -e .
cd ..
# install mimicgen
git clone https://github.com/NVlabs/mimicgen.git 
cd mimicgen
pip install -e .
cd ..
# install robosuite-task-zoo
git clone https://github.com/ARISE-Initiative/robosuite-task-zoo.git
cd robosuite-task-zoo
pip install -e .
cd ..
# downgrade mujoco to previous version
pip install mujoco==2.3.2 mujoco-python-viewer

```

## Characteristic
| Task Name | Cross-Collector                         | Cross-Embodiment          | Scale  | 
|-----------|-----------------------------------------|---------------------------|--------| 
| CE_SquareD0_lowdim_bcrnn       | ✔                                       | ✔                 | Small  |
| CE_SquareD1_lowdim_bcrnn    | ✔ | ✔              | Small  |
| CE_ThreadingD0_lowdim_bcrnn    | ✔ | ✔ | Small  |
| CE_ThreadingD1_lowdim_bcrnn  | ✔ | ✔           | Small  |
| Panda_Lift_lowdim_bcrnn  | ✔ |               | Small  |
|Panda_NutAssemblySquare_lowdim_bcrnn|      ✔                         |                           | Small  |
|Panda_PickPlaceCan_lowdim_bcrnn|      ✔                            |                           | Small  |
|Panda_ToolHang_lowdim_bcrnn|                                         |                           | Medium |
|Panda_TwoArmTransport_lowdim_bcrnn|      ✔                          |                           | Small  |

## Data Preparation
### Robomimic
Download the dataset from huggingface [link](https://huggingface.co/datasets/amandlek/robomimic) into `data`. The architecture of the data should be organized like
```
data
├─ robomimic
│  ├─ can                   
│  ├─ ...
│  └─ square   					 
│     ├─ mh                     
│     │  ├─ low_dim_v15.hdf5          
│     │  └─ demo_v15.hdf5  
│     └─ ph     
│        ├─ low_dim_v15.hdf5          
│        └─ demo_v15.hdf5   
...
```
One can use the following command to download the dataset
```shell
pip install huggingface
cd data/robomimic
huggingface-cli download --repo-type dataset amandlek/robomimic --local-dir .
```

### MimicGen
Download the dataset from the huggingface [link](https://huggingface.co/datasets/amandlek/mimicgen_datasets). The architecture should be organized as:
```
data
├─ mimicgen
│  ├─ core
│  ├─ ...
│  └─ robot   					 
│     ├─ square_d0_iiwa.hdf5                     
│     ...         
...
```
One can use the following command to download the dataset
```shell
pip install huggingface
cd data/robomimic
huggingface-cli download --repo-type dataset amandlek/mimicgen_datasets --local-dir .
```

### Preprocessing Data
After downloading the datasets and organizing them, you need to manually partition some hdf5 for further usages by flgo (i.e., specifying train\val split or client-level split). To do this, please run
```shell
python scripts/group_hf_dataset.py --file data/robomimic/tool_hang/ph/low_dim_v15.hdf5 --val 0.1 --num_groups 20
python scripts/group_hf_dataset.py --file data/mimicgen/robot/*.hdf5 --val 0.1 # replace * with file names 
```

## Run
### Local Only

```shell
python run_local.py --task TASK_NAME
```
### Federated Learning

```shell
python run_fl.py --task TASK_NAME --method ALGO_NAME
```
# Evaluate
To evaluate the task success rate for each trained model, please run the following script
```shell
python scripts/evaluate.py --task TASK_NAME --env_name ENV_NAME --robot ROBOT_NAME --ckpt CHECKPOINT_PATH
```

# Acknowledgement
This repo is based on the open-source repos below
- [Robosuite](https://robosuite.ai/)
- [Robomimic](https://robomimic.github.io/)
- [MimicGen](https://github.com/NVlabs/mimicgen)
