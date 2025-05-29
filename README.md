
<div align="center">
  <img src='assets/logo.png'  width="200"/>
<h1>Embodied Federated Learning</h1>

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
mv v1.5/* ./
```
#### (Optional) Generate Image Modality
Please complete this step before you want to run any tasks with image observations. If you only want to run tasks with low-dim features, just ingore this step. The original datasets only contains the low_dim data that can be used in an out-of-the-box way. You need to extract the image modality manually by the command below. This operation may take several hours.
```shell
# This is an example of converting lift-mg.
python scripts/dataset_states_to_obs.py --done_mode 0 --dataset data/robomimic/lift/mg/demo_v15.hdf5 --output_name image_sparse_v15.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84
```
To generate images for all robomimic datasets, please run the command below
```shell
cd scripts
./extract_obs_from_raw_datasets.sh
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
cd data/mimicgen
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

## TroubleShooting
- **Issues on mujoco_py**
If the code raises the error like `You appear to be missing MuJoCo. We expected to find the file here: path/to/mujoco210`, you can try to replace the line:35 `except ImportError:` with `except:` in the file `robomimic\robomimic\envs\env_robosuite.py` where the code tries to import `mujoco_py`. 
- **Cmake**
If the installation failed due to error like `Compatibility with CMake < 3.5 has been removed from CMake. Update the VERSION argument <min> value.`, please set the configurations as
```shell
export CMAKE_POLICY_VERSION_MINIMUM = X.X # X.X should be replaced by the version of cmake that can be found by `cmake --version`
```
- **Issues on EGL**
If the code raises the error `failed to open swrast: /usr/lib/dri/swrast_dri.so:`, please link the file into the target path
```shell
cd /usr/lib
ln -s /usr/lib/x86_64-linux-gnu/dri ./dri
```

If the code raises the error of `libstdc++.so.6`, please replace this file in the env's lib with the one from `/usr/lib/x86_64-linux-gnu/`
```shell
cd ENV_LIB_PATH # replaced by the lib path of your python interpreter environment, e.g., anaconda3/envs/ENV_NAME/lib
mv libstdc++.so.6 libstdc++.so.6.bak
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ./libstdc++.so.6
```

# Acknowledgement
This repo is based on the open-source repos below. We are grateful to the contributions of these authors
- [Robosuite](https://robosuite.ai/)
- [Robomimic](https://robomimic.github.io/)
- [MimicGen](https://github.com/NVlabs/mimicgen)
