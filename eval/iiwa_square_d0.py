import os
import robomimic.utils.env_utils as EnvUtils
import numpy as np
import sys
from tqdm import tqdm
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.utils import run_rollout

import flgo
import flgo.algorithm.fedavg as fedavg
import algo.fedavg_ph as ph
import robosuite as suite
import argparse
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

env_meta={'env_name': 'Square_D0', 'env_version': '1.4.1', 'type': 1, 'env_kwargs': {'has_renderer': False, 'has_offscreen_renderer': True, 'ignore_done': True, 'use_object_obs': True, 'use_camera_obs': True, 'control_freq': 20, 'controller_configs': {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], 'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'damping': 1, 'impedance_mode': 'fixed', 'kp_limits': [0, 300], 'damping_limits': [0, 10], 'position_limits': None, 'orientation_limits': None, 'uncouple_pos_ori': True, 'control_delta': True, 'interpolation': None, 'ramp_ratio': 0.2}, 'robots': ['IIWA'], 'camera_depths': False, 'camera_heights': 84, 'camera_widths': 84, 'gripper_types': ['Robotiq85Gripper'], 'render_gpu_device_id': 0, 'reward_shaping': False, 'camera_names': ['agentview', 'robot0_eye_in_hand']}}
env_name = 'Square_D0'



if __name__ == "__main__":
    render = True
    task = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'task', 'Panda_Lift_lowdim_bcrnn')
    ckpt = '/home/dataset/wz/MACHINE_LAB/EmbodiedFL/task/Panda_Lift_lowdim_bcrnn/ClientFilter_0/checkpoint/single_client_0/fedavg_Mdefault_model_R200_B100.0_E5_LR1.00e-04_P1.00e+00_S0_LD0_9.9800e-01_WD0.0000e+00_SIMSimulator_LGFullLogger.200'
    server =flgo.init(task, ph, option={'gpu':0, 'num_epochs':1, 'num_rounds':400, 'proportion':1.0, 'learning_rate':0.0001, 'optimizer':'Adam', 'load_checkpoint':ckpt, 'weight_decay':0.0})
    server._load_checkpoint()
    model = server.model
    model.eval()
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_name,
        render=True,
        render_offscreen=False,
        use_image_obs=False,
        use_depth_obs=False,
    )
    # create environment instance
    # env = EnvUtils.create_env_from_metadata(
    #     env_meta=env_meta,
    #     env_name=env_name,
    #     render=False,
    #     render_offscreen=True,
    #     use_image_obs=False,
    #     use_depth_obs=False,
    # )
    num_episodes = 50
    horizons = 400
    # reset the environment
    total_success = 0
    with torch.no_grad():
        for _ in range(num_episodes):
            result = run_rollout(model.model, env, horizon=horizons)
            total_success += result["Return"]
            print(result)
    print(f"Success Rate: {total_success['Return']/num_episodes}")

