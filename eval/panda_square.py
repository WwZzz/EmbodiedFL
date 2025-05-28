import os
import robomimic.utils.env_utils as EnvUtils
import numpy as np
import sys
from tqdm import tqdm
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flgo
import flgo.algorithm.fedavg as fedavg
import algo.fedavg_ph as ph
import robosuite as suite
import argparse
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

env_meta = {'env_name': 'NutAssemblySquare', 'env_version': '1.5.1', 'type': 1, 'env_kwargs': {'has_renderer': False, 'has_offscreen_renderer': False, 'ignore_done': True, 'use_object_obs': True, 'use_camera_obs': False, 'control_freq': 20, 'controller_configs': {'type': 'BASIC', 'body_parts': {'right': {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], 'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'damping': 1, 'impedance_mode': 'fixed', 'kp_limits': [0, 300], 'damping_limits': [0, 10], 'position_limits': None, 'orientation_limits': None, 'uncouple_pos_ori': True, 'control_delta': True, 'interpolation': None, 'ramp_ratio': 0.2, 'input_ref_frame': 'world', 'gripper': {'type': 'GRIP'}}}}, 'robots': ['Panda'], 'camera_depths': False, 'camera_heights': 84, 'camera_widths': 84, 'lite_physics': False, 'reward_shaping': False}}
env_name = 'NutAssemblySquare'


def run_rollout(
        policy,
        env,
        horizon=400,
        use_goals=False,
        render=True,
        terminate_on_success=True,
    ):
    """
    Runs a rollout in an environment with the current network parameters.

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        env (EnvBase instance): environment to use for rollouts.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        render (bool): if True, render the rollout to the screen

        video_writer (imageio Writer instance): if not None, use video writer object to append frames at
            rate given by @video_skip

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

    Returns:
        results (dict): dictionary containing return, success rate, etc.
    """
    policy.reset()
    ob_dict = env.reset()
    goal_dict = None
    if use_goals: goal_dict = env.get_goal()
    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    success = { k: False for k in env.is_success() } # success metrics
    try:
        for step_i in range(horizon):
            # get action from policy
            ac = policy.get_action(ob_dict).squeeze().detach().cpu().numpy()
            # play action
            ob_dict, r, done, _ = env.step(ac)
            # render to screen
            if render:
                env.render(mode="human")
            # compute reward
            total_reward += r
            cur_success_metrics = env.is_success()
            for k in success:
                success[k] = success[k] or cur_success_metrics[k]
            # break if done
            if done or (terminate_on_success and success["task"]):
                break
    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    results["Return"] = total_reward
    results["Horizon"] = step_i + 1
    results["Success_Rate"] = float(success["task"])
    # log additional success metrics
    for k in success:
        if k != "task":
            results["{}_Success_Rate".format(k)] = float(success[k])
    return results

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

