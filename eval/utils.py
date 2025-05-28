import os
import robomimic.utils.env_utils as EnvUtils
import numpy as np
import sys
from tqdm import tqdm
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import robosuite as suite
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils


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
            # ac = policy.get_action(ob_dict).squeeze().detach().cpu().numpy()
            ac = np.random.randn(env.action_dimension)
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