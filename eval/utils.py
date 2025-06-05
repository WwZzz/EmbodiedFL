import os
from copy import deepcopy

import robomimic.utils.env_utils as EnvUtils
import numpy as np
import random
import sys
from tqdm import tqdm
import torch
from tianshou.env import SubprocVectorEnv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import robosuite as suite
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils


def batchify_obs(obs_list):
    keys = list(obs_list[0].keys())
    obs = {
        k: np.stack([obs_list[i][k] for i in range(len(obs_list))]) for k in keys
    }
    return obs

@torch.no_grad()
def run_rollout(
        policy,
        env,
        horizon=400,
        use_goals=False,
        render=True,
        terminate_on_success=True,
        video_writer=None,
        video_skip = 5,
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
    batched = isinstance(env, SubprocVectorEnv)
    policy.reset()
    ob_dict = env.reset()
    goal_dict = None
    if use_goals: goal_dict = env.get_goal()
    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    rews = []
    # success = { k: False for k in env.is_success() } # success metrics
    success = None
    if batched:
        end_step = [None for _ in range(len(env))]
    else:
        end_step = None

    if batched:
        video_frames = [[] for _ in range(len(env))]
    else:
        video_frames = []
    try:
        video_count = 0
        for step_i in range(horizon):
            # get action from policy
            if batched:
                policy_ob = batchify_obs(ob_dict)
                ac = policy.get_action(obs_dict=policy_ob, batched=True).squeeze().detach().cpu().numpy()  # , return_ob=True)
            else:
                policy_ob = ob_dict
                ac = policy.get_action(ob_dict).squeeze().detach().cpu().numpy()
            # ac = np.random.randn(env.action_dimension)
            # play action
            ob_dict, r, done, info = env.step(ac)
            # render to screen
            if render: env.render(mode="human")
            rews.append(r)
            # compute reward
            if batched:
                cur_success_metrics = TensorUtils.list_of_flat_dict_to_dict_of_list([info[i]["is_success"] for i in range(len(info))])
                cur_success_metrics = {k: np.array(v) for (k, v) in cur_success_metrics.items()}
            else:
                cur_success_metrics = env.is_success()
            if success is None:
                success = deepcopy(cur_success_metrics)
            else:
                for k in success:
                    success[k] = success[k] | cur_success_metrics[k]
            if video_writer is not None:
                if video_count % video_skip == 0:
                    if batched:
                        # frames = env.render(mode="rgb_array", height=video_height, width=video_width)
                        frames = []
                        policy_ob = deepcopy(policy_ob)
                        for env_i in range(len(env)):
                            cam_imgs = []
                            for im_name in ["agentview_image", "robot0_eye_in_hand_image"]:
                                im = TensorUtils.to_numpy(
                                    policy_ob[im_name][env_i, -1]
                                )
                                im = np.transpose(im, (1, 2, 0))
                                if policy_ob.get("ret", None) is not None:
                                    im_ret = TensorUtils.to_numpy(
                                        policy_ob["ret"]["obs"][im_name][env_i, :, -1]
                                    )
                                    im_ret = np.transpose(im_ret, (0, 2, 3, 1))
                                    im = np.concatenate((im, *im_ret), axis=0)
                                cam_imgs.append(im)
                            frame = np.concatenate(cam_imgs, axis=1)
                            frame = (frame * 255.0).astype(np.uint8)
                            frames.append(frame)
                        for env_i in range(len(env)):
                            frame = frames[env_i]
                            video_frames[env_i].append(frame)
                    else:
                        frame = env.render(mode="rgb_array", height=512, width=512)
                        video_frames.append(frame)
                video_count += 1
            if batched:
                for env_i in range(len(env)):
                    if end_step[env_i] is not None:
                        continue
                    if done[env_i] or (terminate_on_success and success["task"][env_i]):
                        end_step[env_i] = step_i
            else:
                if done or (terminate_on_success and success["task"]):
                    end_step = step_i
                    break
            # total_reward += r
            # cur_success_metrics = env.is_success()
            # for k in success:
            #     success[k] = success[k] or cur_success_metrics[k]
            # if video_writer is not None:
            #     if video_count % video_skip == 0:
            #         video_img = env.render(mode="rgb_array", height=512, width=512)
            #         video_writer.append_data(video_img)
            # # break if done
            # if done or (terminate_on_success and success["task"]):
            #     break
    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))
    if video_writer is not None:
        if batched:
            for env_i in range(len(video_frames)):
                for frame in video_frames[env_i]:
                    video_writer.append_data(frame)
        else:
            for frame in video_frames:
                video_writer.append_data(frame)
    # results["reward"] = total_reward
    # results["horizon"] = step_i + 1
    # results["success_rate"] = float(success["task"])
    # # log additional success metrics
    # for k in success:
    #     if k != "task":
    #         results["{}_Success_Rate".format(k)] = float(success[k])
    if batched:
        total_reward = np.zeros(len(env))
        rews = np.array(rews)
        for env_i in range(len(env)):
            end_step_env_i = end_step[env_i] or step_i
            total_reward[env_i] = np.sum(rews[:end_step_env_i + 1, env_i])
            end_step[env_i] = end_step_env_i

        results["reward"] = total_reward
        results["horizon"] = np.array(end_step) + 1
        results["success_rate"] = success["task"].astype(float)
    else:
        end_step = end_step or step_i
        total_reward = np.sum(rews[:end_step + 1])
        results["reward"] = total_reward
        results["horizon"] = end_step + 1
        results["success_sate"] = float(success["task"])
    # log additional success metrics
    for k in success:
        if k != "task":
            if batched:
                results["{}_Success_Rate".format(k)] = success[k].astype(float)
            else:
                results["{}_Success_Rate".format(k)] = float(success[k])
    return results

def setup_seed(seed):
    r"""
    Fix all the random seed used in numpy, torch and random module

    Args:
        seed (int): the random seed
    """
    random.seed(1+seed)
    np.random.seed(21+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(12+seed)
    torch.cuda.manual_seed_all(123+seed)