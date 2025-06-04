import collections
import os

import imageio
import robomimic.utils.env_utils as EnvUtils
import sys
from tqdm import tqdm
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.utils import run_rollout, setup_seed
from eval.config import ALL_ENV_CONFIGS
import flgo
import flgo.algorithm.fedavg as fedavg
import argparse
import numpy as np
from collections import defaultdict
import json
from tianshou.env import SubprocVectorEnv

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', help='the task config path', type=str)
parser.add_argument('--robot', help='the robot name', type=str)
parser.add_argument('--task', help='the task name', type=str, default='tmp_task')
parser.add_argument('--ckpt', help='the path of the checkpoint', type=str, default='')
parser.add_argument('--gpu', help='the id of gpu', type=int, default=0)
parser.add_argument('--num_episodes', help='the number of episodes', type=int, default=50)
parser.add_argument('--num_envs', help='the number of episodes', type=int, default=-1)
parser.add_argument('--horizon', help='the horizon of each episode', type=int, default=400)
parser.add_argument('--render', help='if render', action='store_true', default=False)
parser.add_argument('--render_offscreen', help='if render offscreen', action='store_true', default=False)
parser.add_argument('--output_dir', help='the dir of the output path', type=str, default='')
parser.add_argument('--eval_interval', help='the interval of checkpoints to be evaluated', type=int, default=2)
parser.add_argument('--max_eval_times', help='the max times of evaluation', type=int, default=-1)
parser.add_argument('--personalize', help='whether to use local models',action='store_true', default=False)
parser.add_argument('--seed', help='the random seed', type=int, default=0)
args = parser.parse_args()

def load_env_meta(env_name, robot_name):
    return ALL_ENV_CONFIGS[env_name][robot_name]

def process_results(results):
    l = defaultdict(list)
    for ri in results:
        for k in ri.keys():
            l[k].append(ri[k])
    mean_reward = np.array(l['reward']).mean().astype(np.float32)
    success_rate = np.array(l['success_rate']).astype(np.float32).sum()/len(results) * 100
    mean_horizon = np.mean(l['horizon']).astype(np.float32)
    return {'reward': mean_reward.item(), 'success_rate': success_rate.item(), 'horizon': mean_horizon.item()}

def create_env(env_meta, env_name, render=False, render_offscreen=False, use_image_obs=True):
    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_name,
        render=render,
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
        use_depth_obs=False,
    )

if __name__ == "__main__":
    setup_seed(args.seed)
    task = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'task', args.task)
    env_name = args.env_name
    env_meta = load_env_meta(env_name, args.robot)
    if args.num_envs > 0:
        env_list = [create_env(env_meta, env_name, False, args.render_offscreen) for _ in range(args.num_envs)]
        env = SubprocVectorEnv(env_list)
    else:
        env = create_env(env_meta, env_name, args.render, args.render_offscreen)
    if not os.path.isdir(args.ckpt):
        server = flgo.init(task, fedavg, option={'gpu': args.gpu, 'load_checkpoint': args.ckpt, })
        server._load_checkpoint()
        model = server.model
        model.eval()
        # reset the environment
        all_results = []
        if args.render_offscreen:
            video_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(args.ckpt))), 'video', os.path.split(os.path.dirname(args.ckpt))[-1])
            if not os.path.exists(video_dir): os.makedirs(video_dir)
            video_path = os.path.join(video_dir, os.path.split(args.ckpt)[-1]) + '.mp4'
            video_writer = imageio.get_writer(video_path, fps=20)
        else:
            video_writer = None
        for _ in tqdm(range(args.num_episodes)):
            result = run_rollout(model.model, env, render=args.render, horizon=args.horizon, video_writer=video_writer)
            all_results.append(result)
        if video_writer is not None: video_writer.close()
        final_results = process_results(all_results)
        final_results['task'] = args.task
        final_results['env_name'] = env_name
        final_results['num_episodes'] = args.num_episodes
        final_results['horizon'] = args.horizon
        final_results['ckpt'] = args.ckpt
        final_results['round'] = os.path.split(args.ckpt)[-1].split('.')[-1]
        print(final_results)
        if args.output_dir!='':
            if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
            res_name = os.path.join(args.output_dir, f"{args.task}_{args.num_episodes}_{args.horizon}_{args.ckpt.split(os.sep)[-2]}_{os.path.split(args.ckpt)[-1].split('.')[-1]}.json")
            with open(res_name, 'w') as f:
                json.dump(final_results, f)
    else:
        all_ckpts = sorted(os.listdir(args.ckpt), key=lambda x: int(x.split('.')[-1]))
        all_ckpts = [os.path.join(args.ckpt, f) for f in all_ckpts]
        if args.eval_interval>1:
            idxs = np.arange(0, len(all_ckpts), args.eval_interval)
            all_ckpts = [all_ckpts[idx] for idx in idxs]
        server =flgo.init(task, fedavg, option={'gpu':args.gpu, 'load_checkpoint':args.ckpt,})
        all_results = []
        eval_times = 0
        with tqdm(all_ckpts, desc="Processing Checkpoints") as pbar:
            for ckpt in pbar:
                server.option['load_checkpoint'] = ckpt
                server._load_checkpoint()
                model = server.model
                model.eval()
                ckpt_results = []
                if args.render_offscreen:
                    video_dir = os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), 'video', os.path.split(args.ckpt)[-1])
                    if not os.path.exists(video_dir): os.makedirs(video_dir)
                    video_path = os.path.join(video_dir, os.path.split(ckpt)[-1]) + '.mp4'
                    video_writer = imageio.get_writer(video_path, fps=20)
                else:
                    video_writer = None
                for _ in tqdm(range(args.num_episodes), leave=False):
                    result = run_rollout(model.model, env, render=args.render, horizon=args.horizon, video_writer=video_writer)
                    ckpt_results.append(result)
                if video_writer is not None: video_writer.close()
                crt_results = process_results(ckpt_results)
                crt_results['task'] = args.task
                crt_results['env_name'] = env_name
                crt_results['num_episodes'] = args.num_episodes
                crt_results['horizon'] = args.horizon
                crt_results['ckpt'] = ckpt
                crt_results['round'] = os.path.split(ckpt)[-1].split('.')[-1]
                all_results.append(crt_results)
                eval_times += 1
                pbar.set_description(f"Processing Checkpoints | Success Rate: {crt_results['success_rate']:.4f}")
                if args.max_eval_times>0 and eval_times >= args.max_eval_times:
                    break
        all_success_rate = [cres['success_rate'] for cres in all_results]
        max_success_rate = max(all_success_rate)
        optimal_round = all_results[np.argmax(all_success_rate)]['round']
        print(f"Max Success Rate:{max_success_rate} at Round {optimal_round}")
        if args.output_dir!='':
            if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
            res_name = os.path.join(args.output_dir, f"{args.task}_{args.num_episodes}_{args.horizon}_{args.ckpt.split(os.sep)[-1]}_all_{args.eval_interval}_{args.max_eval_times}.json")
            if os.path.exists(res_name):
                s = input("Record already exists. Enter 'y' to overwrite, and any else to break down")
                if s.strip().lower()=='y':
                    with open(res_name, 'w') as f:
                        json.dump(all_results, f)
                else:
                    print(all_results)
            else:
                with open(res_name, 'w') as f:
                    json.dump(all_results, f)




