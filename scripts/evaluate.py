import collections
import os
import robomimic.utils.env_utils as EnvUtils
import sys
from tqdm import tqdm
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.utils import run_rollout
from eval.config import ALL_ENV_CONFIGS
import flgo
import flgo.algorithm.fedavg as fedavg
import argparse
import numpy as np
from collections import defaultdict
import json

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', help='the task config path', type=str)
parser.add_argument('--robot', help='the robot name', type=str)
parser.add_argument('--task', help='the task name', type=str, default='tmp_task')
parser.add_argument('--ckpt', help='the path of the checkpoint', type=str, default='')
parser.add_argument('--gpu', help='the id of gpu', type=int, default=0)
parser.add_argument('--num_episodes', help='the number of episodes', type=int, default=50)
parser.add_argument('--horizon', help='the horizon of each episode', type=int, default=400)
parser.add_argument('--render', help='if render', action='store_true', default=False)
parser.add_argument('--render_offscreen', help='if render offscreen', action='store_true', default=False)
parser.add_argument('--output_dir', help='the dir of the output path', type=str, default='')
parser.add_argument('--eval_interval', help='the interval of checkpoints to be evaluated', type=int, default=5)
parser.add_argument('--max_eval_times', help='the max times of evaluation', type=int, default=10)
parser.add_argument('--personalize', help='whether to use local models',action='store_true', default=False)
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


if __name__ == "__main__":
    task = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'task', args.task)
    env_name = args.env_name
    env_meta = load_env_meta(env_name, args.robot)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_name,
        render=args.render,
        render_offscreen=args.render_offscreen,
        use_image_obs=False,
        use_depth_obs=False,
    )
    if not os.path.isdir(args.ckpt):
        server = flgo.init(task, fedavg, option={'gpu': args.gpu, 'load_checkpoint': args.ckpt, })
        server._load_checkpoint()
        model = server.model
        model.eval()
        # reset the environment
        all_results = []
        with torch.no_grad():
            for _ in tqdm(range(args.num_episodes)):
                result = run_rollout(model.model, env, render=args.render, horizon=args.horizon)
                all_results.append(result)
        final_results = process_results(all_results)
        final_results['task'] = args.task
        final_results['env_name'] = env_name
        final_results['num_episodes'] = args.num_episodes
        final_results['horizon'] = args.horizon
        final_results['ckpt'] = args.ckpt
        final_results['round'] = os.path.split(args.ckpt)[-1].split('.')[-1]
        print(final_results)
        if args.output_dir!='':
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
        for ckpt in tqdm(all_ckpts):
            server.option['load_checkpoint'] = ckpt
            server._load_checkpoint()
            model = server.model
            model.eval()
            ckpt_results = []
            with torch.no_grad():
                for _ in tqdm(range(args.num_episodes)):
                    result = run_rollout(model.model, env, render=args.render, horizon=args.horizon)
                    ckpt_results.append(result)
            crt_results = process_results(ckpt_results)
            crt_results['task'] = args.task
            crt_results['env_name'] = env_name
            crt_results['num_episodes'] = args.num_episodes
            crt_results['horizon'] = args.horizon
            crt_results['ckpt'] = ckpt
            crt_results['round'] = os.path.split(ckpt)[-1].split('.')[-1]
            all_results.append(crt_results)
            eval_times += 1
            if eval_times >= args.max_eval_times:
                break
        all_success_rate = [cres['success_rate'] for cres in all_results]
        max_success_rate = max(all_success_rate)
        optimal_round = all_results[np.argmax(all_success_rate)]['round']
        print(f"Max Success Rate:{max_success_rate} at Round {optimal_round}")
        if args.output_dir!='':
            res_name = os.path.join(args.output_dir, f"{args.task}_{args.num_episodes}_{args.horizon}_{args.ckpt.split(os.sep)[-2]}_all_{args.eval_interval}_{args.max_eval_times}.json")
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




