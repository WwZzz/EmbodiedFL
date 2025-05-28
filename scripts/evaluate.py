import os
import robomimic.utils.env_utils as EnvUtils
import sys
from tqdm import tqdm
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.utils import run_rollout
import flgo
import flgo.algorithm.fedavg as fedavg
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', help='the task config path', type=str)
parser.add_argument('--robot', help='the robot name', type=str)
parser.add_argument('--task', help='the task name', type=str, default='tmp_task')
parser.add_argument('--ckpt', help='the path of the checkpoint', type=str, default='')
parser.add_argument('--gpu', help='the id of gpu', type=int, default=0)
parser.add_argument('--num_episodes', help='the number of episodes', type=int, default=50)
parser.add_argument('--horizon', help='the horizon of each episode', type=int, default=400)
args = parser.parse_args()


def load_env_meta(env_name, robot_name):
    return

if __name__ == "__main__":
    render = True
    task = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'task', args.task)
    server =flgo.init(task, fedavg, option={'gpu':args.gpu, 'load_checkpoint':args.ckpt,})
    server._load_checkpoint()
    model = server.model
    model.eval()
    env_name = args.env_name
    env_meta = load_env_meta(env_name, args.robot)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_name,
        render=True,
        render_offscreen=False,
        use_image_obs=False,
        use_depth_obs=False,
    )
    num_episodes = args.num_episodes
    horizons = args.horizon
    # reset the environment
    total_success = 0
    with torch.no_grad():
        for _ in tqdm(range(num_episodes)):
            result = run_rollout(model.model, env, horizon=horizons)
            total_success += result["Return"]
            print(result)
    print(f"Success Rate: {total_success['Return']/num_episodes}")

