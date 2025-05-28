import flgo
import flgo.algorithm.fedavg as fedavg
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='the task config path', type=str)
parser.add_argument('--task', help='task name', type=str, default='tmp_task')
args = parser.parse_args()

task_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'task')
flgo.gen_direct_task_from_file(args.task, args.config, target_path=task_dir)
task = os.path.join(task_dir, args.task)
flgo.init(task, fedavg, option={'gpu':0, 'num_epochs':1, 'num_rounds':400, 'proportion':1.0, 'learning_rate':0.0001, 'optimizer':'Adam', 'save_checkpoint':'test_ckpt', 'load_checkpoint':'ph3', 'weight_decay':0.0}).run()