import importlib

import flgo
import algo.fedavg as fedavg
import flgo.experiment.logger.full_logger as fel
import os
import argparse
import datetime
import numpy as np
import json
import os
import yaml
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--task', help='the task name', type=str, default='CE_SquareD0_lowdim')
parser.add_argument('--method', help='the method name', type=str, default='fedavg')
parser.add_argument('--gpu', help='the id of gpu', type=int, default=0)
parser.add_argument('--config', help='the config path', type=str, default='')
parser.add_argument('--ckpt_prefix', help='the checkpoint name', type=str, default='')
args = parser.parse_args()

class MyLogger(fel.FullLogger):
    def initialize(self, *args, **kwargs):
        self.writter = SummaryWriter(os.path.join(self.server.option['task'], "runs/" + self.server.option['save_checkpoint']+'_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    def log_once(self, *args, **kwargs):
        # calculate weighted averaging of metrics on training datasets across participants
        local_data_vols = [c.datavol for c in self.server.clients]
        total_data_vol = sum(local_data_vols)
        # calculate weighted averaging and other statistics of metrics on validation datasets across clients
        local_val_metrics = self.server.global_test(flag='val')
        for met_name, met_val in local_val_metrics.items():
            self.output['local_val_'+met_name+'_dist'].append(met_val)
            self.output['local_val_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
            self.writter.add_scalar('val_' + met_name, 1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol, self.server.current_round)
            self.output['mean_local_val_' + met_name].append(np.mean(met_val))
            self.output['std_local_val_' + met_name].append(np.std(met_val))
        # output to stdout
        self.show_current_output()

default_config = {
    'num_rounds':50,
    'num_epochs': 1,
    'drop_last': True,
    'batch_size':100,
    'lr_scheduler':0,
    'learning_rate_decay':0.998 ,
    'proportion':1.0,
    'clip_grad':10,
    'learning_rate':0.001,
    'optimizer':'Adam',
    'weight_decay':0.0,
}

if __name__=='__main__':
    # Initialize Task
    task = os.path.join('task', args.task)
    # Initialize Configuration
    if args.config=='' or not os.path.exists(args.config):
        config =  default_config
    else:
        with open(args.config, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    config['gpu'] = args.gpu
    config['save_checkpoint'] = args.ckpt_prefix + f'{args.method}'
    config['load_checkpoint'] = args.ckpt_prefix + f'{args.method}'
    # Initialize Algorithm
    algo_modules = [".".join(["algo", args.method]), ".".join(["flgo", "algorithm", args.method])]
    algo = None
    for m in algo_modules:
        try:
            algo = importlib.import_module(m)
            break
        except ModuleNotFoundError:
            continue
    if algo is None: raise ModuleNotFoundError("{} was not found".format(algo))
    # Run
    config['load_mode'] = "mmap"
    runner = flgo.init(task, algo, option=config, Logger=MyLogger)
    runner.run()