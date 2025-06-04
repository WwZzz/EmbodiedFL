import argparse
import datetime
import numpy as np
from flgo.decorator import BasicDecorator
import flgo
import algo.centralized
import flgo.experiment.logger.full_logger as fel
import json
import os
import yaml
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as tud

class Centralized(BasicDecorator):
    def __call__(self, runner, *args, **kwargs):
        all_train_data = tud.ConcatDataset([c.train_data for c in runner.clients])
        all_val_data = tud.ConcatDataset([c.val_data for c in runner.clients]) if runner.clients[0].val_data is not None and len(runner.clients[0].val_data) > 0 else None
        runner.clients.set_data(all_train_data, 'train')
        runner.clients.set_data(all_val_data, 'val')
        runner.reset_clients([runner.clients[0]])
        runner.gv.logger.clients = [runner.clients[0]]
        self.register_runner(runner)
        return runner

    def __str__(self):
        return f"Centralized"

default_config = {
    'num_rounds':50,
    'num_epochs': 1,
    'drop_last': True,
    'batch_size':100,
    'lr_scheduler':0,
    'learning_rate_decay':0.998 ,
    'proportion':1.0,
    'clip_grad':10,
    'learning_rate':0.0001,
    'optimizer':'Adam',
    'weight_decay':0.0,
}

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

parser = argparse.ArgumentParser()
parser.add_argument('--task', help='the task name', type=str, default='tmp_task')
parser.add_argument('--gpu', help='the id of gpu', type=int, default=0)
parser.add_argument('--config', help='the config path', type=str, default='')
parser.add_argument('--ckpt_prefix', help='the checkpoint name', type=str, default='')
args = parser.parse_args()

if __name__ == '__main__':
    # Init Task Path
    task = os.path.join('task', args.task)
    # Load Configuration
    if args.config=='' or not os.path.exists(args.config):
        config =  default_config
    else:
        with open(args.config, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    config['gpu'] = args.gpu
    # Get Total Number of Clients
    with open(os.path.join(task, 'data.json'), 'r') as f:
        data = json.load(f)
        num_clients = len(data['client_names'])
    # Train a Local Model for Each Client and Save Them
    config_i = config.copy()
    config_i['save_checkpoint'] = args.ckpt_prefix + f'all_centralized'
    config_i['load_checkpoint'] = args.ckpt_prefix + f'all_centralized'
    runner = flgo.init(task, algo.centralized, option=config_i, Logger=MyLogger)
    Centralized()(runner)
    runner.run()
