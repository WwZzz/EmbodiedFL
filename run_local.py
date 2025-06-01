import argparse
import datetime
import numpy as np
from flgo.decorator import BasicDecorator
import flgo
import algo.local as local
import flgo.experiment.logger.full_logger as fel
import json
import os
import yaml
from torch.utils.tensorboard import SummaryWriter

class ClientFilter(BasicDecorator):
    """
    Preserving partial clients for the training process

    Args:
        preserved_idxs (list|int): the indices of clients to be preserved
    """
    def __init__(self, preserved_idxs):
        if isinstance(preserved_idxs, int): preserved_idxs = [preserved_idxs]
        self.preserved_idxs = sorted(preserved_idxs)
        self.removed_clients = []
        self.preserved_clients = []

    def __call__(self, runner, *args, **kwargs):
        clients = runner.clients
        num_clients = len(clients)
        all_clients = list(range(num_clients))
        removed_idx = [k for k in all_clients if k not in self.preserved_idxs]
        if len(removed_idx) > 0:
            self.removed_clients = [runner.clients[eid] for eid in removed_idx]
            self.preserved_clients = [runner.clients[cid] for cid in range(len(runner.clients)) if cid not in removed_idx]
            runner.reset_clients(self.preserved_clients)
        else:
            self.removed_clients = []
            self.preserved_clients = runner.clients
        self.register_runner(runner)
        return runner

    def __str__(self):
        return f"ClientFilter_{'_'.join([str(k) for k in self.preserved_idxs])}"

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
        local_data_vols = [c.datavol for c in self.clients]
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
parser.add_argument('--client_id', help='the id of client', type=int, default=-1)
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
    if args.client_id < 0: # all client
        for i in range(num_clients):
            config_i = config.copy()
            config_i['save_checkpoint'] = f'single_client_{i}'
            runner = flgo.init(task, local, option=config_i, Logger=MyLogger)
            ClientFilter(preserved_idxs=[i])(runner)
            runner.run()
    else:
        config_i = config.copy()
        assert args.client_id < num_clients
        config_i['save_checkpoint'] = f'single_client_{args.client_id}'
        runner = flgo.init(task, local, option=config_i, Logger=MyLogger)
        ClientFilter(preserved_idxs=[args.client_id])(runner)
        runner.run()
