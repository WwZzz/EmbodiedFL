import argparse
from flgo.decorator import BasicDecorator
import flgo
import flgo.algorithm.fedavg as fedavg
import flgo.experiment.logger.full_logger as fel
import json
import os

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

# task = 'task/Panda_Lift_lowdim'
# task = 'task/Panda_TwoArmTransport_lowdim'
# task = 'task/Panda_PickPlaceCan_lowdim'
# task = 'task/Panda_NutAssemblySquare_lowdim'
# task = 'task/Panda_ToolHang_lowdim'
# task = 'task/CE_SquareD0_lowdim'
# task = 'task/CE_ThreadingD0_lowdim'

parser = argparse.ArgumentParser()
parser.add_argument('--task', help='the task name', type=str, default='tmp_task')
parser.add_argument('--method', help='the method name', type=str, default='')
parser.add_argument('--gpu', help='the id of gpu', type=int, default=0)
args = parser.parse_args()
config = {
    'gpu':0,
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
if __name__ == '__main__':
    task = os.path.join('task', args.task)
    with open(os.path.join(task, 'data.json'), 'r') as f:
        data = json.load(f)
        num_clients = len(data['client_names'])
    for i in range(num_clients):
        config_i = config.copy()
        config_i['save_checkpoint'] = f'single_client_{i}'
        runner = flgo.init(task, fedavg, option=config_i, Logger=fel.FullLogger)
        ClientFilter(preserved_idxs=[i])(runner)
        runner.run()