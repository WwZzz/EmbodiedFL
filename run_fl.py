from flgo.decorator import BasicDecorator
import flgo
import flgo.algorithm.fedavg as fedavg
import flgo.experiment.logger.full_logger as fel
import json
import os
import argparse

# task = 'task/Panda_Lift_lowdim'
# task = 'task/Panda_TwoArmTransport_lowdim'
# task = 'task/Panda_PickPlaceCan_lowdim'
# task = 'task/Panda_NutAssemblySquare_lowdim'
# task = 'task/Panda_ToolHang_lowdim'
# task = 'task/CE_SquareD0_lowdim'
# task = 'task/CE_ThreadingD0_lowdim'

parser = argparse.ArgumentParser()
parser.add_argument('--task', help='the task name', type=str, default='CE_SquareD0_lowdim')
parser.add_argument('--method', help='the method name', type=str, default='')
parser.add_argument('--gpu', help='the id of gpu', type=int, default=0)
args = parser.parse_args()

if __name__=='__main__':
    task = os.path.join('task', args.task)
    config = {
        'gpu':args.gpu,
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
        'save_checkpoint': 'fedavg'
    }
    runner = flgo.init(task, fedavg, option=config, Logger=fel.FullLogger)
    runner.run()