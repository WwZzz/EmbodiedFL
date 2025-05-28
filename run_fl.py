from flgo.decorator import BasicDecorator
import flgo
import flgo.algorithm.fedavg as fedavg
import flgo.experiment.logger.full_logger as fel
import json
import os


# task = 'task/Panda_Lift_lowdim_bcrnn'
# task = 'task/Panda_TwoArmTransport_lowdim_bcrnn'
# task = 'task/Panda_PickPlaceCan_lowdim_bcrnn'
# task = 'task/Panda_NutAssemblySquare_lowdim_bcrnn'
task = 'task/Panda_ToolHang_lowdim_bcrnn'
# task = 'task/CE_SquareD0_lowdim_bcrnn'
# task = 'task/CE_ThreadingD0_lowdim_bcrnn'


runner = flgo.init(task, fedavg, option={'gpu':0, 'num_rounds':50, 'drop_last': True,'batch_size':100,'lr_scheduler':0,'learning_rate_decay':0.998 ,'proportion':1.0, 'clip_grad':10, 'learning_rate':0.0001,'optimizer':'Adam','weight_decay':0.0, 'save_checkpoint':f'fedavg'}, Logger=fel.FullLogger)
runner.run()