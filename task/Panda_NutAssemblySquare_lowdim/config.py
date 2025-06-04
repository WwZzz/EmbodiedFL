from robomimic.utils.dataset import SequenceDataset
from robomimic.config.config import Config
import torch.nn as nn
import torch.utils.data as tud
import robomimic.utils.obs_utils as ObsUtils
from  collections import OrderedDict, defaultdict
from utils.data_utils import Float32Converter
from utils.model_utils import BCRNN, BCRNN_ENCODER_CONFIG
from torch.optim import Adam
import torch
import os

ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'robomimic')
MODALITIES = {
        "obs": {
            "low_dim": [
                "object",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            ],
            "rgb": [],
            "depth": [],
            "scan": []
        },
        "goal": {
            "low_dim": [],
            "rgb": [],
            "depth": [],
            "scan": []
        }
    }
obs_key_shapes = OrderedDict([('object', [14]), ('robot0_eef_pos', [3]), ('robot0_eef_quat', [4]), ('robot0_gripper_qpos', [2])])
ObsUtils.initialize_obs_utils_with_obs_specs(MODALITIES)
ac_dim = 7
RNN_CONFIG = {
    "enabled": True,
    "horizon": 10,
    "hidden_dim": 400,
    "rnn_type": "LSTM",
    "num_layers": 2,
    "open_loop": False,
    "kwargs": { "bidirectional": False }
}
train_paras = [
    (ROOT_DIR + "/square/ph/low_dim_v15.hdf5", 'train'),
    (ROOT_DIR + "/square/mh/low_dim_v15.hdf5", 'better_operator_1_train'),
    (ROOT_DIR + "/square/mh/low_dim_v15.hdf5", 'better_operator_2_train'),
    (ROOT_DIR + "/square/mh/low_dim_v15.hdf5", 'okay_operator_1_train'),
    (ROOT_DIR + "/square/mh/low_dim_v15.hdf5", 'okay_operator_2_train'),
    (ROOT_DIR + "/square/mh/low_dim_v15.hdf5", 'worse_operator_1_train'),
    (ROOT_DIR + "/square/mh/low_dim_v15.hdf5", 'worse_operator_2_train'),
]
val_paras = [
    (ROOT_DIR + "/square/ph/low_dim_v15.hdf5", 'valid'),
    (ROOT_DIR + "/square/mh/low_dim_v15.hdf5", 'better_operator_1_valid'),
    (ROOT_DIR + "/square/mh/low_dim_v15.hdf5", 'better_operator_2_valid'),
    (ROOT_DIR + "/square/mh/low_dim_v15.hdf5", 'okay_operator_1_valid'),
    (ROOT_DIR + "/square/mh/low_dim_v15.hdf5", 'okay_operator_2_valid'),
    (ROOT_DIR + "/square/mh/low_dim_v15.hdf5", 'worse_operator_1_valid'),
    (ROOT_DIR + "/square/mh/low_dim_v15.hdf5", 'worse_operator_2_valid'),
]

#------------------------------------------------------------------------------------------------------------------------------------------------------

def create_config(data_path, filter_by_attribute, seq_length=10):
    return {
        'hdf5_path': data_path,
        'obs_keys': list(obs_key_shapes.keys()),
        'dataset_keys': ('actions', 'rewards', 'dones'),
        'load_next_obs': False,
        'frame_stack': 1,
        'seq_length': seq_length,
        'pad_frame_stack': True,
        'pad_seq_length': True,
        'get_pad_mask': False,
        'goal_mode': None,
        'hdf5_cache_mode': None,
        'hdf5_use_swmr': True,
        'hdf5_normalize_obs': False,
        'filter_by_attribute': filter_by_attribute,
    }

trains = [Float32Converter(SequenceDataset(**create_config(*pi))) for pi in train_paras]
vals = [Float32Converter(SequenceDataset(**create_config(*pi))) for pi in val_paras]
train_data = [tud.ConcatDataset([ti,vi]) for ti,vi in zip(trains, vals)]

def get_model():
    algo_config_rnn = Config(**RNN_CONFIG)
    obs_config = Config(**{
        "modalities": MODALITIES,
        "encoder": BCRNN_ENCODER_CONFIG
    })
    return BCRNN(obs_config, obs_key_shapes, ac_dim, algo_config_rnn)

def dict_to_device(d, device):
    for k,v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device)
        elif isinstance(v, dict):
            dict_to_device(v, device)
        else:
            continue
    return d

def data_to_device(batch_data, device):
    batch_data = dict_to_device(batch_data, device)
    return batch_data

def loss_func(predictions, batch):
    losses = OrderedDict()
    a_target = batch["actions"]
    actions = predictions["actions"]
    losses["loss"] = nn.MSELoss()(actions, a_target)
    return losses

def compute_loss(model, batch_data, device):
    input_batch = data_to_device(batch_data, device)
    predictions = model(input_batch)
    if isinstance(predictions, dict):
        losses = loss_func(predictions, input_batch)
    else:
        losses = {'loss': predictions}
    return losses

def eval(model, data_loader, device):
    total_losses = defaultdict(float)
    for batch in data_loader:
        batch = data_to_device(batch, device)
        batch_losses = compute_loss(model, batch, device)
        for k, v in batch_losses.items():
            total_losses[k] += v.item()/len(data_loader)
    return total_losses

if __name__=='__main__':
    model = get_model().to("cuda")
    optimizer = Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loader = tud.DataLoader(trains[0], batch_size=64, shuffle=True, drop_last=True)
    batch = next(iter(loader))
    loss = compute_loss(model, batch, "cuda")['loss']
    loss.backward()
    optimizer.step()

