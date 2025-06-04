import torch.utils.data as tud
import numpy as np

class Float32Converter(tud.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        res = self.dataset[index]
        for k,v in res.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float64: res[k] = v.astype(np.float32)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, np.ndarray) and vv.dtype == np.float64: v[kk] = vv.astype(np.float32)
        return res

class ObsPaddingDataset(tud.Dataset):
    def __init__(self, sequence_data, obs_key_shapes):
        self.sequence_data = sequence_data
        self.obs_key_shapes = obs_key_shapes

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        res = self.sequence_data[idx]
        for k in res['obs'].keys():
            dim = self.obs_key_shapes[k] if not isinstance(self.obs_key_shapes[k], list) else np.prod(np.array(self.obs_key_shapes[k]))
            if res['obs'][k].shape[1]<dim:
                res['obs'][k] = np.hstack([res['obs'][k], np.zeros((res['obs'][k].shape[0], dim-res['obs'][k].shape[1]))])
        return res