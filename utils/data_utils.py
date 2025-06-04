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