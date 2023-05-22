import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from .scalers import StandardScaler, MinMaxScaler

class TorchDataset(Dataset):
    def __init__(self,
                 data,
                 mask,
                 eval_mask,
                 window=24,
                 stride=1,
                 type='standard'):
        
        super(TorchDataset, self).__init__()
        # Store data
        self.data = data.values[..., np.newaxis]
        self.mask = mask[..., np.newaxis]
        self.eval_mask = eval_mask[..., np.newaxis]
        self.index = data.index
        # Store offset information
        self.window = window
        self.stride = stride
        # Identify the indices of the samples
        self._indices = np.arange(self.data.shape[0] - self.window + 1)
        self.set_scaler(type)

    def __getitem__(self, item):
        return self.get(item)

    def __len__(self):
        return len(self._indices)

    def __repr__(self):
        return "{}(n_samples={})".format(self.__class__.__name__, len(self.data.shape[0]))

    # Getter and setter for data

    def set_scaler(self, type='standard'):
        if type == 'standard':
            self.scaler = StandardScaler(axis=0)
            self.scaler.fit(self.data, self.mask)
        elif type == 'minmax':
            self.scaler = MinMaxScaler(axis=0)
            self.scaler.fit(self.data, self.mask)

    # Dataset properties

    @property
    def n_steps(self):
        return self.data.shape[0]

    @property
    def n_channels(self):
        return self.data.shape[-1]

    # Item getters

    def get(self, item, preprocess=True):
        idx = self._indices[item]
        res = dict()
        
        if self.stride != 1:
            if idx // self.stride != (idx + self.window) // self.stride:
                idx = (idx + self.window) // self.stride * self.stride - self.window
        res['x'] = res['y'] = self.data[idx:idx + self.window]
        res['mask'] = self.mask[idx:idx + self.window]
        res['eval_mask'] = self.eval_mask[idx:idx + self.window]
        
        if preprocess:
            res['x'] = self.scaler.transform(res['x'])
            res['y'] = self.scaler.transform(res['y'])
        
        return res

    # Data utilities

    def expand_indices(self, indices=None, merge=False):
        indices = np.arange(len(self._indices)) if indices is None else indices
        ds_indices = np.array([np.arange(idx, idx + self.window) for idx in self._indices[indices]])
        if merge:
            ds_indices = np.concatenate(ds_indices)
            ds_indices = np.unique(ds_indices)
        return ds_indices

    def data_timestamps(self, indices=None):
        ds_indices = self.expand_indices(indices, merge=False)
        ds_timestamps = self.index[ds_indices]
        return ds_timestamps
    
    def predict_dataframe(self, y=None, index=None, columns=None):
        dfs = [pd.DataFrame(data=data.reshape(data.shape[:2]), index=idx, columns=columns) for data, idx in zip(y, index)]
        df = pd.concat(dfs)
        step_df = df.groupby(df.index)
        step_df = step_df.mean()
        return step_df
    
    def overlapping_indices(self, idxs1, idxs2, as_mask=False):
        ts1 = self.data_timestamps(idxs1)
        ts2 = self.data_timestamps(idxs2)
        common_ts = np.intersect1d(np.unique(ts1), np.unique(ts2))
        is_overlapping = lambda sample: np.any(np.in1d(sample, common_ts))
        m1 = np.apply_along_axis(is_overlapping, 1, ts1)
        m2 = np.apply_along_axis(is_overlapping, 1, ts2)
        if as_mask:
            return m1, m2
        return np.sort(idxs1[m1]), np.sort(idxs2[m2])
