import numpy as np
import pandas as pd
from .scalers import StandardScaler

class PandasDataset:
    def __init__(self, name=None, data_path=None, dist_path=None, reduce="1P", scale=True):
        """
        Initialize a tsl dataset from a pandas dataframe.
        """
        super().__init__()
        self.name = name
        self.data_path = data_path
        self.dist_path = dist_path
        self.reduce = reduce
        self.scaler = StandardScaler() if scale else None

        if 'T' in self.reduce:
            self.samples_per_day = int(60 / int(reduce[:-1]) * 24)


    def load(self, thr=0.1):
        self.df = pd.read_csv(self.data_path, index_col=0).astype(np.float32)
        if 'P' in self.reduce:
            p = float(self.reduce[:-1])
            idx = np.random.randint(0, int((1-p)*self.df.shape[0])+1)
            self.df = self.df.iloc[idx:idx+int(p*self.df.shape[0])]
        elif 'T' in self.reduce:
            self.df.index = pd.to_datetime(self.df.index)
            self.resample_(self.reduce)
        
        if self.name == 'electricity':
            values = self.df.values
            # std = np.nanstd(values, axis=0)
            # col = np.argmax(std)
            # values = np.delete(values, col, axis=1)
            # columns = np.delete(self.df.columns, col)
            self.df = pd.DataFrame(values, index=self.df.index, columns=columns)
        elif self.name == 'traffic':
            values = np.hstack((self.df.values, np.zeros((len(self.df.index), 2)).astype(np.float32)))
            add_col = np.arange(len(self.df.columns), len(self.df.columns)+2)
            columns = np.hstack((self.df.columns, [str(col) for col in add_col]))
            self.df = pd.DataFrame(values, index=self.df.index, columns=columns)
        
        self.mask = ~np.isnan(self.df.values)
        if self.mask is not None:
            self.mask = np.asarray(self.mask).astype('uint8')
        self.df.fillna(method='ffill', axis=0, inplace=True)
        
        if self.scaler is not None:
            # self.scaler.fit(self.df.values, self.mask)
            # values = self.scaler.transform(self.df.values)
            values = self.df.values
            for i in range(values.shape[1]):
                values[:, i] = self.scaler.fit_transform(values[:, i], self.mask[:, i])
            self.df = pd.DataFrame(values, index=self.df.index, columns=self.df.columns)

        if self.dist_path is not None:
            self.dist = np.load(self.dist_path).astype(np.float32)
            # self.dist = geographical_distance(self.pos, to_rad=True).values
            finite_dist = self.dist.reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            theta = finite_dist.std()
            self.weight = np.exp(-np.square(self.dist / theta))
            self.weight[self.weight < thr] = 0.
        else:
            self.weight = np.ones((self.df.shape[1], self.df.shape[1]))

    
    def load_data(self, data, graph):
        self.df = pd.DataFrame(data).astype(np.float32)
        self.mask = np.ones_like(self.df.values).astype('uint8')
        self.weight = graph.astype(np.float32)
    

    def splitter(self, val_len=0, test_len=0, window=0):
        if self.name == 'dream':
            idx = np.arange(self.length // 21)
            if test_len > 0:
                test_len = int(test_len * len(idx))
            if val_len > 0:
                val_len = int(val_len * len(idx))
            test_start = (len(idx) - test_len) * 21
            val_start = (len(idx) - test_len - val_len) * 21
        else:
            idx = np.arange(self.length)
            if test_len > 0:
                test_len = int(test_len * len(idx))
            if val_len > 0:
                val_len = int(val_len * len(idx))
            test_start = len(idx) - test_len
            val_start = len(idx) - test_len - val_len
        
        return [np.arange(val_start - window), np.arange(val_start, test_start - window), np.arange(test_start, self.length-window)]
    

    def generate_mask(self, p_block, p_noise, max_seq, min_seq):
        rand = np.random.random
        randint = np.random.randint
        mask = rand(self.df.shape) < p_block
        for col in range(mask.shape[1]):
            idxs = np.flatnonzero(mask[:, col])
            if not len(idxs):
                continue
            fault_len = min_seq
            if max_seq > min_seq:
                fault_len = fault_len + int(randint(max_seq - min_seq))
            idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
            idxs = np.unique(idxs_ext)
            idxs = np.clip(idxs, 0, mask.shape[0] - 1)
            mask[idxs, col] = True
        self.eval_mask = mask | (rand(mask.shape) < p_noise)
    
    
    def resample_(self, reduce, aggr='nearest'):
        resampler = self.df.resample(reduce)
        if aggr == 'sum':
            self.df = resampler.sum()
        elif aggr == 'mean':
            self.df = resampler.mean()
        elif aggr == 'nearest':
            self.df = resampler.nearest()
        else:
            raise ValueError(f'{aggr} if not a valid aggregation method.')
    

    @property
    def training_mask(self):
        return self.mask if self.eval_mask is None else (self.mask & (1 - self.eval_mask)).astype(np.uint8)

    @property
    def length(self):
        return self.df.values.shape[0]

    @property
    def n_nodes(self):
        return self.df.values.shape[1]
