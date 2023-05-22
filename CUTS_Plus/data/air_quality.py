import os

import numpy as np
import pandas as pd

from .utils import disjoint_months, infer_mask, compute_mean, geographical_distance, thresholded_gaussian_kernel


class AirQuality:
    SEED = 3210

    def __init__(self, path, impute_nans=True, small=False, cluster=1, reduce=None):
        self.name = "AirQuality"
        self.random = np.random.default_rng(self.SEED)
        self.test_months = [3, 6, 9, 12]
        self.infer_eval_from = 'next'
        self.eval_mask = None
        self.reduce = reduce
        df, dist, mask = self.load(path, impute_nans=impute_nans, small=small, cluster=cluster)
        self.df = df
        self.dist = dist
        self._mask = mask
        self.weight = self.get_similarity(thr=None)

    def load_raw(self, path, small=False, cluster=1):
        # if small:
        #     h5path = os.path.join(path, 'small36.h5')
        #     eval_mask = pd.DataFrame(pd.read_hdf(path, 'eval_mask'))
        # else:
        #     h5path = os.path.join(path, 'full437.h5')
        #     eval_mask = None
        # df = pd.DataFrame(pd.read_hdf(h5path, 'pm25'))
        # stations = pd.DataFrame(pd.read_hdf(h5path, 'stations'))
        
        # df = pd.read_csv(os.path.join(path, 'airquality.csv'))
        # df = df.pivot(index='time', columns='station_id', values='PM25_Concentration')
        
        df = pd.DataFrame(pd.read_hdf(os.path.join(path, 'full437.h5'), 'pm25'))
        station = pd.read_csv(os.path.join(path, 'station.csv'))
        
        if not small:
            city = pd.read_csv(os.path.join(path, 'city.csv'))
            district = pd.read_csv(os.path.join(path, 'district.csv'))
            st1, st2 = [], []
            for st in df.columns:
                district_id = station.loc[station['station_id'] == st]['district_id'].values[0]
                city_id = district.loc[district['district_id'] == district_id]['city_id'].values[0]
                cluster_id = city.loc[city['city_id'] == city_id]['cluster_id'].values[0]
                if cluster_id == 1:
                    st1.append(st)
                else:
                    st2.append(st)
            
            station = station.set_index('station_id')
            df_cluster = [pd.DataFrame(df.loc[:, st1], df.index, st1), pd.DataFrame(df.loc[:, st2], df.index, st2)]
            station_cluster = [pd.DataFrame(station.loc[st1, :], st1, station.columns), pd.DataFrame(station.loc[st2, :], st2, station.columns)]
            
            df = df_cluster[cluster]
            station = station_cluster[cluster]
        else:
            station = station.set_index('station_id')
            df = df.iloc[:, :36]
            station = station.iloc[:36, :]
        
        if 'P' in self.reduce and (not small):
            p = float(self.reduce[:-1])
            # idx = np.random.randint(0, int((1-p)*df.shape[0])+1)
            idx = 0
            df = df.iloc[idx:idx+int(p*df.shape[0])]
        
            del_list = []
            for i in range(df.values.shape[1]):
                ivalue = df.values[:, i]
                if np.isnan(ivalue).sum() > 0.9*len(ivalue):
                    del_list.append(i)
                values = np.delete(df.values, del_list, axis=1)
                columns = np.delete(df.columns, del_list)
                df = pd.DataFrame(values, index=df.index, columns=columns)
                values = np.delete(station.values, del_list, axis=0)
                index = np.delete(station.index, del_list)
                station = pd.DataFrame(values, index=index, columns=station.columns)
        
        return df, station

    def load(self, path, impute_nans=True, small=False, cluster=1, masked_sensors=None):
        # load readings and stations metadata
        df, station = self.load_raw(path, small, cluster)
        # compute the masks
        mask = (~np.isnan(df.values)).astype('uint8')  # 1 if value is not nan else 0
        eval_mask = infer_mask(df, infer_from=self.infer_eval_from)
        eval_mask = eval_mask.values.astype('uint8')
        if masked_sensors is not None:
            eval_mask[:, masked_sensors] = np.where(mask[:, masked_sensors], 1, 0)
        self.eval_mask = eval_mask  # 1 if value is ground-truth for imputation else 0
        # eventually replace nans with weekly mean by hour
        if impute_nans:
            df = df.fillna(compute_mean(df))
            # df = df.fillna(0)
        # compute distances from latitude and longitude degrees
        self.st_coord = station.loc[:, ['latitude', 'longitude']]
        dist = geographical_distance(self.st_coord, to_rad=True).values
        return df, dist, mask

    def splitter(self, dataset, val_len=1., in_sample=False, window=0):
        nontest_idxs, test_idxs = disjoint_months(dataset, months=self.test_months)
        if in_sample:
            train_idxs = np.arange(len(dataset))
            val_months = [(m - 1) % 12 for m in self.test_months]
            _, val_idxs = disjoint_months(dataset, months=val_months)
        else:
            # take equal number of samples before each month of testing
            val_len = (int(val_len * len(nontest_idxs)) if val_len < 1 else val_len) // len(self.test_months)
            # get indices of first day of each testing month
            delta_idxs = np.diff(test_idxs)
            end_month_idxs = test_idxs[1:][np.flatnonzero(delta_idxs > delta_idxs.min())]
            if len(end_month_idxs) < len(self.test_months):
                end_month_idxs = np.insert(end_month_idxs, 0, test_idxs[0])
            # expand month indices
            month_val_idxs = [np.arange(v_idx - val_len, v_idx) - window for v_idx in end_month_idxs]
            val_idxs = np.concatenate(month_val_idxs) % len(dataset)
            # remove overlapping indices from training set
            ovl_idxs, _ = dataset.overlapping_indices(nontest_idxs, val_idxs, as_mask=True)
            train_idxs = nontest_idxs[~ovl_idxs]
        return [train_idxs, val_idxs, test_idxs]

    def get_similarity(self, thr=0.1, include_self=False, force_symmetric=False, sparse=False, **kwargs):
        theta = np.std(self.dist[:36, :36])  # use same theta for both air and air36
        adj = thresholded_gaussian_kernel(self.dist, theta=theta, threshold=thr)
        if not include_self:
            adj[np.diag_indices_from(adj)] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
        return adj
    
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
        self.eval_mask = (mask | (rand(mask.shape) < p_noise)).astype('uint8')

    @property
    def mask(self):
        return self._mask
    
    @property
    def length(self):
        return self.df.values.shape[0]

    @property
    def n_nodes(self):
        return self.df.values.shape[1]

    @property
    def training_mask(self):
        return self._mask if self.eval_mask is None else (self._mask & (1 - self.eval_mask))

    def test_interval_mask(self, dtype=bool, squeeze=True):
        m = np.in1d(self.df.index.month, self.test_months).astype(dtype)
        if squeeze:
            return m
        return m[:, None]
