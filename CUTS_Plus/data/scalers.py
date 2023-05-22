from abc import ABC
import numpy as np
import torch

class Scaler(ABC):
    def __init__(self, offset=0., scale=1.):
        self.bias = offset
        self.scale = scale
        super(Scaler, self).__init__()

    def params(self):
        return dict(bias=self.bias, scale=self.scale)

    def fit(self, x, mask=None, keepdims=True):
        pass

    def transform(self, x):
        if not isinstance(x, np.ndarray):
            return torch.tensor((x.cpu().numpy() - self.bias) / self.scale).to(x.device)
        else:
            return (x - self.bias) / (self.scale + 1e-8)

    def inverse_transform(self, x):
        if not isinstance(x, np.ndarray):
            return torch.tensor(x.cpu().numpy() * self.scale + self.bias).to(x.device)
        else:
            return x * self.scale + self.bias

    def fit_transform(self, x, mask=None, keepdims=True):
        self.fit(x, mask, keepdims)
        return self.transform(x)


class StandardScaler(Scaler):
    def __init__(self, axis=0):
        self.axis = axis
        super(StandardScaler, self).__init__()

    def fit(self, x, mask=None, keepdims=True):
        if mask is not None:
            x = np.where(mask, x, np.nan)
            self.bias = np.nanmean(x, axis=self.axis, keepdims=keepdims)
            self.scale = np.nanstd(x, axis=self.axis, keepdims=keepdims)
        else:
            self.bias = x.mean(axis=self.axis, keepdims=keepdims)
            self.scale = x.std(axis=self.axis, keepdims=keepdims)


class MinMaxScaler(Scaler):
    def __init__(self, axis=0):
        self.axis = axis
        super(MinMaxScaler, self).__init__()

    def fit(self, x, mask=None, keepdims=True):
        if mask is not None:
            x = np.where(mask, x, np.nan)
            self.bias = np.nanmin(x, axis=self.axis, keepdims=keepdims)
            self.scale = (np.nanmax(x, axis=self.axis, keepdims=keepdims) - self.bias)
        else:
            self.bias = x.min(axis=self.axis, keepdims=keepdims)
            self.scale = (x.max(axis=self.axis, keepdims=keepdims) - self.bias)
        return self
