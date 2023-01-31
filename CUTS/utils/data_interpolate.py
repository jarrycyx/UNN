from scipy import interpolate
import numpy as np
from copy import deepcopy
from einops import rearrange
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


def interp_masked_data(data, mask, interp="zoh"):
    if interp == "zoh":
        T, D = data.shape
        new_data = deepcopy(data)
        for i in range(1,T):
            new_data[i] = data[i]*mask[i] + new_data[i-1]*(1-mask[i])
        return new_data
    else:
        x = np.argwhere(mask>0)[:,0]
        func = interpolate.interp1d(x, data[x,:], kind=interp, 
                                    axis=0, copy=True, fill_value="extrapolate")
        new_x = np.arange(0, data.shape[0])
        return func(new_x)

def interp_multivar_data(data, mask, interp="zoh"):
    if interp == "GP":
        return interp_multivar_with_gauss_process(data, mask)
    else:
        new_data = np.zeros_like(data)
        T, N, D = data.shape
        for node_i in range(N):
            new_data[:,node_i,:] = interp_masked_data(data[:,node_i,:], mask[:,node_i], interp=interp)
        return new_data


def interp_multivar_with_gauss_process(data, mask):
    data = data * mask
    x = rearrange(data[:-1], "t n d -> t (n d)")
    y = rearrange(data[1:], "t n d -> t (n d)")
        
    gpr = GaussianProcessRegressor(random_state=0).fit(x, y)
    pred = gpr.predict(y)
    pred = rearrange(pred, "t (n d) -> t n d", n=data.shape[1])

    new_data = np.zeros_like(data)
    new_data[1:] = (mask * data)[1:] + (1 - mask)[1:] * pred
    new_data[0] = data[0]
    
    return new_data


if __name__=="__main__":
    # data = np.array([[0,0],[1,2],[0,0],[2,3]])
    # mask = np.array([0,1,0,1])
    # print(interp_masked_data(data, mask))
    
    interp_multivar_with_gauss_process()