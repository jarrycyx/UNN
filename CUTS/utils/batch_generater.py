
import numpy as np
import torch
from einops import rearrange



def batch_generater(data, observ_mask, bs, n_nodes, input_step, pred_step, sample_period=1):
    t, n, d = data.shape
    first_sample_t = input_step
    random_t_list = np.arange(first_sample_t, t, sample_period).tolist()
    np.random.shuffle(random_t_list)

    for batch_i in range(len(random_t_list) // bs):
        x = torch.zeros([bs, n_nodes, n_nodes, input_step, d]).to(data.device)
        y = torch.zeros([bs, n_nodes, pred_step, d]).to(data.device)
        t = torch.zeros([bs]).to(data.device).long()
        mask = torch.zeros([bs, n_nodes, pred_step, d]).to(data.device)
        for data_i in range(bs):
            data_t = random_t_list.pop()
            x_data = rearrange(data[data_t-input_step : data_t, :], "t n d -> 1 n t d")
            x[data_i, :, :, :] = x_data.expand(n_nodes, -1, -1, -1)
            y[data_i, :, :, :] = rearrange(data[data_t : data_t+pred_step*sample_period : sample_period, :], "t n d -> n t d")
            t[data_i] = data_t
            mask[data_i, :, :, :] = rearrange(observ_mask[data_t:data_t+pred_step, :], "t n d -> n t d")

        yield x, y, t, mask