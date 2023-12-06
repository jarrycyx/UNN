import os, sys
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops

sys.path.append(opj(opd(__file__), ".."))

from einops import rearrange
import torch
import torch.nn as nn
import numpy as np
from nets.time_series_attention import TimeSeries_SelfAttn
from nets.gumbel_softmax import gumbel_softmax
from nets.prob_graph import bernonlli_sample, gumbel_sample


def causal_masking(x_dy, x_st, mask_dy, mask_st, ref):
    if isinstance(ref, str):
        if ref == "absent_feat":
            dy_ref = torch.zeros_like(x_dy)
            dy_ref[:, :, :, 1] = 1
            st_ref = torch.zeros_like(x_st)
            st_ref[:, :, 1] = 1
        elif ref == "zero":
            dy_ref = torch.zeros_like(x_dy)
            st_ref = torch.zeros_like(x_st)
    elif isinstance(ref, torch.Tensor):
        assert ref.shape == mask_dy.shape[1] + mask_st.shape[1], "ref shape should be (n_dy+n_st)"
        dy_ref = ref[None, : mask_dy.shape[1]].repeat(x_dy.shape[0], x_dy.shape[1], 1)
        dy_ref[x_dy[:, :, :, 1] == 1] = 0
        dy_ref_abs = dy_ref == 0
        dy_ref = torch.cat([dy_ref, dy_ref_abs], axis=-1)

        st_ref = ref[None, mask_dy.shape[1]:, None].repeat(x_st.shape[0], 1, 2)
        st_ref[:, :, 1] = 0
    else:
        raise ValueError("ref should be absent_feat or zero or tensor of shape (n_dy+n_st)")

    if len(mask_dy.shape) == 3:
        x_dy_masked = torch.einsum("blnd,bln->blnd", x_dy, mask_dy) + torch.einsum("blnd,bln->blnd", dy_ref, 1 - mask_dy)
    else:
        x_dy_masked = torch.einsum("blnd,bn->blnd", x_dy, mask_dy) + torch.einsum("blnd,bn->blnd", dy_ref, 1 - mask_dy)

    x_st_masked = torch.einsum("bnd,bn->bnd", x_st, mask_st) + torch.einsum("bnd,bn->bnd", st_ref, 1 - mask_st)

    return x_dy_masked, x_st_masked


class MLP(nn.Module):
    def __init__(self, in_dim, n_hid, out_dim, n_layer, dropout=0, act="leakyrelu"):
        """Component for encoder and decoder

        Args:
            in_dim (int): input dimension.
            n_hid (int): model layer dimension.
            out_dim (int): output dimension.
        """
        super(MLP, self).__init__()
        dims = [(in_dim, n_hid)] + [(n_hid, n_hid) for _ in range(n_layer - 1)] + [(n_hid, out_dim)]
        fc_layers = [nn.Linear(pair[0], pair[1]) for pair in dims]
        # bn_layers = [BatchNorm(n_hid) for _ in range(n_layer)]

        if act == "leakyrelu":
            act_fn = nn.LeakyReLU(0.05)
        elif act == "relu":
            act_fn = nn.ReLU()
        elif act == "tanh":
            act_fn = nn.Tanh()
        elif act == "sigmoid":
            act_fn = nn.Sigmoid()
        else:
            raise NotImplementedError

        act_layers = [act_fn for _ in range(n_layer)]
        layers = []
        for i in range(n_layer):
            layers.append(fc_layers[i])
            # layers.append(bn_layers[i])
            layers.append(act_layers[i])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(fc_layers[-1])
        self.network = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x):
        return self.network(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


class MaskedMLP(nn.Module):
    def __init__(self, in_dim, n_hid, out_dim, n_layer, dropout=0, act="leakyrelu"):
        """Component for encoder and decoder

        Args:
            in_dim (int): input dimension.
            n_hid (int): model layer dimension.
            out_dim (int): output dimension.
        """
        super(MaskedMLP, self).__init__()
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.out_dim = out_dim

        dims = [(in_dim, n_hid)] + [(n_hid, n_hid) for _ in range(n_layer - 1)] + [(n_hid, out_dim)]
        fc_layers = [nn.Linear(pair[0], pair[1]) for pair in dims]
        # bn_layers = [BatchNorm(n_hid) for _ in range(n_layer)]
        # lr_layers = [nn.LeakyReLU(0.05) for _ in range(n_layer)]

        if isinstance(out_dim, list):
            print("out_dim is a list, only predict the first element")
            out_dim = 1

        if act == "leakyrelu":
            act_fn = nn.LeakyReLU(0.05)
        elif act == "relu":
            act_fn = nn.ReLU()
        elif act == "tanh":
            act_fn = nn.Tanh()
        elif act == "sigmoid":
            act_fn = nn.Sigmoid()
        else:
            raise NotImplementedError

        act_layers = [act_fn for _ in range(n_layer)]
        layers = []
        for i in range(n_layer):
            layers.append(fc_layers[i])
            # layers.append(bn_layers[i])
            layers.append(act_layers[i])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(fc_layers[-1])
        self.net = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x, x_st=None, mask=None, mask_st=None, ref="zero"):
        self.forward(x, mask, ref)

    def forward(self, x, x_st=None, mask=None, mask_st=None, ref="zero"):
        x_masked, _ = causal_masking(x, x_st, mask, mask_st, ref)

        b, l, n, d = x.shape
        x_masked = rearrange(x_masked, "b l n d -> b (l n d)")

        y_pred = self.net(x_masked)
        # y_prob = nn.functional.softmax(y_pred, dim=1)
        return y_pred

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


class MaskedLSTM(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, mlp_layers):
        super(MaskedLSTM, self).__init__()
        self.fitting_model = nn.ModuleList([nn.LSTM(in_dim, hid_dim, num_layers=3, batch_first=True), MLP(hid_dim, hid_dim, out_dim, mlp_layers)])

    def forward(self, x, mask):
        b, n, d = x.shape
        x_masked = torch.einsum("bnd,bd->bnd", x, mask)
        # x_masked = x
        lstm_out, hid = self.fitting_model[0](x_masked)
        y_pred = self.fitting_model[1](lstm_out[:, -1])
        # y_porb = nn.functional.softmax(y_pred, dim=1)
        return y_pred


class MultitaskLSTM(nn.Module):
    def __init__(self, dynamic_dim, static_dim, hid_dim, task_dims, mlp_layers, dropout=0):
        super(MultitaskLSTM, self).__init__()
        self.task_dims = task_dims
        self.fitting_model = nn.ModuleList(
            [nn.LSTM(dynamic_dim, hid_dim, num_layers=3, batch_first=True, dropout=dropout), MLP(static_dim, hid_dim, hid_dim, mlp_layers, dropout=dropout), MLP(hid_dim * 2, hid_dim, hid_dim, mlp_layers, dropout=dropout)]
        )
        self.multitask_output = nn.ModuleList([MLP(hid_dim, hid_dim, out_dim, mlp_layers, dropout=dropout) for out_dim in task_dims])

    def forward(self, x_dy, x_st, mask_dy, mask_st, ref="zero", **kwargs):
        x_dy_masked, x_st_masked = causal_masking(x_dy, x_st, mask_dy, mask_st, ref)

        x_dy_masked = rearrange(x_dy_masked, "b l n d -> b l (n d)")
        x_st_masked = rearrange(x_st_masked, "b n d -> b (n d)")
        lstm_out, hid = self.fitting_model[0](x_dy_masked)
        mlp_out = self.fitting_model[1](x_st_masked)
        fusion_in = torch.concat([lstm_out[:, -1], mlp_out], axis=1)
        fitting_model_out = self.fitting_model[2](fusion_in)

        tasks_pred = []
        for i, _ in enumerate(self.task_dims):
            task_out = self.multitask_output[i](fitting_model_out)
            tasks_pred.append(task_out)
        return tasks_pred


class MultitaskMLP(nn.Module):
    def __init__(self, dynamic_dim, static_dim, hid_dim, task_dims, mlp_layers, dropout=0):
        super(MultitaskMLP, self).__init__()
        self.task_dims = task_dims
        self.fitting_model = nn.ModuleList([MLP(dynamic_dim, hid_dim, hid_dim, mlp_layers, dropout=dropout), MLP(static_dim, hid_dim, hid_dim, mlp_layers, dropout=dropout), MLP(hid_dim * 2, hid_dim, hid_dim, mlp_layers, dropout=dropout)])
        self.multitask_output = nn.ModuleList([MLP(hid_dim, hid_dim, out_dim, mlp_layers, dropout=dropout) for out_dim in task_dims])

    def forward(self, x_dy, x_st, mask_dy, mask_st, ref="zero", **kwargs):
        x_dy_masked, x_st_masked = causal_masking(x_dy, x_st, mask_dy, mask_st, ref)

        x_dy_masked = rearrange(x_dy_masked, "b l n d -> b (l n d)")
        x_st_masked = rearrange(x_st_masked, "b n d -> b (n d)")
        mlp_out_dy = self.fitting_model[0](x_dy_masked)
        mlp_out = self.fitting_model[1](x_st_masked)
        fusion_in = torch.concat([mlp_out_dy, mlp_out], axis=1)
        fitting_model_out = self.fitting_model[2](fusion_in)

        tasks_pred = []
        for i, _ in enumerate(self.task_dims):
            task_out = self.multitask_output[i](fitting_model_out)
            tasks_pred.append(task_out)
        return tasks_pred


class MultitaskTransformer(nn.Module):
    def __init__(self, dynamic_num, static_num, dynamic_dim, static_dim, hid_dim, task_dims, mlp_layers, 
                 dropout=0, local_expl=True, time_length=168, 
                 local_sample_type="gumbel", local_sigma=0.5, local_time_cumu_type="prod", local_time_cumulative=True, local_time_chunk_num=14):
        super(MultitaskTransformer, self).__init__()
        self.task_dims = task_dims
        self.local_expl = local_expl
        self.local_sigma = local_sigma
        self.local_time_cumu_type = local_time_cumu_type
        self.local_time_cumulative = local_time_cumulative
        self.time_length = time_length
        self.local_time_chunk_num = local_time_chunk_num
        self.local_sample_type = local_sample_type
        self.fitting_model = nn.ModuleList(
            [
                TimeSeries_SelfAttn(window=time_length, out_dim=hid_dim, in_dim=dynamic_dim, n_multiv=dynamic_num, drop_prob=dropout, d_k=128, d_v=128, d_model=128, d_inner=128, n_layers=2, n_head=8),
                MLP(static_num * static_dim, hid_dim, hid_dim, mlp_layers, dropout=dropout),
                MLP(hid_dim * 2, hid_dim, hid_dim, mlp_layers, dropout=dropout),
            ]
        )
        if local_expl:
            self.local_expl_model = nn.ModuleList(
                [
                    TimeSeries_SelfAttn(window=time_length, out_dim=dynamic_num * local_time_chunk_num, in_dim=dynamic_dim, n_multiv=dynamic_num, drop_prob=dropout, d_k=32, d_v=32, d_model=64, d_inner=128, n_layers=2, n_head=4),
                    MLP(static_num * static_dim, hid_dim, static_num, mlp_layers, dropout=dropout),
                ]
            )
        self.multitask_output = nn.ModuleList([MLP(hid_dim, hid_dim, out_dim, mlp_layers, dropout=dropout) for out_dim in task_dims])

    def forward(self, x_dy, x_st, mask_dy, mask_st, ref="zero", tau=0.1, suspend_local_expl=False):

        # Global masking
        x_dy_masked1, x_st_masked1 = causal_masking(x_dy, x_st, mask_dy, mask_st, ref)

        # Local masking
        if self.local_expl:
            mu_dy_local = self.local_expl_model[0](x_dy)
            x_st = rearrange(x_st, "b n d -> b (n d)")
            mu_st_local = self.local_expl_model[1](x_st)
            
            if self.local_sample_type == "gumbel":
                mu_dy_local = rearrange(mu_dy_local, "b (l n) -> b l n", l=self.local_time_chunk_num)
                mask_dy_local, prob_dy_local = gumbel_sample(mu_dy_local, batch_size=x_st.shape[0], tau=tau, t_length=self.time_length, 
                                                            time_cumu_type=self.local_time_cumu_type, time_cumulative=self.local_time_cumulative,
                                                            batch_dim=True, time_dim=True, causalgraph_2d=False)
                mask_st_local, prob_st_local = gumbel_sample(mu_st_local, batch_size=x_st.shape[0], tau=tau, t_length=self.time_length, 
                                                            batch_dim=True, time_dim=False, causalgraph_2d=False)
            elif self.local_sample_type == "gaussian":
                mu_dy_local = rearrange(mu_dy_local, "b (l n) -> b l n", l=self.time_length)
                gaussian_noise_dy = torch.randn_like(mu_dy_local) * self.local_sigma
                mask_dy_local = torch.clamp(0.5 + mu_dy_local + gaussian_noise_dy, 0, 1)
                gaussian_noise_st = torch.randn_like(mu_st_local) * self.local_sigma
                mask_st_local = torch.clamp(0.5 + mu_st_local + gaussian_noise_st, 0, 1)
            else:
                raise NotImplementedError
            
            if not suspend_local_expl:
                x_dy_masked, x_st_masked = causal_masking(x_dy_masked1, x_st_masked1, mask_dy_local, mask_st_local, ref)
            else:
                x_dy_masked, x_st_masked = x_dy_masked1, x_st_masked1
        else:
            x_dy_masked, x_st_masked = x_dy_masked1, x_st_masked1
            

        attention_out = self.fitting_model[0](x_dy_masked)
        x_st_masked = rearrange(x_st_masked, "b n d -> b (n d)")
        mlp_out = self.fitting_model[1](x_st_masked)
        # print(attention_out.shape, mlp_out.shape)
        fusion_in = torch.concat([attention_out, mlp_out], axis=1)
        fitting_model_out = self.fitting_model[2](fusion_in)
        tasks_pred = []
        for i, _ in enumerate(self.task_dims):
            task_out = self.multitask_output[i](fitting_model_out)
            tasks_pred.append(task_out)

        if self.local_expl:
            return tasks_pred, prob_dy_local, prob_st_local
        else:
            return tasks_pred


if __name__ == "__main__":
    x_dy = torch.zeros(128, 168, 99, 2).cuda()
    x_st = torch.zeros(128, 3, 2).cuda()
    mask_dy = torch.zeros(128, 168, 99).cuda()
    mask_st = torch.zeros(128, 3).cuda()

    net = MultitaskTransformer(99, 3, 2, 2, 32, [2, 2, 2], 3).cuda()
    y_list = net(x_dy, x_st, mask_dy, mask_st=mask_st)
    for y in y_list:
        print(y)
