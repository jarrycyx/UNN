from einops import rearrange
import torch
import torch.nn as nn
import numpy as np
from nets.time_series_attention import TimeSeries_SelfAttn

class MLP(nn.Module):
    def __init__(self, in_dim, n_hid, out_dim, n_layer, dropout=0):
        """Component for encoder and decoder

        Args:
            in_dim (int): input dimension.
            n_hid (int): model layer dimension.
            out_dim (int): output dimension.
        """
        super(MLP, self).__init__()
        dims = (
            [(in_dim, n_hid)]
            + [(n_hid, n_hid) for _ in range(n_layer - 1)]
            + [(n_hid, out_dim)]
        )
        fc_layers = [nn.Linear(pair[0], pair[1]) for pair in dims]
        # bn_layers = [BatchNorm(n_hid) for _ in range(n_layer)]
        lr_layers = [nn.LeakyReLU(0.05) for _ in range(n_layer)]
        layers = []
        for i in range(n_layer):
            layers.append(fc_layers[i])
            # layers.append(bn_layers[i])
            layers.append(lr_layers[i])
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
                
               

class MultiMLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, mlp_layers, mlp_num):
        super(MultiMLP, self).__init__()
        self.fitting_model = nn.ModuleList([MLP(in_dim, hid_dim, out_dim, mlp_layers) for _ in range(mlp_num)])
        
    def forward(self, x, mask):
        b, n, m, t, d = x.shape
        x_masked = torch.einsum("bnmtd,bnmt->bnmtd", x, mask)
        # x_masked = x
        y_preds = []
        for i in range(n):
            x_i_masked = rearrange(x_masked[:, i], "b m t d -> b (m t d)")
            y_pred_i = self.fitting_model[i](x_i_masked)
            y_preds.append(rearrange(y_pred_i, "b (t d) -> b 1 t d", d=d))
        return torch.concat(y_preds, axis=1)





class MultiLSTM(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, mlp_layers, mlp_num):
        super(MultiLSTM, self).__init__()
        self.fitting_model = nn.ModuleList([
            nn.ModuleList([
                nn.LSTM(in_dim, hid_dim, num_layers=3, batch_first=True),
                MLP(hid_dim, hid_dim, out_dim, mlp_layers)
                ]) 
            for _ in range(mlp_num)])
        
    def forward(self, x, mask, ref="zero"):
        x = rearrange(x, "b t n d -> b n t d") # b t n d -> b n t d
        x = x[:, None, :, :, :].expand(-1, len(self.fitting_model), -1, -1, -1) # b n t d -> b n n t d
        b, n, m, t, d = x.shape
        x_masked = torch.einsum("bnmtd,bnm->bnmtd", x, mask)
        # x_masked = x
        y_preds = []
        for i in range(n):
            x_i_masked = rearrange(x_masked[:, i], "b m t d -> b t (m d)")
            lstm_out, hid = self.fitting_model[i][0](x_i_masked)
            y_pred_i = self.fitting_model[i][1](lstm_out[:, -1])
            y_preds.append(rearrange(y_pred_i[:,0], "b -> b 1"))
        return torch.concat(y_preds, axis=1)
    
    
    
class GraphImputNet(nn.Module):
    def __init__(self, pred_model, pred_window=10):
        super(GraphImputNet, self).__init__()
        self.pred_model = pred_model
        self.pred_window = pred_window
        
    def forward(self, x, mask, ref="zero"):
        b, t, n, d = x.shape
        x_hat_list = [] # drop absence feature, b t n d -> b n t
        # for s_t in range(0, t-self.pred_window):
        for s_t in range(0, t-self.pred_window, 10):
            x_input = x[:, s_t:s_t+self.pred_window, :, :] # b t n d -> b w n d
            x_hat_t = self.pred_model(x_input, mask) # b w n d -> b n
            x_hat_list.append(rearrange(x_hat_t, "b n -> b n 1"))
        
        x_hat = torch.concat(x_hat_list, axis=-1) # b n t
        return x_hat # b n t