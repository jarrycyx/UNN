import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from .utils import DataEmbedding
from .utils import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len,
                 pred_len=0,
                 top_k=3,
                 d_model=64,
                 d_ff=64,
                 num_kernels=6):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = x[:, -(length - (self.seq_len + self.pred_len)):, :]
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(nn.Module):
    def __init__(self, d_in,
                 seq_len,
                 pred_len=0,
                 d_model=64,
                 d_ff=64,
                 e_layers=2,
                 top_k=3,
                 num_kernels=6,
                 dropout=0.1,
                 freq='h',
                 embed='timeF'):
        super(TimesNet, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model = nn.ModuleList([TimesBlock(seq_len=seq_len,
                                               pred_len=pred_len,
                                               top_k=top_k,
                                               d_model=d_model,
                                               d_ff=d_ff,
                                               num_kernels=num_kernels) for _ in range(e_layers)])
        self.enc_embedding = DataEmbedding(d_in, d_model, embed, freq, dropout)
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, d_in, bias=True)

    def forward(self, x, mask=None, x_mark_enc=None, **kwargs):
        x = x.squeeze()
        mask = mask.squeeze() if mask is not None else None
        
        # Normalization from Non-stationary Transformer
        
        # means = torch.sum(x, dim=1) / torch.sum(mask == 1, dim=1)
        # means = means.unsqueeze(1).detach()
        # x = x - means
        # x = x.masked_fill(mask == 0, 0)
        # stdev = torch.sqrt(torch.sum(x * x, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        # stdev = stdev.unsqueeze(1).detach()
        # x /= stdev

        # embedding
        enc_out = self.enc_embedding(x, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        
        # dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        # dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.unsqueeze(-1)
        if self.training:
            return dec_out, [dec_out]
        else:
            return dec_out
