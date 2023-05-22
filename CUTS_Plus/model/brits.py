import torch
from torch import nn
from torch.autograd import Variable
from .utils import TemporalDecay, FeatureRegression


class BRITSNet(nn.Module):
    def __init__(self,
                 d_in,
                 d_hidden=64):
        super(BRITSNet, self).__init__()
        self.birits = BRITS(input_size=d_in,
                            hidden_size=d_hidden)

    def forward(self, x, mask=None, **kwargs):
        # x: [batches, steps, features]
        x = x.squeeze()
        mask = mask.squeeze()
        imputations, predictions = self.birits(x, mask=mask)
        # predictions: [batch, directions, steps, features] x 3
        out = torch.mean(imputations, dim=1)  # -> [batch, steps, features]
        out = out.unsqueeze(-1)
        # reshape
        predictions = torch.cat(predictions, 1)
        predictions = predictions.unsqueeze(-1)
        predictions = torch.transpose(predictions, 0, 1)  # rearrange(predictions, 'b d s f -> d b s f')
        if self.training:
            return out, predictions
        else:
            return out


class BRITS(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rits_fwd = RITS(input_size, hidden_size)
        self.rits_bwd = RITS(input_size, hidden_size)

    def forward(self, x, mask=None):

        def reverse_tensor(tensor=None, axis=-1):
            if tensor is None:
                return None
            if tensor.dim() <= 1:
                return tensor
            indices = range(tensor.size()[axis])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad=False).to(tensor.device)
            return tensor.index_select(axis, indices)
        
        # x: [batches, steps, features]
        # forward
        imp_fwd, pred_fwd = self.rits_fwd(x, mask)
        # backward
        x_bwd = reverse_tensor(x, axis=1)
        mask_bwd = reverse_tensor(mask, axis=1) if mask is not None else None
        imp_bwd, pred_bwd = self.rits_bwd(x_bwd, mask_bwd)
        imp_bwd, pred_bwd = reverse_tensor(imp_bwd, axis=1), [reverse_tensor(pb, axis=1) for pb in pred_bwd]
        # stack into shape = [batch, directions, steps, features]
        imputation = torch.stack([imp_fwd, imp_bwd], dim=1)
        predictions = [torch.stack([pf, pb], dim=1) for pf, pb in zip(pred_fwd, pred_bwd)]
        c_h, z_h, x_h = predictions

        return imputation, (c_h, z_h, x_h)

    @staticmethod
    def consistency_loss(imp_fwd, imp_bwd):
        loss = 0.1 * torch.abs(imp_fwd - imp_bwd).mean()
        return loss


class RITS(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64):
        super(RITS, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)

        self.rnn_cell = nn.LSTMCell(2 * self.input_size, self.hidden_size)

        self.temp_decay_h = TemporalDecay(d_in=self.input_size, d_out=self.hidden_size, diag=False)
        self.temp_decay_x = TemporalDecay(d_in=self.input_size, d_out=self.input_size, diag=True)

        self.hist_reg = nn.Linear(self.hidden_size, self.input_size)
        self.feat_reg = FeatureRegression(self.input_size)

        self.weight_combine = nn.Linear(2 * self.input_size, self.input_size)

    def init_hidden_states(self, x):
        return Variable(torch.zeros((x.shape[0], self.hidden_size))).to(x.device)

    def forward(self, x, mask=None, delta=None):
        # x : [batch, steps, features]
        steps = x.shape[-2]

        if mask is None:
            mask = torch.ones_like(x, dtype=torch.uint8)
        if delta is None:
            delta = TemporalDecay.compute_delta(mask)

        # init rnn states
        h = self.init_hidden_states(x)
        c = self.init_hidden_states(x)

        imputation = []
        predictions = []
        for step in range(steps):
            d = delta[:, step, :]
            m = mask[:, step, :]
            x_s = x[:, step, :]

            gamma_h = self.temp_decay_h(d)

            # history prediction
            x_h = self.hist_reg(h)
            x_c = m * x_s + (1 - m) * x_h
            h = h * gamma_h

            # feature prediction
            z_h = self.feat_reg(x_c)

            # predictions combination
            gamma_x = self.temp_decay_x(d)
            alpha = self.weight_combine(torch.cat([gamma_x, m], dim=1))
            alpha = torch.sigmoid(alpha)
            c_h = alpha * z_h + (1 - alpha) * x_h

            c_c = m * x_s + (1 - m) * c_h
            inputs = torch.cat([c_c, m], dim=1)
            h, c = self.rnn_cell(inputs, (h, c))

            imputation.append(c_c)
            predictions.append(torch.stack((c_h, z_h, x_h), dim=0))

        # imputation -> [batch, steps, features]
        imputation = torch.stack(imputation, dim=-2)
        # predictions -> [predictions, batch, steps, features]
        predictions = torch.stack(predictions, dim=-2)
        c_h, z_h, x_h = predictions

        return imputation, (c_h, z_h, x_h)