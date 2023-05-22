import torch
from einops import rearrange
from torch import nn
from torch.autograd import Variable

class GRUCutsNet(nn.Module):
    def __init__(self, n_nodes,
                 d_in=1,
                 d_hidden=64,
                 d_emb=8,
                 n_layers=1,
                 merge='mlp'):
        super(GRUCutsNet, self).__init__()
        self.in_ch = d_in
        self.hidden_ch = d_hidden
        self.emb = nn.parameter.Parameter(torch.empty(d_emb, n_nodes))
        nn.init.kaiming_normal_(self.emb, nonlinearity='relu')
        
        self.fwd_module = FrameBlock(n_nodes, d_in, d_hidden, n_layers)
        self.bwd_module = FrameBlock(n_nodes, d_in, d_hidden, n_layers)
        
        if merge == 'mlp':
            self._impute_from_states = True
            self.out = nn.Sequential(
                nn.Conv2d(in_channels=2 * d_hidden + d_in + d_emb,
                          out_channels=d_hidden, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=d_hidden, out_channels=d_in, kernel_size=1)
            )
        elif merge in ['mean', 'sum', 'min', 'max']:
            self._impute_from_states = False
            self.out = getattr(torch, merge)
        else:
            raise ValueError("Merge option %s not allowed." % merge)
    
    def forward(self, x, mask, fwd_graph, bwd_graph=None, ind_graph=None):
        
        def reverse_tensor(tensor=None, axis=-1):
            if tensor is None:
                return None
            if tensor.dim() <= 1:
                return tensor
            indices = range(tensor.size()[axis])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad=False).to(tensor.device)
            return tensor.index_select(axis, indices)
        
        x = rearrange(x, 'b s n c -> b c n s')
        mask = rearrange(mask, 'b s n c -> b c n s').byte()
        if bwd_graph is None:
            bwd_graph = torch.zeros_like(fwd_graph).to(fwd_graph.device)
        if ind_graph is None:
            ind_graph = torch.zeros_like(fwd_graph).to(fwd_graph.device)
        
        fwd_graph = fwd_graph / (fwd_graph.sum(1, keepdim=True) + 1e-6)
        bwd_graph = bwd_graph / (bwd_graph.sum(1, keepdim=True) + 1e-6)
        ind_graph = ind_graph / (ind_graph.sum(1, keepdim=True) + 1e-6)
        
        fwd_imp, fwd_pred, fwd_repr = self.fwd_module(x, mask, fwd_graph, ind_graph)
        rev_x, rev_mask = [reverse_tensor(tensor) for tensor in (x, mask)]
        bwd_res = self.bwd_module(rev_x, rev_mask, bwd_graph, ind_graph)
        bwd_imp, bwd_pred, bwd_repr = [reverse_tensor(res) for res in bwd_res]
        
        # ind_imp, ind_pred, ind_repr = self.ind_module(x, mask, ind_graph)
        
        if self._impute_from_states:
            inputs = [fwd_repr, bwd_repr, mask]
            if self.emb is not None:
                b, *_, s = fwd_repr.shape
                inputs += [self.emb.view(1, *self.emb.shape, 1).expand(b, -1, -1, s)]
            out = torch.cat(inputs, dim=1)
            out = self.out(out)
        else:
            out = torch.stack([fwd_imp, bwd_imp], dim=1) # fwd_out: [batches, input(1), nodes, steps]
            out = self.out(out, dim=1)
        
        pred = torch.stack([fwd_imp, bwd_imp, fwd_pred, bwd_pred], dim=0)
        
        out = torch.transpose(out, -3, -1)
        pred = torch.transpose(pred, -3, -1)
        
        if self.training:
            return out, pred
        else:
            return out


class FrameBlock(nn.Module):
    def __init__(self, n_nodes,
                 in_ch=1,
                 hidden_ch=64,
                 n_layers=1):
        super().__init__()
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.n_layers = n_layers
        
        self.init_pred = nn.Conv1d(in_channels=hidden_ch, out_channels=in_ch, kernel_size=1)
        self.out_pred = nn.Conv1d(in_channels=hidden_ch, out_channels=in_ch, kernel_size=1)
        self.cells = nn.ModuleList()
        for i in range(n_layers):
            self.cells.append(nn.GRUCell(input_size=in_ch if i==0 else hidden_ch, hidden_size=hidden_ch))
        self.h0 = self.init_state(n_nodes)
    
    def init_state(self, n_nodes):
        h = []
        for layer in range(self.n_layers):
            std = 1. / torch.sqrt(torch.tensor(self.hidden_ch))
            val = torch.distributions.Normal(0, std).sample((self.hidden_ch, n_nodes))
            h.append(nn.parameter.Parameter(val))
        return nn.ParameterList(h)
    
    def forward(self, x, mask, graph, ind_graph):
        bs, in_ch, n_nodes, steps = x.shape
        
        h = [h_.expand(bs, -1, -1) for h_ in self.h0.to(x.device)]
        
        imps, preds, reprs = [], [], []
        for step in range(steps):
            x_now = x[..., step]
            m_now = mask[..., step]
            h_now = h[-1]
            
            x_hat1 = self.init_pred(h_now)
            x_hat = torch.where(m_now, x_now, x_hat1)
            
            h_now = torch.einsum('ncv,wv->ncw', (h_now, graph))
            x_hat = torch.einsum('ncv,wv->ncw', (x_hat, ind_graph))
            
            h_new = []
            for node in range(x_hat.shape[2]):
                x_hat_n = x_hat[:, :, node]
                h_now_n = self.cells[0](x_hat_n, h_now[:, :, node])
                for layer in range(1, self.n_layers):
                    h_now_n = self.cells[layer](h_now_n, h_now_n)
                h_new.append(h_now_n)
            h_new = torch.stack(h_new, dim=2)
            
            x_hat2 = self.out_pred(h_new)
            x_hat = torch.where(m_now, x_hat, x_hat2)
            
            h.append(h_new)
            
            imps.append(x_hat2)
            preds.append(x_hat1)
            reprs.append(h_new)
        
        imps = torch.stack(imps, dim=-1)
        preds = torch.stack(preds, dim=-1)
        reprs = torch.stack(reprs, dim=-1)
        return imps, preds, reprs
        