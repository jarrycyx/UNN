import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import math


class CGCN(nn.Module):
    def __init__(self, d_in,
                 window,
                 d_hidden=64):
        super(CGCN, self).__init__()
        self.Causal = cLSTM(d_in, d_hidden)
        self.Imputation = GAT_(d_in, window)

    def forward(self, x, mask=None, **kwargs):
        # 数据处理，对缺失值使用padding填充
        # XX_numpy = XX.cpu().detach().numpy()
        # M_nan = (1-M.cpu().detach().numpy()).astype(bool)
        # XX_numpy[M_nan] = np.nan
        # XX_numpy = [pd.DataFrame(XX_numpy[i].T).fillna(method = 'pad').to_numpy() for i in range(XX_numpy.shape[0])]
        # XX_numpy = np.array(XX_numpy)
        # XX_numpy = [pd.DataFrame(XX_numpy[i]).fillna(0.5).to_numpy() for i in range(XX_numpy.shape[0])]
        # XX_numpy = torch.Tensor(XX_numpy).to(XX.device)
        # XX_numpy = XX_numpy.permute(0,2,1)
        # XX_numpy = double_exponential_smoothing(XX_numpy, alpha = 0.3, beta = 0.2)[1:,:]      

        # XX = XX.permute(0,2,1)
        x = x.squeeze()
        pred = [self.Causal.networks[i](x)[0] for i in range(x.size(2))]
        GC_est = self.Causal.GC(threshold=False)

        W_est = GC_est
        D = W_est.sum(1)
        D_12 = torch.diag(D**(-0.5))
        Graph_L = D_12.mm(torch.diag(D)-W_est).mm(D_12)

        x = x.permute(0,2,1)
        re_x = self.Imputation(x, Graph_L)
        re_x = re_x.permute(0,2,1)
        re_x = re_x.unsqueeze(-1)
        # pred = torch.cat(pred, axis=2).unsqueeze(-1)
        
        if self.training:
            return re_x, pred
        else:
            return re_x


class LSTM(nn.Module):
    def __init__(self, num_series, hidden):
        '''
        LSTM model with output layer to generate predictions.
        Args:
          num_series: number of input time series.
          hidden: number of hidden units.
        '''
        super(LSTM, self).__init__()
        self.p = num_series
        self.hidden = hidden

        # Set up network.
        self.lstm = nn.LSTM(num_series, hidden, batch_first=True)
        self.lstm.flatten_parameters()
        self.linear = nn.Conv1d(hidden, 1, 1)

    def init_hidden(self, batch):
        '''Initialize hidden states for LSTM cell.'''
        device = self.lstm.weight_ih_l0.device
        return (torch.zeros(1, batch, self.hidden, device=device),
                torch.zeros(1, batch, self.hidden, device=device))

    def forward(self, X, hidden=None):
        # Set up hidden state.
        if hidden is None:
            hidden = self.init_hidden(X.shape[0])

        # Apply LSTM.
        X, hidden = self.lstm(X, hidden)

        # Calculate predictions using output layer.
        X = X.transpose(2, 1)
        X = self.linear(X)
        return X.transpose(2, 1), hidden


class cLSTM(nn.Module):
    def __init__(self, num_series, hidden):
        '''
        cLSTM model with one LSTM per time series.
        Args:
          num_series: dimensionality of multivariate time series.
          hidden: number of units in LSTM cell.
        '''
        super(cLSTM, self).__init__()
        self.p = num_series
        self.hidden = hidden

        # Set up networks.
        self.networks = nn.ModuleList([
            LSTM(num_series, hidden) for _ in range(num_series)])

    def forward(self, X, hidden=None):
        '''
        Perform forward pass.
        Args:
          X: torch tensor of shape (batch, T, p).
          hidden: hidden states for LSTM cell.
        '''
        if hidden is None:
            hidden = [None for _ in range(self.p)]
        pred = [self.networks[i](X, hidden[i])
                for i in range(self.p)]
        pred, hidden = zip(*pred)
        pred = torch.cat(pred, dim=2)
        return pred, hidden

    def GC(self, threshold=True):
        '''
        Extract learned Granger causality.
        Args:
          threshold: return norm of weights, or whether norm is nonzero.
        Returns:
          GC: (p x p) matrix. Entry (i, j) indicates whether variable j is
            Granger causal of variable i.
        '''
        GC = [torch.norm(net.lstm.weight_ih_l0, dim=0)
              for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (GC > 0).int()
        else:
            return GC


class AttentionLayer(nn.Module):
    """
    Attention Layer
    """

    def __init__(self, LSTM_dim, time_step):
        super(AttentionLayer, self).__init__()
        self.LSTM_dim = LSTM_dim
        self.time_step = time_step
        self.Linear_atten = nn.Linear(self.LSTM_dim, self.time_step)
        self.soft = nn.Softmax()
        self.tanh = nn.Tanh()

    def forward(self, input):
        # a = input.permute(0,2,1)   # 把时间维度放在后面  b n t 
        # print('a',a.size())
        a = self.Linear_atten(input)  # b t t 
        a = self.tanh(a)  # b t t 
        a = self.soft(a)  # shape: b t t
        # print('a:',a.size(),'x:',input.size())
        # print('a size:', a.size(), 'input size:', input.size())
        x =  torch.matmul(a,input)  # b t t  *  b t n 
        return x

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))  # 128 132
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        # print('d size', d.size())
        # print('W size',self.W.size())
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma
    
class LSTM_decay(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTM_decay, self).__init__()
        self.in_dim = input_size
        self.lstm_dim = output_size
        # self.rnn_cell = nn.LSTMCell(35 * 2, RNN_HID_SIZE)
        self.temp_decay_h = TemporalDecay(input_size = self.in_dim, output_size = self.lstm_dim, diag = False)
        # self.temp_decay_x = TemporalDecay(input_size = 35, output_size = 35, diag = True)
        self.hist_reg = nn.Linear(self.lstm_dim, self.in_dim)
        # self.feat_reg = FeatureRegression(35)
        # self.weight_combine = nn.Linear(35 * 2, 35)
        self.dropout = nn.Dropout(p = 0.25)
        self.SEQ_LEN = 48
        self.act = nn.Sigmoid()
        self.W_h = Parameter(torch.Tensor(self.lstm_dim, self.lstm_dim))
        nn.init.kaiming_normal(self.W_h) 
        self.U_h = Parameter(torch.Tensor(self.in_dim, self.lstm_dim))
        nn.init.kaiming_normal(self.U_h) 
        self.b_h = Parameter(torch.Tensor(self.lstm_dim))
        nn.init.uniform_(self.b_h) 

    def forward(self, X, M):

        D = torch.zeros_like(M)
        D[:,0,:] = 0
        for t_k in range(1,D.size(1)):
            D[:,t_k,:] = 1 + ( (~ (M[:,t_k-1,:]==1)).to(torch.float64)) * D[:,t_k-1,:]


        H = Variable(torch.zeros((X.size()[0], self.lstm_dim, self.SEQ_LEN))).to(X.device)
        h = Variable(torch.zeros((X.size()[0], self.lstm_dim))).to(X.device)


        for t in range(self.SEQ_LEN):
            # print(x.size(),.size(),d.size())

            x = X[:, t, :]
            m = M[:, t, :]
            d = D[:, t, :]
            # print('x', x)
            # print(x.size(),m.size(),d.size())
            gamma_h = self.temp_decay_h(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)     # \hat x_t = W_x * h_{t-1} +b_x 

            x_c =  m * x +  (1 - m) * x_h   

            h = torch.matmul(h, self.W_h) + torch.matmul(x_c * m, self.U_h) + self.b_h

            h = self.act(h)

            H[:,:,t] = h

        return H.permute(0,2,1)


class GAT_(nn.Module):
    def __init__(self, d_in, window):
        """Dense version of GAT."""
        super(GAT_, self).__init__()
        # self.dropout = dropout
        self.input_dim = d_in
        self.LSTM_dim = 512
        # self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)
        self.LSTM = nn.GRU(input_size=self.input_dim, hidden_size = self.LSTM_dim, batch_first = True) #, bidirectional  = True
        # self.Gcn = nn.Linear()
        self.reg_lstm = nn.Linear(self.LSTM_dim, self.input_dim)
        self.LSTM_DECAY = LSTM_decay(input_size = self.input_dim, output_size = self.LSTM_dim)  
        # self.BN0 = nn.BatchNorm1d(self.input_dim)
        # self.BN2 = nn.BatchNorm1d((self.input_dim)*48)  # 128*2+121
        # self.dacay_att = AttentionLayer(self.input_dim,48)

        self.lstm_att = AttentionLayer(self.LSTM_dim, window) 
        # self.reg1 = nn.Linear((self.input_dim)*48, 512)
        # self.reg2 = nn.Linear(512,128)
        # self.reg3 = nn.Linear(128,512)
        # self.reg4 = nn.Linear(512,self.input_dim*48)
        # self.tanh = nn.Tanh()
        # self.drop = nn.Dropout(0.2)


    def forward(self, XX, G, eval_flag = 0):
        # print(XX.size(), M.size())
        # print('XX:',XX[0])
        # print('M:',M[0])
        # XX = (XX*M + (XX.sum(2)/((M).sum(2)+0.1)).unsqueeze(2).repeat(1,1,48)*(1-M))
        # print('X_in',X_in[0])
        # x = self.BN0(x)
        # out = torch.cat([att(X_in) for att in self.attentions], dim=2)
        # x = F.dropout(x, self.dropout, training=self.training)
        # out = self.out_att(out)
        # out = self.tanh(out)
        # out = out.permute(0,2,1)
        # M = M.permute(0,2,1)

        out = XX + G.matmul(XX)
        out = out.permute(0,2,1)
        # LSTM decay + Attention 
        # h_decay = self.LSTM_DECAY(out, M)   # batchsize 48, 121
        # h_decay = self.dacay_att(h_decay)         # decay和attention不能一起用

        h_lstm,_ = self.LSTM(out)
        out = self.lstm_att(h_lstm)
        out = 0.1*out + h_lstm
        # out = torch.cat((h_decay,h_lstm), dim = 2)
        # out = h_decay
        out_1 = self.reg_lstm(out).permute(0,2,1)

        # x = x.view(T, N, -1)  # (7580, 228, 3)
        return out_1

