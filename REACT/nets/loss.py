import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        if inputs.shape[1] == 1:
            inputs = torch.cat([1-inputs, inputs], dim=1)
        
        ce_loss = F.cross_entropy(inputs, targets, 
                                  reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()
    

class Multitask_CE(nn.Module):
    def __init__(self, base_loss="focalloss"):
        super().__init__()
        
        if base_loss == "ce" or base_loss == "crossentropy": 
            self.base_loss = nn.CrossEntropyLoss()
        elif base_loss == "focalloss":
            self.base_loss = FocalLoss()
        else:
            raise NotImplementedError
        
    def forward(self, y_list, label_list):
        if not isinstance(y_list, list):
            y_list = [y_list]
        if not isinstance(label_list, list):
            label_list = [label_list]
        
        val = 0
        for y, label in zip(y_list, label_list):
            val += self.base_loss(y, label)
        return val


class Gauss_Gate_Regularizer_Gauss(nn.Module):
    def __init__(self, sigma):
        self.sigma = sigma
        super().__init__()
        
    def forward(self, mu):
        graph_erf = 0.5 - 0.5 * torch.erf(-(mu + 0.5) / (np.sqrt(2) * self.sigma))
        return torch.mean(graph_erf)


class L1_Regularizer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, mu):
        return torch.norm(mu, p=1)
    
class Causal_Loss(nn.Module):
    def __init__(self, data_loss, lambda_s, norm="l1", norm_by_shape=True):
        super().__init__()
        
        self.data_loss = data_loss
        self.lambda_s = lambda_s
        self.norm_by_shape = norm_by_shape
        if norm == "l1":
            self.norm = lambda x: torch.norm(x, p=1)
        else:
            raise NotImplementedError
        
    def forward(self, y_list, label_list, graph_list, graph_weights):
        
        loss_sparsity = 0
        for graph, weight in zip(graph_list, graph_weights):
            norm = self.norm(graph) * weight
            if self.norm_by_shape:
                norm /= np.prod(graph.shape)
            loss_sparsity += norm
            
        loss_data = self.data_loss(y_list, label_list)
        loss = loss_sparsity * self.lambda_s + loss_data
        
        return loss, loss_sparsity, loss_data


class Irregular_Recon_Loss(nn.Module):
    def __init__(self, data_loss):
        super().__init__()
        self.data_loss = data_loss
        
    def forward(self, y, gt):
        if len(gt.shape) == 3: # b n d
            gt_arr = gt[:, :, 0]
            gt_abs = gt[:, :, 1]
            b, n, d = gt.shape
        elif len(gt.shape) == 4: # b n t d
            gt_arr = gt[:, :, :, 0]
            gt_abs = gt[:, :, :, 1]
            b, n, t, d = gt.shape
        
        loss = self.data_loss(y * (1 - gt_abs), gt_arr * (1 - gt_abs)) / torch.mean(1 - gt_abs)
        return loss
    