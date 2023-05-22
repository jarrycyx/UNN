import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import math
import numpy as np
import matplotlib.pyplot as plt

class SCGL(nn.Module):
    def __init__(self,
                 n_nodes,
                 window,
                 pre_window,
                 k_list=[50],
                 p_list=[50]*6,
                 compress_p_list=[80],
                 dropout=0.1):
        super(SCGL, self).__init__()
        self.m = n_nodes
        self.w = window
        self.pre_win = pre_window
        
        self.k_list = k_list
        self.add_k_list = [self.m]+self.k_list
        self.len_k_list = len(self.k_list)
        self.p_list = p_list
        self.p_allsum = np.sum(self.p_list)
        self.len_p_list = len(self.p_list)
        self.compress_p_list = compress_p_list
        self.len_compress_p_list = len(self.compress_p_list)

        self.sparse_label = []
        self.orthgonal_label = []
        ## p_list extention
        self.P = [] #w->hid
        for p_i in np.arange(0,self.len_p_list):
            self.P.append(WL_Model())
        self.P = nn.ModuleList(self.P)
        
        self.linears = [(nn.Linear(self.w, self.p_list[0]))] #w->hid
        self.sparse_label.append(0)
        self.orthgonal_label.append(1)
        if self.len_p_list>1:
            for p_i in np.arange(1,self.len_p_list):
                self.linears.append((nn.Linear(self.p_list[p_i-1], self.p_list[p_i]))) #w->hid
                self.sparse_label.append(0)
                self.orthgonal_label.append(1)

        
        ## graph layers
        for p_i in np.arange(0,self.len_p_list):
            for k_i in np.arange(0,self.len_k_list):
                self.linears.append(nn.utils.weight_norm(nn.Linear(self.add_k_list[k_i], self.add_k_list[k_i+1], bias = False))) #m->k
                self.sparse_label.append(1) 
                if k_i ==0:
                    self.orthgonal_label.append(1)
                else:
                    self.orthgonal_label.append(2)
                self.linears.append(nn.BatchNorm1d(self.p_list[-1])) #m->k
                self.sparse_label.append(0)
                self.orthgonal_label.append(0)

            
            self.linears.append(SV_Model(self.k_list[-1])) #k->k
            self.sparse_label.append(0)
            self.orthgonal_label.append(0)
            
            for k_i in np.arange(self.len_k_list,0,-1):       
                self.linears.append(nn.utils.weight_norm(nn.Linear(self.add_k_list[k_i], self.add_k_list[k_i-1], bias = False))) #m->m, supervised
                self.sparse_label.append(1) 
                if k_i == 1:
                    self.orthgonal_label.append(1)
                else:
                    self.orthgonal_label.append(2)
                self.linears.append(nn.BatchNorm1d(self.p_list[-1])) #m->k
                self.sparse_label.append(0)
                self.orthgonal_label.append(0)
        
                
        if self.len_compress_p_list>0:
            self.linears.append( (nn.Linear(self.p_allsum, self.compress_p_list[0])))
            self.sparse_label.append(0)
            self.orthgonal_label.append(1)
            for p_j in np.arange(1,self.len_compress_p_list):
                self.linears.append( (nn.Linear(self.compress_p_list[p_j-1], self.compress_p_list[p_j])))
                self.sparse_label.append(0)
                self.orthgonal_label.append(1)
          
#        
        self.linears.append(FR_Model(n_nodes=n_nodes, window=window, pre_window=pre_window, p_list=p_list, compress_p_list=compress_p_list)) #k->k  
        self.sparse_label.append(1)
        self.orthgonal_label.append(0)
        
        
        self.linears = nn.ModuleList(self.linears)
        self.dropout = nn.Dropout(dropout)

        for layer_i in range(len(self.linears)):
            if not isinstance(self.linears[layer_i], nn.InstanceNorm1d) and not isinstance(self.linears[layer_i], nn.BatchNorm1d) and not isinstance(self.linears[layer_i], SV_Model):
                W = self.linears[layer_i].weight.transpose(0,1).detach().numpy()
                ## sparsity
                if W.ndim >=2 and self.orthgonal_label[layer_i]==1: ## sparsity
                    #nn.init.xavier_normal_(self.linears[layer_i].weight)
                    self.linears[layer_i].weight = nn.init.orthogonal_(self.linears[layer_i].weight)
                if W.ndim >=2 and self.orthgonal_label[layer_i]>1: ## sparsity
                    #nn.init.xavier_normal_(self.linears[layer_i].weight)
                    tmp = self.linears[layer_i].weight
                    self.linears[layer_i].weight = np.eye(tmp.shape[0],tmp.shape[1])
                    
    
    def forward(self, x, **kwargs):
        x = x.squeeze()
        x = x.transpose(2,1).contiguous() #mxp
        x = self.dropout(x)
        x_org = x
        x_p = []

        if self.p_list[0] > self.w:
            padding = nn.ConstantPad2d((0, self.p_list[0]-self.w, 0, 0), 0)
            x_0n = padding(x_org)
        
        x_0 = x_org
        for layer_i in range(self.len_p_list):  
            #pl = 1-0.5*layer_i/self.len_p_list
            x_i = self.linears[layer_i](x_0)
            #x_i = F.relu(x_i)
            x_i = F.relu(self.P[layer_i](x_i) + x_0n)
            x_0n = x_i
            x_0 = x_i
            x_p.append(x_i)
        
        x_p_m = []  
        for layer_i in range(self.len_p_list):
            
            x_sp =  x_p[layer_i].transpose(2,1).contiguous() ## read the data piece  
            
            x_sp_tmp = []
            x_sp_tmp.append(x_sp)
            for k_i in np.arange(0,self.len_k_list):   
                x_sp = self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1)+2*k_i](x_sp)  #lxk 
                x_sp = self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1)+2*k_i+1](x_sp)  #lxk 
                x_sp = F.tanh(x_sp/5.)
                x_sp = self.dropout(x_sp)
                x_sp_tmp.append(x_sp)
                #x_sp = self.dropout(x_sp)
            
            #pdb.set_trace()
            x_sp = self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1) + 2*self.len_k_list](x_sp)  #lxk 
            #x_sp = F.layer_norm(x_sp, [x_sp.shape[-2],x_sp.shape[-1]])
            
            for k_i in np.arange(0,self.len_k_list):  
                #pdb.set_trace()
                #if k_i>0:
                #    x_sp = x_sp + x_sp_tmp[-1*(k_i+1)]
                x_sp = self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1)+2*self.len_k_list + 1+2*k_i](x_sp)  #lxm
                x_sp = self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1)+2*self.len_k_list + 1+2*k_i+1](x_sp)  #lxm
                #x_sp = x_sp + x_sp_tmp[-1*(k_i+2)]
                x_sp = F.relu(x_sp/1.)
                x_sp = self.dropout(x_sp)
            
            x_sp = x_sp.transpose(2,1).contiguous() #mxl
            x_p_m.append(x_sp)
            
        x_p_m = torch.cat(x_p_m, dim = 2) 
        #x_p_m = self.linears[self.len_p_list+self.len_p_list*(2*self.len_k_list+1)](x_p_m)
        
            
        if self.len_compress_p_list>0:
            for p_j in range(self.len_compress_p_list): 
                x_p_m = self.linears[self.len_p_list+self.len_p_list*(4*self.len_k_list+1)+p_j](x_p_m) #mx2
                x_p_m = F.tanh(x_p_m/5.)
                x_sp = self.dropout(x_sp)
         
        #pdb.set_trace()
        
        final_y = self.linears[-1](x_p_m)
        final_y = final_y.unsqueeze(-1)

        if self.training:
            return final_y, [final_y]
        else:
            return final_y


    def predict_relationship(self):
        
        
        CGraph_list1 = []
        CGraph_list2 = []

        G_1 = np.zeros((self.m,self.m))
        G_2 = np.zeros((self.m,self.m))
        G_3 = np.zeros((self.m,self.m))
        G_4 = np.zeros((self.m,self.m))
        
                
        for layer_i in range(self.len_p_list):
            pl = self.P[layer_i].weight.data
            
            A = self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1)+0].weight.transpose(0,1).cpu().detach().numpy()
            B = np.diag(self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1)+2*self.len_k_list].weight.transpose(0,1).detach().cpu().numpy().ravel())
            C = self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1)+4*self.len_k_list+1-2].weight.transpose(0,1).cpu().detach().numpy()
            #CGraph1 = np.abs(np.dot(A,C))#
            CGraph1 = np.abs(np.dot(np.dot(A,B),C))
            CGraph1[range(self.m), range(self.m)] = 0    
            CGraph_list1.append(CGraph1)
            
            A = np.abs(A) 
            B = np.abs(B) 
            C = np.abs(C) 
            #CGraph2 = np.abs(np.dot(A,C))#
            CGraph2 = np.abs(np.dot(np.dot(A,B),C))
            CGraph2[range(self.m), range(self.m)] = 0    
            CGraph_list2.append(CGraph2)   
   
                
            G_1 = np.add(G_1, CGraph1)
            G_2 = np.add(G_2, CGraph2)
            
            G_3 = np.add(G_3, np.multiply(CGraph1, pl.cpu().detach().numpy())) 
            G_4 = np.add(G_4, np.multiply(CGraph2, pl.cpu().detach().numpy()))    
      
        G_1[range(self.m), range(self.m)] = 0 
        G_2[range(self.m), range(self.m)] = 0 
        G_3[range(self.m), range(self.m)] = 0 
        G_4[range(self.m), range(self.m)] = 0 
           
        return G_3, G_4,  G_1, G_2


class FR_Model(nn.Module):
    def __init__(self,
                 n_nodes,
                 window,
                 pre_window,
                 p_list=[20]*6,
                 compress_p_list=[50]):
        super(FR_Model, self).__init__()
        self.m = n_nodes
        self.pre_win = pre_window
        
        self.p_list = (p_list) 
        self.len_p_list = len(p_list) 
        self.compress_p_list = compress_p_list
        self.p_allsum = np.sum(self.p_list)
        self.len_compress_p_list = len(self.compress_p_list)
        if self.len_compress_p_list>0:
            
            self.compress_p = compress_p_list[-1]
            self.weight = nn.Parameter(torch.ones([self.m, self.compress_p, self.pre_win]))
        else:
            self.weight = nn.Parameter(torch.ones([self.m, self.p_allsum, self.pre_win]))
        
        #nn.init.orthogonal_(self.weight)
        #nn.init.sparse_(self.weight, sparsity=0.3)
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.orthogonal_(self.weight)
        self.bias = Parameter(torch.Tensor(self.m,self.pre_win)) 
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        #self.weight = nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        
        
        
    def forward(self, x):
        #l = x.shape[1]
        #k = x.shape[2]
        
        if self.pre_win ==1:
            final_y = torch.empty(x.shape[0], self.m) 
        else :
            final_y = torch.empty(x.shape[0], self.pre_win, self.m) 
        
        #x_latest = x_latest.view(x_latest.shape[0], x_latest.shape[1], 1)
        #x = torch.cat((x, x_latest), 2)
        for j in range(self.m):           
            if self.pre_win ==1:   
                #pdb.set_trace()
                final_y[:,j] = F.linear(x[:,j,:], self.weight[j,:].view(1, self.weight.shape[1]), self.bias[j,:]).view(-1)               
            else:
                #pdb.set_trace()
                final_y[:,:,j] = F.linear(x[:,j,:], self.weight[j,:].transpose(1,0), self.bias[j,:])               
        
        final_y = final_y.to(x.device)
        
        return final_y
    
    def get_pi_weight(self):
        if self.len_compress_p_list>0:
            func_1 = nn.MaxPool1d(kernel_size=self.compress_p, stride=self.compress_p)
        else:
            func_1 = nn.MaxPool1d(kernel_size=self.p_list[0], stride=self.p_list[0])
        func_2 = nn.MaxPool1d(kernel_size=self.m, stride=self.m)
        
        weight1_norm_all = np.zeros((self.weight.shape[0], self.len_p_list))
        weight2_norm_all = np.zeros((self.len_p_list))
        for layer_i in range(self.weight.shape[-1]):
            weight_tmp = self.weight[:,:,layer_i]
            weight0 = weight_tmp.view(1, self.weight.shape[0],self.weight.shape[1])
            weight1 = func_1(torch.abs(weight0)) 
            weight1_inv = weight1.transpose(2,1).contiguous() #mxp
            weight2 = func_2(weight1_inv).detach().numpy().ravel() 
            weight2_norm = weight2/np.sum(weight2)
            weight1_norm = F.normalize(weight1, p=1, dim=1).view(weight1.shape[1], weight1.shape[2]).detach().numpy()
            
            weight1_norm_all = weight1_norm_all + weight1_norm
            #pdb.set_trace()
            weight2_norm_all = weight2_norm_all + weight2_norm
        
        #pdb.set_trace()
        return weight1_norm_all, weight2_norm_all


class SV_Model(nn.Module):
    def __init__(self, lowrank):
        super(SV_Model, self).__init__()
        self.weight = nn.Parameter(torch.ones([lowrank,1]))
        
        
    def forward(self, x):
        k = x.shape[2]
        
        y = torch.Tensor(x.shape).to(x.device)
        
        for j in range(k):
            tmp_new = torch.mul(x[:,:,j], self.weight[j,0])
            y[:,:,j] = tmp_new
                
        return y


class WL_Model(nn.Module):
    def __init__(self):
        super(WL_Model, self).__init__()
        self.weight = nn.Parameter(torch.ones([1]))
        
        
    def forward(self, x):
        y = torch.mul(x, self.weight)
        return y

