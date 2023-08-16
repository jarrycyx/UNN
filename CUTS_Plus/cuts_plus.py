import logging
import os, sys
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops

sys.path.append(opj(opd(__file__), ".."))

import tqdm
import numpy as np
from matplotlib import pyplot as plt
import argparse
from omegaconf import OmegaConf
from copy import deepcopy
import torch
from torch import dropout, nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, roc_auc_score

from utils.gumbel_softmax import gumbel_softmax
from utils.misc import calc_and_log_metrics, log_time_series, plot_causal_matrix
from utils.opt_type import MultiCADopt
from utils.logger import MyLogger

from datetime import datetime
from model.cuts_plus_net import CUTS_Plus_Net

import os
from einops import rearrange

def plot_matrix(name, mat, log, log_step, vmin=None, vmax=None):
    if len(mat.shape) == 3:
        mat = np.max(mat, axis=-1)
    n, m = mat.shape

    # Show Discovered Graph (Probability)
    sub_cg = plot_causal_matrix(
        mat,
        figsize=[1.5*n, 1*n],
        show_text=False,
        vmin=vmin, vmax=vmax)
    log.log_figures(sub_cg, name=name, iters=log_step)


def generate_indices(input_step, pred_step, t_length, block_size=None):
    if block_size is None:
        block_size = t_length
        
    offsets_in_block = np.arange(input_step, block_size-pred_step+1)
    assert t_length % block_size == 0, "t_length % block_size != 0"
    random_t_list = []
    for block_start in range(0, t_length, block_size):
        random_t_list += (offsets_in_block + block_start).tolist()
    
    np.random.shuffle(random_t_list)
    return random_t_list



def batch_generater(data, observ_mask, bs, n_nodes, input_step, pred_step, block_size=None):
    t, n, d = data.shape
    first_sample_t = input_step
    random_t_list = generate_indices(input_step, pred_step, t_length=t, block_size=block_size)

    for batch_i in range(len(random_t_list) // bs):
        x = torch.zeros([bs, n_nodes, input_step, d]).to(data.device)
        y = torch.zeros([bs, n_nodes, pred_step, d]).to(data.device)
        t = torch.zeros([bs]).to(data.device).long()
        mask_x = torch.zeros([bs, n_nodes, input_step, d]).to(data.device)
        mask_y = torch.zeros([bs, n_nodes, pred_step, d]).to(data.device)
        for data_i in range(bs):
            data_t = random_t_list.pop()
            x[data_i, :, :, :] = rearrange(data[data_t-input_step : data_t, :], "t n d -> n t d")
            y[data_i, :, :, :] = rearrange(data[data_t : data_t+pred_step, :], "t n d -> n t d")
            t[data_i] = data_t
            mask_x[data_i, :, :, :] = rearrange(observ_mask[data_t-input_step : data_t, :], "t n d -> n t d")
            mask_y[data_i, :, :, :] = rearrange(observ_mask[data_t:data_t+pred_step, :], "t n d -> n t d")

        yield x, y, t, mask_x, mask_y
        
        



class MultiCAD(object):
    def __init__(self, args: MultiCADopt.MultiCADargs, log, device="cuda"):
        self.log: MyLogger = log
        self.args = args
        self.device = device

        # self.fitting_model = CUTS_Plus_LSTM(self.args.data_dim,
        #                                self.args.data_pred.mlp_hid,
        #                                self.args.data_dim * self.args.data_pred.pred_step,
        #                                self.args.data_pred.mlp_layers,
        #                                self.args.n_nodes).to(self.device)
        self.fitting_model = CUTS_Plus_Net(self.args.n_nodes, in_ch=self.args.data_dim,
                                           n_layers=self.args.data_pred.gru_layers,
                                           hidden_ch=self.args.data_pred.mlp_hid,
                                           shared_weights_decoder=self.args.data_pred.shared_weights_decoder,
                                           concat_h=self.args.data_pred.concat_h,
                                           ).to(self.device)

        self.data_pred_loss = nn.MSELoss()
        self.data_pred_optimizer = torch.optim.Adam(self.fitting_model.parameters(),
                                                    lr=self.args.data_pred.lr_data_start,
                                                    weight_decay=self.args.data_pred.weight_decay)
        
        
        if "every" in self.args.fill_policy:
            lr_schedule_length = int(self.args.fill_policy.split("_")[-1])
        else:
            lr_schedule_length = self.args.total_epoch
            
        gamma = (self.args.data_pred.lr_data_end / self.args.data_pred.lr_data_start) ** (1 / lr_schedule_length)
        self.data_pred_scheduler = torch.optim.lr_scheduler.StepLR(
            self.data_pred_optimizer, step_size=1, gamma=gamma)
        
        self.n_groups = self.args.n_groups
        print("n_groups: ", self.n_groups)
        if self.args.group_policy == "None":
            self.args.group_policy = None

        end_tau, start_tau = self.args.graph_discov.end_tau, self.args.graph_discov.start_tau
        self.gumbel_tau_gamma = (end_tau / start_tau) ** (1 / self.args.total_epoch)
        self.gumbel_tau = start_tau
        self.start_tau = start_tau
        
        end_lmd, start_lmd = self.args.graph_discov.lambda_s_end, self.args.graph_discov.lambda_s_start
        self.lambda_gamma = (end_lmd / start_lmd) ** (1 / self.args.total_epoch)
        self.lambda_s = start_lmd
    
    def set_graph_optimizer(self, epoch=None):
        if epoch == None:
            epoch = 0
        
        gamma = (self.args.graph_discov.lr_graph_end / self.args.graph_discov.lr_graph_start) ** (1 / self.args.total_epoch)
        self.graph_optimizer = torch.optim.Adam([self.GT], lr=self.args.graph_discov.lr_graph_start * gamma ** epoch)
        self.graph_scheduler = torch.optim.lr_scheduler.StepLR(self.graph_optimizer, step_size=1, gamma=gamma)
        

    def latent_data_pred(self, x, y, mask_x, mask_y):
        
        def sample_bernoulli(sample_matrix, batch_size):
            sample_matrix = sample_matrix[None].expand(batch_size, -1, -1)
            return torch.bernoulli(sample_matrix).float()
        
        def sample_multinorm(sample_matrix, batch_size):
            sampled = torch.multinomial(sample_matrix, batch_size, replacement=True).T
            return F.one_hot(sampled).float()
            
        
        bs, n, t, d = x.shape
        self.fitting_model.train()
        self.data_pred_optimizer.zero_grad()
        
        GT_prob = self.GT
        G_prob = self.G
        
        Graph = torch.einsum("nm,ml->nl", G_prob, torch.sigmoid(GT_prob))
        graph_sampled = sample_bernoulli(Graph, self.args.batch_size)
            
        y_pred = self.fitting_model(x, mask_x, graph_sampled)

        # print(y_pred.shape, y.shape, observ_mask.shape)
        loss = self.data_pred_loss(y * mask_y, y_pred * mask_y) / torch.mean(mask_y)
        loss.backward()
        self.data_pred_optimizer.step()
        return y_pred, loss

    def graph_discov(self, x, y, mask_x, mask_y):

        def gumbel_sigmoid_sample(graph, batch_size, tau=1):
            prob = graph[None, :, :, None].expand(batch_size, -1, -1, -1)
            logits = torch.concat([prob, (1-prob)], axis=-1)
            samples = gumbel_softmax(logits, tau=tau, hard=True)[:, :, :, 0]
            return samples
        
        gn, n = self.GT.shape
        # self.fitting_model.eval()
        self.graph_optimizer.zero_grad()
        GT_prob = self.GT
        G_prob = self.G
        
        Graph = torch.einsum("nm,ml->nl", G_prob, torch.sigmoid(GT_prob))
        graph_sampled = gumbel_sigmoid_sample(Graph, self.args.batch_size) 
        
        loss_sparsity = torch.linalg.norm(Graph.flatten(), ord=1) / (n * n)

        y_pred = self.fitting_model(x, mask_x, graph_sampled)
        
        loss_data = self.data_pred_loss(y * mask_y, y_pred * mask_y) / torch.mean(mask_y)
        loss = loss_sparsity * self.lambda_s + loss_data
        loss.backward()
        self.graph_optimizer.step()
        
        return loss, loss_sparsity, loss_data



    def train(self, data, observ_mask, original_data, true_cm=None):

        original_data = torch.from_numpy(original_data).float().to(self.device)
        observ_mask = torch.from_numpy(observ_mask).float().to(self.device)
        data = torch.from_numpy(data).float().to(self.device)
        
        if self.args.supervision_policy == "masked":
            print("Using masked supervision for data prediction...")
        elif self.args.supervision_policy == "full":
            print("Using full supervision for data prediction......")
            observ_mask = torch.ones_like(observ_mask)
        elif "masked_before" in self.args.supervision_policy:
            print(f"Using masked supervision for data prediction ({self.args.supervision_policy:s})......")

        latent_pred_step = 0
        graph_discov_step = 0
        pbar = tqdm.tqdm(total=self.args.total_epoch)
        data_interp = deepcopy(data)
        original_mask = deepcopy(observ_mask)
        auc = 0
        for epoch_i in range(self.args.total_epoch):
            
            if self.args.group_policy is not None:
                group_mul = int(self.args.group_policy.split("_")[1])
                group_every = int(self.args.group_policy.split("_")[3])
                if epoch_i % group_every == 0 and self.n_groups < self.args.n_nodes:
                    if epoch_i != 0:
                        self.n_groups *= group_mul
                    if self.n_groups > self.args.n_nodes:
                        self.n_groups = self.args.n_nodes
                    
                    self.G = torch.zeros([self.args.n_nodes, self.n_groups]).to(self.device)

                    for i in range(0, self.n_groups):
                        for j in range(0, self.args.n_nodes // self.n_groups):
                            self.G[i*(self.args.n_nodes // self.n_groups) + j, i] = 1
                    for k in range(i*(self.args.n_nodes // self.n_groups) + j, self.args.n_nodes):
                        self.G[k, i] = 1

                    # inv_A = torch.linalg.inv(torch.mm(torch.t(self.fwd_graphA), self.fwd_graphA))
                    # fwd_graphB_init = torch.mm(inv_A, torch.mm(torch.t(self.fwd_graphA), self.fwd_graph))

                    if hasattr(self, "GT"):
                        GT_init = torch.sigmoid(self.GT).detach().cpu().repeat_interleave(group_mul, 0)[:self.n_groups, :]
                        GT_init = 1 - (1 - GT_init)**(1 / group_mul)
                    else:
                        GT_init = torch.ones((self.n_groups, self.args.n_nodes))*0.5

                    self.GT = nn.Parameter(GT_init.to(self.device))
                    
                    self.set_graph_optimizer(epoch_i)
                elif epoch_i == 0 and self.n_groups == self.args.n_nodes:
                    self.G = torch.eye(self.args.n_nodes).to(self.device)
                    GT_init = torch.ones((self.n_groups, self.args.n_nodes))*0.5
                    self.GT = nn.Parameter(GT_init.to(self.device))
                    self.set_graph_optimizer(epoch_i)
                    
            
            if "every" in self.args.fill_policy:
                update_every = int(self.args.fill_policy.split("_")[-1])
                if (epoch_i+1) % update_every == 0:
                    data = data_pred
                    print("Update data!")
                    # self.graph_optimizer.param_groups[0]['lr'] = self.args.graph_discov.lr_graph_start
                    self.data_pred_optimizer.param_groups[0]['lr'] = self.args.data_pred.lr_data_start
                    observ_mask = torch.ones_like(original_mask)
            elif "rate" in self.args.fill_policy:
                update_rate = float(self.args.fill_policy.split("_")[1])
                update_after = int(self.args.fill_policy.split("_")[3])
                if epoch_i+1 > update_after:
                    if epoch_i == update_after:
                        print("Data update started!")
                    data = data * (1 - update_rate) + data_pred * update_rate
            else:
                # no data update
                pass
            
            if "masked_before" in self.args.supervision_policy:
                masked_before = int(self.args.supervision_policy.split("_")[2])
                if epoch_i == masked_before:
                    print("Using full supervision for data prediction......")
                    observ_mask = torch.ones_like(original_mask)
                    self.gumbel_tau = self.start_tau
            
            # Data Prediction
            if hasattr(self.args, "data_pred"):
                if hasattr(self.args, "block_size"):
                    block_size = self.args.block_size
                else:
                    block_size = None
                ## 
                batch_gen = batch_generater(data, observ_mask, # !!!!! TO-DO
                                            bs=self.args.batch_size, 
                                            n_nodes=self.args.n_nodes, 
                                            input_step=self.args.input_step, 
                                            pred_step=self.args.data_pred.pred_step,
                                            block_size=block_size)
                batch_gen = list(batch_gen)
                
                data_pred = deepcopy(data) # masked data points are predicted
                data_pred_all = deepcopy(data)
                for x, y, t, mask_x, mask_y in batch_gen:
                    latent_pred_step += self.args.batch_size
                    y_pred, loss = self.latent_data_pred(x, y, mask_x, mask_y)
                    data_pred[t] = (y_pred*(1-mask_y) + y*mask_y).clone().detach()[:,:,0]
                    data_pred_all[t] = y_pred.clone().detach()[:,:,0]
                    self.log.log_metrics({"latent_data_pred/pred_loss": loss.item()}, latent_pred_step)
                    pbar.set_postfix_str(f"S1 loss={loss.item():.2f}, spr=IDLE, auc={auc:.4f}")

                current_data_pred_lr = self.data_pred_optimizer.param_groups[0]['lr']
                self.log.log_metrics({"graph_discov/lr": current_data_pred_lr}, latent_pred_step)
                self.data_pred_scheduler.step()
                mse_pred_to_original = self.data_pred_loss(original_data, data_pred)
                mse_interp_to_original = self.data_pred_loss(original_data, data_interp)
                
                self.log.log_metrics({"latent_data_pred/mse_pred_to_original": mse_pred_to_original,
                                      "latent_data_pred/mse_interp_to_original": mse_interp_to_original}, latent_pred_step)
                
            # Graph Discovery
            if hasattr(self.args, "graph_discov"):
                for x, y, t, mask_x, mask_y in batch_gen:
                    graph_discov_step += self.args.batch_size
                    if hasattr(self.args, "disable_graph") and self.args.disable_graph:
                        pass
                    else:
                        loss, loss_sparsity, loss_data = self.graph_discov(x, y, mask_x, mask_y)
                        self.log.log_metrics({"graph_discov/sparsity_loss": loss_sparsity.item(),
                                            "graph_discov/data_loss": loss_data.item(),
                                            "graph_discov/total_loss": loss.item()}, graph_discov_step)
                        pbar.set_postfix_str(f"S2 loss={loss_data.item():.2f}, spr={loss_sparsity.item():.2f}, auc={auc:.4f}")
                    
                self.graph_scheduler.step()
                # self.group_scheduler.step()
                current_graph_disconv_lr = self.graph_optimizer.param_groups[0]['lr']
                self.log.log_metrics({"graph_discov/lr": current_graph_disconv_lr}, graph_discov_step)
                self.log.log_metrics({"graph_discov/tau": self.gumbel_tau}, graph_discov_step)
                self.gumbel_tau *= self.gumbel_tau_gamma
                self.lambda_s *= self.lambda_gamma

            pbar.update(1)
            
            plot_roc = False
            
            G_prob = self.G.detach().cpu().numpy()
            GT_prob = self.GT.detach().cpu().numpy()
            Graph = np.einsum("nm,ml->nl", G_prob, GT_prob) 
            
            
            if (epoch_i+1) % self.args.show_graph_every == 0:
                avg_mask = np.mean(observ_mask.cpu().numpy(), axis=(0,2))
                if np.min(avg_mask) < 1:
                    time_series_idx = int(np.argwhere(avg_mask < 1)[0])
                else:
                    time_series_idx = 0
                log_time_series(original_data.cpu()[-100:,time_series_idx], 
                                data_interp.cpu()[-100:,time_series_idx], 
                                data_pred_all.cpu()[-100:,time_series_idx], log=self.log, log_step=latent_pred_step)
                # plot_causal_matrix_in_training(G_A0_GT, self.log, graph_discov_step, threshold=threshold)
                plot_matrix("G", G_prob, self.log, graph_discov_step, vmin=0, vmax=1)
                plot_matrix("GT", GT_prob, self.log, graph_discov_step, vmin=0, vmax=1)
                plot_matrix("Graph", Graph, self.log, graph_discov_step, vmin=0, vmax=1)
                np.save(os.path.join(self.log.log_dir, 'Graph.npy'), Graph)
                plot_roc = True
            
            # Show TPR FPR AUC ROC
            if true_cm is not None:
                # Graph = rearrange(Graph, "n m -> m n")
                auc = calc_and_log_metrics(Graph, true_cm, 
                                           self.log, graph_discov_step, plot_roc=plot_roc)
        
        return Graph
                
           
def prepross_data(data):
    T, N, D = data.shape
    new_data = np.zeros_like(data, dtype=float)
    for i in range(N):
        node = data[:,i,:]
        new_data[:,i,:] = (node - np.mean(node)) / np.std(node)
        
    return new_data
         


def main(data, mask, true_cm, opt, log, device="cuda"):
    if opt.n_nodes == "auto":
        opt.n_nodes = data.shape[1]
        
    data = data[:,:,None]
    mask = mask[:,:,None]
    data = prepross_data(data)
    
    multicad = MultiCAD(opt, log, device=device)
    Graph = multicad.train(data, mask, data, true_cm)
    return Graph

                

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    parser = argparse.ArgumentParser(description='Batch Compress')
    parser.add_argument('-opt', type=str, default=opj(opd(__file__),
                        'opt/multi_cad_lorenz.yaml'), help='yaml file path')
    parser.add_argument('-g', help='availabel gpu list', default='2', type=str)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-log', action='store_true')
    args = parser.parse_args()

    if args.g == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = "mps"
    elif args.g == "cpu":
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.g
        device = "cuda"
        
    main(OmegaConf.load(args.opt), device=device)
