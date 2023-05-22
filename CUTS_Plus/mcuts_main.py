import os
from os.path import join as opj
from os.path import dirname as opd

import numpy as np
import argparse
import tqdm
import time
from datetime import datetime
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, RandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.gumbel_softmax import gumbel_softmax
from utils.misc import reproduc, plot_causal_matrix_in_training, calc_and_log_metrics
from utils.opt_type import MultiCADopt
from utils.logger import MyLogger
from utils.earlystopping import EarlyStopping

from data.simu_data import simulate_var, simulate_random_var, simulate_var_from_links, load_netsim_data, load_springs_data, load_dream_data, simulate_lorenz_96
from data.air_quality import AirQuality
from data.pd_dataset import PandasDataset
from data.torch_dataset import TorchDataset

from model.brits import BRITSNet
from model.grin import GRINet
from model.cgcn import CGCN
from model.timesnet import TimesNet
from model.mcuts import MCutsNet

from metric.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from metric.numpy_metrics import masked_mae, masked_mape, masked_mse, masked_mre

class MCUTS(object):
    def __init__(self, args: MultiCADopt.MultiCADargs, log, device="cuda"):
        self.log: MyLogger = log
        self.args = args
        self.device = device
        
        # graph_discov setting
        self.graph_mask = torch.ones([self.args.n_nodes, self.args.n_nodes]).to(self.device)
        for i in range(self.args.n_nodes):
            self.graph_mask[i, i] = 0
        
        calc, val = self.args.causal_thres.split("_")
        if calc == "value":
            self.threshold = float(val)
        else:
            self.threshold = None
        
        self.disable_bwd = False
        self.update_bwd = True
        self.disable_ind = False
        
        if self.args.graph_discov.disable_bwd:
            self.disable_bwd = True
            self.update_bwd = False
        elif not self.args.graph_discov.separate_bwd:
            self.update_bwd = False
        
        if self.args.graph_discov.disable_ind:
            self.disable_ind = True
        
        self.fwd_graph = nn.Parameter(torch.ones([self.args.n_nodes, self.args.n_nodes]).float().to(self.device))
        if self.update_bwd:
            self.bwd_graph = nn.Parameter(torch.ones([self.args.n_nodes, self.args.n_nodes]).float().to(self.device))
        
        if self.args.total_epoch == 0 or self.args.total_epoch == 100:
            self.total_epoch = 100
            self.callback = EarlyStopping(patience=self.args.patience, verbose=True, path=opj(self.log.log_dir, 'data_model.pt'))
        else:
            self.total_epoch = self.args.total_epoch
            self.callback = None
        self.max_epoch = 100
        
        if self.args.group_policy == "None":
            self.args.group_policy = None
        
        if self.args.group_policy is not None:
            self.n_groups = self.args.n_groups
        else:
            self.set_graph_optimizer()
        
        self.gumbel_tau_gamma = self.args.graph_discov.tau_ratio ** (1 / self.max_epoch)
        self.gumbel_tau = self.args.graph_discov.tau_start
        
        self.lambda_gamma = self.args.graph_discov.lambda_s_ratio ** (1 / self.max_epoch)
        self.lambda_s = self.args.graph_discov.lambda_s_start
        
        
        # data_pred setting
        self.model_str = self.args.data_pred.model
        if self.model_str == 'brits':
            self.fitting_model = BRITSNet(d_in=args.n_nodes).to(self.device)
        elif self.model_str == 'grin':
            self.fitting_model = GRINet(n_nodes=args.n_nodes, d_in=args.data_dim, merge=self.args.data_pred.merge_policy).to(self.device)        
        elif self.model_str == 'cgcn':
            self.fitting_model = CGCN(d_in=args.n_nodes, window=args.window_step).to(self.device)
        elif self.model_str == 'timesnet':
            self.fitting_model = TimesNet(d_in=args.n_nodes, seq_len=args.window_step).to(self.device)
        elif self.model_str == 'mcuts':
            self.fitting_model = MCutsNet(n_nodes=args.n_nodes, d_in=args.data_dim, merge=self.args.data_pred.merge_policy).to(self.device)
        else:
            raise NotImplementedError
        
        self.data_pred_loss = nn.MSELoss()
        self.data_pred_mae = MaskedMAE().to(self.device)
        self.data_pred_mape = MaskedMAPE().to(self.device)
        self.data_pred_mse = MaskedMSE().to(self.device)
        self.data_pred_mre = MaskedMRE().to(self.device)
        self.data_pred_optimizer = torch.optim.Adam(self.fitting_model.parameters(),
                                                    lr=self.args.data_pred.lr_data_start,
                                                    weight_decay=self.args.data_pred.weight_decay)
        
        self.data_pred_scheduler = CosineAnnealingLR(self.data_pred_optimizer, T_max=self.max_epoch, eta_min=self.args.data_pred.lr_data_start*self.args.data_pred.lr_data_ratio)



    def set_graph_optimizer(self, epoch=None):
        if epoch == None:
            epoch = 0
        update_graph = []
        if self.args.group_policy is not None:
            update_graph.append(self.fwd_graphB)
            if self.update_bwd:
                update_graph.append(self.bwd_graphB)
        else:
            update_graph.append(self.fwd_graph)
            if self.update_bwd:
                update_graph.append(self.bwd_graph)
        
        self.graph_optimizer = torch.optim.Adam(update_graph, lr=self.args.graph_discov.lr_graph_start)
        self.graph_scheduler = CosineAnnealingLR(self.graph_optimizer, T_max=self.max_epoch, eta_min=self.args.graph_discov.lr_graph_start*self.args.graph_discov.lr_graph_ratio)
        for _ in range(epoch):
            self.graph_scheduler.step()

    
    
    def get_graph(self):
        if self.args.graph_discov.disable_graph:
            fwd_graph = torch.ones([self.args.n_nodes, self.args.n_nodes]).to(self.device)
            bwd_graph = torch.ones([self.args.n_nodes, self.args.n_nodes]).to(self.device)
            ind_graph = torch.ones([self.args.n_nodes, self.args.n_nodes]).to(self.device)
        elif self.args.graph_discov.use_true_graph:
            fwd_graph = self.true_cm
            bwd_graph = self.true_cm
            ind_graph = self.true_cm
        else:
            if self.args.group_policy is not None:
                fwd_graph = torch.mm(self.fwd_graphA, torch.sigmoid(self.fwd_graphB))
                
                if self.update_bwd:
                    bwd_graph = torch.mm(self.bwd_graphA, torch.sigmoid(self.bwd_graphB))
                else:
                    if not self.disable_bwd:
                        bwd_graph = fwd_graph.T
                    else:
                        bwd_graph = None
            
            else:
                fwd_graph = torch.sigmoid(self.fwd_graph)
                
                if self.update_bwd:
                    bwd_graph = torch.sigmoid(self.bwd_graph)
                else:
                    if not self.disable_bwd:
                        bwd_graph = fwd_graph.T
                    else:
                        bwd_graph = None
                
            if not self.disable_ind:
                if bwd_graph is not None:
                    ind_graph = (fwd_graph + bwd_graph)/2
                else:
                    ind_graph = (fwd_graph + fwd_graph.T)/2
            else:
                ind_graph = None
            
            # if hasattr(self.args, "graph_policy"):
            #     low_thres = float(self.args.graph_policy.split('-')[0])
            #     high_thres = float(self.args.graph_policy.split('-')[1])
            #     if fwd_graph is not None:
            #         fwd_graph[fwd_graph < low_thres] = 0.
            #         fwd_graph[fwd_graph > high_thres] = 1.
            #     if bwd_graph is not None:
            #         bwd_graph[bwd_graph < low_thres] = 0.
            #         bwd_graph[bwd_graph > high_thres] = 1.
            #     if ind_graph is not None:
            #         ind_graph[ind_graph < low_thres] = 0.
            #         ind_graph[ind_graph > high_thres] = 1.
            
        return fwd_graph, bwd_graph, ind_graph




    def latent_data_pred(self, x, y, observ_mask, train=True):
        
        def sample_graph(matrix, prob=True, diag=False):
            if matrix is None:
                return None
            min_value = np.min(matrix.detach().cpu().numpy())
            max_value = np.max(matrix.detach().cpu().numpy())
            if (max_value - min_value) > 0.01 and not self.cal_auc:
                matrixs = (matrix - min_value) / (max_value - min_value)
            else:
                matrixs = matrix
            if diag:
                matrix_masked = torch.mul(matrixs, self.graph_mask)
                return torch.bernoulli(matrix_masked) if prob else matrix_masked
            else:
                return torch.bernoulli(matrixs) if prob else matrixs
        
        self.data_pred_optimizer.zero_grad()
        
        fwd_graph, bwd_graph, ind_graph = self.get_graph()
        
        fwd_graph = sample_graph(fwd_graph, self.args.data_pred.prob, diag=False)
        bwd_graph = sample_graph(bwd_graph, self.args.data_pred.prob, diag=False)
        ind_graph = sample_graph(ind_graph, self.args.data_pred.prob, diag=True)
        
        if train:
            imputation, predictions = self.fitting_model(x, mask=observ_mask, fwd_graph=fwd_graph, bwd_graph=bwd_graph, ind_graph=ind_graph)
            if self.model_str == 'cgcn':
                imp_loss = F.mse_loss(y * observ_mask, imputation * observ_mask)
                smo_loss = F.l1_loss(imputation[:, :, 1:, :], imputation[:, :, :-1, :])
                pre_loss = sum([F.mse_loss(predictions[i][:, :, 0], y[:, :, i, 0]) for i in range(y.size(2))])
                spa_loss = torch.tensor(0.).to(self.device)
                for net in self.fitting_model.Causal.networks:
                    for param in net.parameters():
                        spa_loss += torch.norm(param, p=1) + torch.norm(param, p=2)
                loss = 20*imp_loss + 2*smo_loss + 0.1*pre_loss + 0.00001*spa_loss
                loss.backward(retain_graph=True)
            else:
                loss = self.data_pred_loss(y * observ_mask, imputation * observ_mask) / torch.mean(observ_mask)
                if len(predictions) > 0:
                    for pred in predictions:
                        loss += self.data_pred_loss(y * observ_mask, pred * observ_mask) / torch.mean(observ_mask)
                loss.backward(retain_graph=True)
            self.data_pred_optimizer.step()
        
        else:
            imputation = self.fitting_model(x, mask=observ_mask, fwd_graph=fwd_graph, bwd_graph=bwd_graph, ind_graph=ind_graph)
            loss = self.data_pred_loss(y * observ_mask, imputation * observ_mask) / torch.mean(observ_mask)
        
        return imputation, loss




    def graph_discov(self, x, y, observ_mask):

        def sigmoid_gumbel_sample(graph, tau=1, diag=False):
            if graph is None:
                return None
            prob = graph[:, :, None]
            logits = torch.concat([prob, (1-prob)], axis=-1)
            samples = gumbel_softmax(logits, tau=tau)[:, :, 0]
            if diag:
                samples_masked = torch.mul(samples, self.graph_mask)
                return samples_masked
            else:
                return samples

        self.graph_optimizer.zero_grad()
        
        fwd_graph, bwd_graph, ind_graph = self.get_graph()
        
        fwd_graph = sigmoid_gumbel_sample(fwd_graph, tau=self.gumbel_tau, diag=False)
        bwd_graph = sigmoid_gumbel_sample(bwd_graph, tau=self.gumbel_tau, diag=False)
        ind_graph = sigmoid_gumbel_sample(ind_graph, tau=self.gumbel_tau, diag=True)

        # print(self.graph.shape, sample_graph.shape)
        imputation, predictions = self.fitting_model(x, mask=observ_mask, fwd_graph=fwd_graph, bwd_graph=bwd_graph, ind_graph=ind_graph)
        
        # fwd_prob_graph = torch.sigmoid(fwd_graph)
        loss_sparsity1 = torch.linalg.matrix_norm(fwd_graph, 1)
        if bwd_graph is not None:
            # bwd_prob_graph = torch.sigmoid(bwd_graph)
            loss_sparsity2 = torch.linalg.matrix_norm(bwd_graph, 1)
        else:
            loss_sparsity2 = torch.tensor(0.).to(self.device)
        
        loss_data = self.data_pred_loss(y * observ_mask, imputation * observ_mask) / torch.mean(observ_mask)
        # loss_data = self.data_pred_loss(y * observ_mask, predictions[0] * observ_mask) / torch.mean(observ_mask)
        
        gs = fwd_graph.shape
        loss = (loss_sparsity1 + loss_sparsity2) / (gs[0] * gs[1]) * self.lambda_s + loss_data
        loss.backward()
        self.graph_optimizer.step()
        
        return loss, (loss_sparsity1 + loss_sparsity2)/2, loss_data



    def train_model(self, dataset, train_loader, latent_pred_step, pbar):
        train_loss = 0
        self.fitting_model.train()
        for batch_data in train_loader:
            latent_pred_step += self.args.batch_size
            # x = (batch_data['x'].to(self.device) - bias) / scale
            # y = (batch_data['y'].to(self.device) - bias) / scale
            x = batch_data['x'].to(self.device)
            y = batch_data['y'].to(self.device)
            batch_mask = batch_data['mask'].to(self.device)
            mask = torch.bernoulli(batch_mask.clone().detach().float() * 0.95)
            eval_mask = batch_data['eval_mask'].to(self.device)
            eval_mask = (batch_mask | eval_mask) - mask.byte()
            x = torch.mul(x, mask)
            
            y_pred, loss = self.latent_data_pred(x, y, mask)
            train_loss += loss.detach()
            y = dataset.scaler.inverse_transform(y)
            y_pred = dataset.scaler.inverse_transform(y_pred.detach())
            self.data_pred_mae.update(y_pred, y, eval_mask)
            self.data_pred_mape.update(y_pred, y, eval_mask)
            self.data_pred_mse.update(y_pred, y, eval_mask)
            self.data_pred_mre.update(y_pred, y, eval_mask)
            pbar.set_postfix_str(f"S1 loss={loss.detach().item():.2f}")
        
        self.data_pred_scheduler.step()
        train_loss /= len(train_loader)
        train_mae = self.data_pred_mae.compute().item()
        train_mape = self.data_pred_mape.compute().item()
        train_mse = self.data_pred_mse.compute().item()
        train_mre = self.data_pred_mre.compute().item()
        
        
        current_data_pred_lr = self.data_pred_optimizer.param_groups[0]['lr']
        self.log.log_metrics({"data_pred_train/lr": current_data_pred_lr}, latent_pred_step)
        self.log.log_metrics({"data_pred_train/train_loss": train_loss}, latent_pred_step)
        self.log.log_metrics({"data_pred_train/train_mae": train_mae}, latent_pred_step)
        self.log.log_metrics({"data_pred_train/train_mape": train_mape}, latent_pred_step)
        self.log.log_metrics({"data_pred_train/train_mse": train_mse}, latent_pred_step)
        self.log.log_metrics({"data_pred_train/train_mre": train_mre}, latent_pred_step)

        return latent_pred_step
    


    def train_graph(self, train_loader, graph_discov_step, pbar):
        
        for batch_data in train_loader:
            graph_discov_step += self.args.batch_size
            x = batch_data['x'].to(self.device)
            y = batch_data['y'].to(self.device)
            batch_mask = batch_data['mask'].to(self.device)
            mask = torch.bernoulli(batch_mask.clone().detach().float() * 0.95)
            x = torch.mul(x, mask)
            
            loss, loss_sparsity, loss_data = self.graph_discov(x, y, mask)
            
            self.log.log_metrics({"graph_discov/sparsity_loss": loss_sparsity.item(),
                                    "graph_discov/data_loss": loss_data.item(),
                                    "graph_discov/total_loss": loss.item()}, graph_discov_step)
            pbar.set_postfix_str(f"S2 loss={loss_data.item():.2f}, spr={loss_sparsity.item():.2f}")
    
        self.graph_scheduler.step()
        current_graph_discov_lr = self.graph_optimizer.param_groups[0]['lr']
        self.log.log_metrics({"graph_discov/lr": current_graph_discov_lr}, graph_discov_step)
        self.log.log_metrics({"graph_discov/lambda": self.lambda_s}, graph_discov_step)
        self.log.log_metrics({"graph_discov/tau": self.gumbel_tau}, graph_discov_step)
        self.lambda_s *= self.lambda_gamma
        self.gumbel_tau *= self.gumbel_tau_gamma

        return graph_discov_step

    
    
    def validate(self, dataset, val_loader, latent_pred_step, pbar):

        for batch_data in val_loader:
            x = batch_data['x'].to(self.device)
            y = batch_data['y'].to(self.device)
            mask = batch_data['mask'].to(self.device).float()
            eval_mask = batch_data['eval_mask'].to(self.device)
            x = torch.mul(x, mask)
            
            y_pred, loss = self.latent_data_pred(x, y, mask, train=False)
            y = dataset.scaler.inverse_transform(y)
            y_pred = dataset.scaler.inverse_transform(y_pred.detach())
            self.data_pred_mae.update(y_pred, y, eval_mask)
            self.data_pred_mape.update(y_pred, y, eval_mask)
            self.data_pred_mse.update(y_pred, y, eval_mask)
            self.data_pred_mre.update(y_pred, y, eval_mask)
            pbar.set_postfix_str(f"Validation loss={loss.detach().item():.2f}")
        
        val_mae = self.data_pred_mae.compute().item()
        val_mape = self.data_pred_mape.compute().item()
        val_mse = self.data_pred_mse.compute().item()
        val_mre = self.data_pred_mre.compute().item()
        
        self.log.log_metrics({"data_pred_val/val_mae": val_mae}, latent_pred_step)
        self.log.log_metrics({"data_pred_val/val_mape": val_mape}, latent_pred_step)
        self.log.log_metrics({"data_pred_val/val_mse": val_mse}, latent_pred_step)
        self.log.log_metrics({"data_pred_val/val_mre": val_mre}, latent_pred_step)
        
        pbar.update(1)

        return val_mse
    
    
    
    def test(self, dataset, test_loader, pbar):
        y_pred_test = []
        for batch_data in test_loader:
            x = batch_data['x'].to(self.device)
            y = batch_data['y'].to(self.device)
            mask = batch_data['mask'].to(self.device).float()
            eval_mask = batch_data['eval_mask'].to(self.device)
            x = torch.mul(x, mask)
            
            y_pred, loss = self.latent_data_pred(x, y, mask, train=False)
            y = dataset.scaler.inverse_transform(y)
            y_pred = dataset.scaler.inverse_transform(y_pred.detach())
            y_pred_test.append(y_pred.cpu().numpy())
            self.data_pred_mae.update(y_pred, y, eval_mask)
            self.data_pred_mape.update(y_pred, y, eval_mask)
            self.data_pred_mse.update(y_pred, y, eval_mask)
            self.data_pred_mre.update(y_pred, y, eval_mask)
        
        test_mae = self.data_pred_mae.compute().item()
        test_mape = self.data_pred_mape.compute().item()
        test_mse = self.data_pred_mse.compute().item()
        test_mre = self.data_pred_mre.compute().item()
        pbar.set_postfix_str(f"Test mse={test_mse:.2f}, spr=IDLE")
        self.log.log_metrics({"data_pred_test/test_mae": test_mae}, 0)
        self.log.log_metrics({"data_pred_test/test_mape": test_mape}, 0)
        self.log.log_metrics({"data_pred_test/test_mse": test_mse}, 0)
        self.log.log_metrics({"data_pred_test/test_mre": test_mre}, 0)
        
        return y_pred_test
    
    
    def run(self, dataset):
        torch_dataset = TorchDataset(dataset.df,
                                     mask=dataset.training_mask,
                                     eval_mask=dataset.eval_mask,
                                     window=self.args.window_step,
                                     stride=self.args.stride)
        
        if dataset.__class__.__name__ == 'AirQuality':
            dataset.data_path = "AirQuality"
            train_idxs, val_idxs, test_idxs = dataset.splitter(torch_dataset, val_len=0.1, in_sample=self.args.data_pred.in_sample, window=self.args.window_step)
            train_indice = torch_dataset.expand_indices(train_idxs, merge=True)
            if not self.args.data_pred.in_sample:
                torch_dataset.mask[train_indice] |= torch_dataset.eval_mask[train_indice]
        else:
            train_idxs, val_idxs, test_idxs = dataset.splitter(val_len=0.1, test_len=0.2, window=self.args.window_step)
        
        if dataset.data_path is None:
            self.cal_auc = True
        else:
            self.cal_auc = False
        
        trainset = Subset(torch_dataset, train_idxs if train_idxs is not None else [])
        valset = Subset(torch_dataset, val_idxs if val_idxs is not None else [])
        testset = Subset(torch_dataset, test_idxs if test_idxs is not None else [])
        
        Sampler = RandomSampler(trainset, replacement=True, num_samples=self.args.samples_per_epoch)
        train_loader = DataLoader(trainset, self.args.batch_size, sampler=Sampler, drop_last=True)
        val_loader = DataLoader(valset, self.args.batch_size, shuffle=False)
        test_loader = DataLoader(testset, self.args.batch_size, shuffle=False)
        
        self.true_cm = dataset.weight
        plot_causal_matrix_in_training(self.true_cm[:,:,None], "true_graph", self.log, 0, threshold=self.threshold)
        self.true_cm = torch.from_numpy(self.true_cm).float().to(self.device)
        
        if self.args.data_pred.model == "grin":
            self.fitting_model.adj = self.true_cm


        latent_pred_step = 0
        graph_discov_step = 0
        pbar = tqdm.tqdm(total=self.total_epoch)
        
        try:
            for epoch_i in range(self.total_epoch):
                
                # time_epoch_start = time.time()
                
                if self.args.group_policy is not None:
                    group_mul = int(self.args.group_policy.split("_")[1])
                    group_every = int(self.args.group_policy.split("_")[3])
                    if epoch_i % group_every == 0 and self.n_groups < self.args.n_nodes:
                        if epoch_i != 0:
                            self.n_groups *= group_mul
                        if self.n_groups > self.args.n_nodes:
                            self.n_groups = self.args.n_nodes
                        
                        self.fwd_graphA = torch.zeros([self.args.n_nodes, self.n_groups]).to(self.device)

                        for i in range(0, self.n_groups):
                            for j in range(0, self.args.n_nodes // self.n_groups):
                                self.fwd_graphA[i*(self.args.n_nodes // self.n_groups) + j, i] = 1
                        for k in range(i*(self.args.n_nodes // self.n_groups) + j, self.args.n_nodes):
                            self.fwd_graphA[k, i] = 1

                        # inv_A = torch.linalg.inv(torch.mm(torch.t(self.fwd_graphA), self.fwd_graphA))
                        # fwd_graphB_init = torch.mm(inv_A, torch.mm(torch.t(self.fwd_graphA), self.fwd_graph))

                        if hasattr(self, "fwd_graphB"):
                            fwd_graphB_init = torch.sigmoid(self.fwd_graphB).detach().cpu().repeat_interleave(group_mul, 0)[:self.n_groups, :]
                            fwd_graphB_init = 1 - (1 - fwd_graphB_init)**(1 / group_mul)
                        else:
                            fwd_graphB_init = torch.ones((self.n_groups, self.args.n_nodes))*0.5

                        self.fwd_graphB = nn.Parameter(fwd_graphB_init.to(self.device))
                        
                        if self.update_bwd:
                            self.bwd_graphA = self.fwd_graphA

                            # bwd_graphB_init = torch.mm(inv_A, torch.mm(torch.t(self.bwd_graphA), self.bwd_graph))

                            if hasattr(self, "bwd_graphB"):
                                bwd_graphB_init = torch.sigmoid(self.bwd_graphB).detach().cpu().repeat_interleave(group_mul, 0)[:self.n_groups, :]
                                bwd_graphB_init = 1 - (1 - bwd_graphB_init)**(1 / group_mul)
                            else:
                                bwd_graphB_init = torch.ones((self.n_groups, self.args.n_nodes))*0.5
                            
                            self.bwd_graphB = nn.Parameter(bwd_graphB_init.to(self.device))
                        
                        self.set_graph_optimizer(epoch=epoch_i)
                
                self.warming = False
                warmup = int(self.args.warmup)
                if warmup > 0:
                    if epoch_i < warmup:
                        self.fwd_graph = self.true_cm
                        if self.update_bwd:
                            self.bwd_graph = self.true_cm
                        self.warming = True


                # Training Step
                # Data Prediction
                
                
                latent_pred_step = self.train_model(torch_dataset, train_loader, latent_pred_step, pbar)
                
                
                # Graph Discovery
                # self.fitting_model.eval()

                
                if self.model_str == 'mcuts' and (not self.warming):
                    graph_discov_step = self.train_graph(train_loader, graph_discov_step, pbar)


                # Validation Step
                
                
                self.fitting_model.eval()
                val_mse = self.validate(torch_dataset, val_loader, latent_pred_step, pbar)
                
                if not self.cal_auc:
                    self.callback(val_mse, self.fitting_model)
                
                
                # Plot Graph
                
                if (epoch_i+1) % self.args.show_graph_every == 0:
                    if self.model_str == 'mcuts':
                        if self.args.group_policy is not None:
                            # fwd_matB = self.fwd_graphB.detach().cpu().numpy()[:,:,None] if self.fwd_graphB is not None else None
                            # bwd_matB = self.bwd_graphB.detach().cpu().numpy()[:,:,None] if self.bwd_graphB is not None else None
                            fwd_graph, bwd_graph, ind_graph = self.get_graph()
                            fwd_mat = fwd_graph.detach().cpu().numpy()[:,:,None] if fwd_graph is not None else None
                            bwd_mat = bwd_graph.detach().cpu().numpy()[:,:,None] if bwd_graph is not None else None
                            ind_mat = ind_graph.detach().cpu().numpy()[:,:,None] if ind_graph is not None else None
                            # plot_causal_matrix_in_training(fwd_matB, "fwd_graphB", self.log, graph_discov_step, threshold=self.threshold)
                            # plot_causal_matrix_in_training(bwd_matB, "bwd_graphB", self.log, graph_discov_step, threshold=self.threshold)
                            plot_causal_matrix_in_training(fwd_mat, "fwd_graph", self.log, graph_discov_step, threshold=self.threshold)
                            plot_causal_matrix_in_training(bwd_mat, "bwd_graph", self.log, graph_discov_step, threshold=self.threshold)
                            plot_causal_matrix_in_training(ind_mat, "ind_graph", self.log, graph_discov_step, threshold=self.threshold)
                            
                        else:
                            fwd_graph, bwd_graph, ind_graph = self.get_graph()
                            fwd_mat = fwd_graph.detach().cpu().numpy()[:,:,None] if fwd_graph is not None else None
                            bwd_mat = bwd_graph.detach().cpu().numpy()[:,:,None] if bwd_graph is not None else None
                            ind_mat = ind_graph.detach().cpu().numpy()[:,:,None] if ind_graph is not None else None
                            plot_causal_matrix_in_training(fwd_mat, "fwd_graph", self.log, graph_discov_step, threshold=self.threshold)
                            plot_causal_matrix_in_training(bwd_mat, "bwd_graph", self.log, graph_discov_step, threshold=self.threshold)
                            plot_causal_matrix_in_training(ind_mat, "ind_graph", self.log, graph_discov_step, threshold=self.threshold)

                        if self.cal_auc:
                            auc = calc_and_log_metrics(fwd_mat, self.true_cm.cpu().numpy(), self.log, graph_discov_step, threshold=self.threshold)
                            if self.callback is not None:
                                self.callback(-auc, self.fitting_model)
                    
                    elif self.model_str == 'cgcn':
                        fwd_graph = self.fitting_model.Causal.GC(threshold=False)
                        fwd_mat = fwd_graph.detach().cpu().numpy()[:,:,None]
                        plot_causal_matrix_in_training(fwd_mat, "fwd_graph", self.log, latent_pred_step, threshold=self.threshold)
                
                # time_epoch_end = time.time()
                # time_epoch = time_epoch_end - time_epoch_start
                # print("One epoch time: %.2f" % (time_epoch))
                # self.log.log_metrics({"data_pred_train/epoch_time": time_epoch}, latent_pred_step)
                
                if self.callback is not None and self.callback.early_stop:
                    print("Training Stopped!")
                    break
        
        except KeyboardInterrupt:
            pass
        
        # Testing Step

        if not self.cal_auc:
            
            # self.fitting_model.eval()
            # y_pred_test = self.test(torch_dataset, test_loader, pbar)
            
            # y_pred_test = np.concatenate(y_pred_test, axis=0)
            # test_indice = torch_dataset.expand_indices(testset.indices, merge=True)
            # eval_mask = dataset.eval_mask[test_indice]
            # df_true = dataset.df.iloc[test_indice]
            # metrics = {
            #     'mae': masked_mae,
            #     'mse': masked_mse,
            #     'mre': masked_mre,
            #     'mape': masked_mape
            # }
            # index = torch_dataset.data_timestamps(test_idxs)
            # df_hat = torch_dataset.predict_dataframe(y_pred_test, index, dataset.df.columns)
            # for metric_name, metric_fn in metrics.items():
            #     error = metric_fn(df_hat.values, df_true.values, eval_mask).item()
            #     print(f' {metric_name}: {error:.4f}')
            #     self.log.log_metrics({"data_pred_test/test_" + metric_name: error}, 1)
            
            # Use saved best model
            
            self.fitting_model.load_state_dict(torch.load(self.callback.path))
            y_pred_test = self.test(torch_dataset, test_loader, pbar)

            y_pred_test = np.concatenate(y_pred_test, axis=0)
            test_indice = torch_dataset.expand_indices(testset.indices, merge=True)
            eval_mask = dataset.eval_mask[test_indice]
            df_true = dataset.df.iloc[test_indice]
            metrics = {
                'mae': masked_mae,
                'mse': masked_mse,
                'mre': masked_mre,
                'mape': masked_mape
            }
            index = torch_dataset.data_timestamps(test_idxs)
            df_hat = torch_dataset.predict_dataframe(y_pred_test, index, dataset.df.columns)
            for metric_name, metric_fn in metrics.items():
                error = metric_fn(df_hat.values, df_true.values, eval_mask).item()
                print(f' {metric_name}: {error:.4f}')
                self.log.log_metrics({"data_pred_test/test_" + metric_name: error}, 1)
        
        else:
            print(f' AUC: {auc:.4f}')
            if self.callback is not None:
                print(f' Best AUC: {self.callback.best_score:.4f}')




def prepare_data(opt):
    if opt.pre_sample == "point_missing":
        p_fault, p_noise = 0., float(opt.missing_prob)
        max_seq, min_seq = 0, 0
    elif opt.pre_sample == "block_missing":
        p_fault, p_noise = float(opt.missing_prob), 0.1
        max_seq, min_seq = 48, 12
    else:
        raise NotImplementedError
    
    if not hasattr(opt, "dist_path"):
        opt.dist_path = None
    if not hasattr(opt, "reduce"):
        opt.reduce = "1P"
    if not hasattr(opt, "scale"):
        opt.scale = False
    
    if "pm25" in opt.name:
        if "part" in opt.name:
            dataset = AirQuality(path=opt.data_path, small=True, reduce=opt.reduce)
        elif "all" in opt.name:
            dataset = AirQuality(path=opt.data_path, small=False, reduce=opt.reduce)
        else:
            raise NotImplementedError
        dataset.generate_mask(p_fault, p_noise, max_seq, min_seq)
    elif any(name in opt.name for name in ["traffic", "electricity", "bay", "la"]):
        dataset = PandasDataset(opt.name, opt.data_path, opt.dist_path, opt.reduce, opt.scale)
        dataset.load()
        dataset.generate_mask(p_fault, p_noise, max_seq, min_seq)
    else:
        if opt.name == "uniform_var":
            data, beta, true_cm = simulate_var(**opt.param)
        elif opt.name == "random_var":
            data, true_cm = simulate_random_var(**opt.param)
        elif opt.name == "var":
            data, true_cm = simulate_var_from_links(**opt.param)
        elif opt.name == "lorenz96":
            data, true_cm = simulate_lorenz_96(**opt.param)
        elif opt.name == "netsim":
            data, true_cm = load_netsim_data(**opt.param)
        elif opt.name == "dream":
            data, true_cm = load_dream_data(**opt.param)
        elif opt.name == "spring":
            data, true_cm = load_springs_data(**opt.param)
        else:
            raise NotImplementedError
        
        dataset = PandasDataset(opt.name)
        dataset.load_data(data, true_cm)
        dataset.generate_mask(p_fault, p_noise, max_seq, min_seq)
    
    original_data = dataset.df.values
    mask = dataset.training_mask
    print("Data shape: ", original_data.shape)
    print(f"Generated random missing with missing_prob: {1-np.mean(mask):.4f}")
    
    return dataset


def main(opt: MultiCADopt, device="cuda"):
    reproduc(**opt.reproduc)
    
    timestamp = datetime.now().strftime("_%Y_%m%d_%H%M%S_%f")
    opt.task_name += timestamp
    proj_path = opj(opt.dir_name, opt.task_name)
    log = MyLogger(log_dir=proj_path, **opt.log)
    log.log_opt(opt)

    dataset = prepare_data(opt.data)
    
    mcuts = MCUTS(opt.multi_cad, log, device=device)
    mcuts.run(dataset)

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    parser = argparse.ArgumentParser(description='Batch Compress')
    parser.add_argument('-opt', type=str, default=opj(opd(__file__),
                        'opt/multi_cad_lorenz.yaml'), help='yaml file path')
    parser.add_argument('-g', help='available gpu list', default='1', type=str)
    # parser.add_argument('-debug', action='store_true')
    # parser.add_argument('-log', action='store_true')
    args = parser.parse_args()

    if args.g == "cpu":
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.g
        device = "cuda"
    
    main(OmegaConf.load(args.opt), device=device)

