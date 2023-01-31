import logging
import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops

import tqdm
import numpy as np
import argparse
from omegaconf import OmegaConf
from copy import deepcopy
import torch
from torch import dropout, nn

from utils.cuts_parts import *
from utils.gumbel_softmax import gumbel_softmax
from utils.misc import plot_causal_matrix, reproduc, plot_causal_matrix_in_training, calc_and_log_metrics, log_time_series, prepross_data
from utils.batch_generater import batch_generater
from utils.opt_type import CUTSopt
from utils.logger import MyLogger
from utils.data_interpolate import interp_multivar_data
from utils.load_data import simulate_var_from_links, simulate_var, simulate_lorenz_96_process, load_netsim_data

from datetime import datetime

import os
from einops import rearrange


class CUTS(object):
    def __init__(self, args: CUTSopt.CUTSargs, log, device="cuda"):
        self.log: MyLogger = log
        self.args = args
        self.device = device

        if self.args.data_pred.model == "multi_mlp":
            self.fitting_model = MultiMLP(self.args.input_step * self.args.n_nodes * self.args.data_dim,
                                          self.args.data_pred.mlp_hid,
                                          self.args.data_dim * self.args.data_pred.pred_step,
                                          self.args.data_pred.mlp_layers,
                                          self.args.n_nodes).to(self.device)
        elif self.args.data_pred.model == "multi_lstm":
            self.fitting_model = MultiLSTM(self.args.n_nodes * self.args.data_dim,
                                          self.args.data_pred.mlp_hid,
                                          self.args.data_dim * self.args.data_pred.pred_step,
                                          self.args.data_pred.mlp_layers,
                                          self.args.n_nodes).to(self.device)
        else:
            raise NotImplementedError

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
        
        if hasattr(self.args, "disable_graph") and self.args.disable_graph:
            print("Using full graph and disable graph discovery...")
            self.graph = nn.Parameter(torch.ones([self.args.n_nodes, self.args.n_nodes, self.args.input_step]).to(self.device) * 1000)
        else:
            self.graph = nn.Parameter(torch.ones([self.args.n_nodes, self.args.n_nodes, self.args.input_step]).to(self.device) * 0)
        # self.graph = nn.Parameter(torch.zeros([self.args.n_nodes, self.args.n_nodes, self.args.input_step]).to(self.device))
        self.graph_optimizer = torch.optim.Adam([self.graph], lr=self.args.graph_discov.lr_graph_start)
        gamma = (self.args.graph_discov.lr_graph_end / self.args.graph_discov.lr_graph_start) ** (1 / self.args.total_epoch)
        self.graph_scheduler = torch.optim.lr_scheduler.StepLR(self.graph_optimizer, step_size=1, gamma=gamma)

        end_tau, start_tau = self.args.graph_discov.end_tau, self.args.graph_discov.start_tau
        self.gumbel_tau_gamma = (end_tau / start_tau) ** (1 / self.args.total_epoch)
        self.gumbel_tau = start_tau
        self.start_tau = start_tau
        
        end_lmd, start_lmd = self.args.graph_discov.lambda_s_end, self.args.graph_discov.lambda_s_start
        self.lambda_gamma = (end_lmd / start_lmd) ** (1 / self.args.total_epoch)
        self.lambda_s = start_lmd
        
        

    def latent_data_pred(self, x, y, observ_mask):
        
        def sample_graph(sample_matrix, batch_size, prob=True):
            sample_matrix = torch.sigmoid(
                sample_matrix[None, :, :, :].expand(batch_size, -1, -1, -1))
            if prob:
                return torch.bernoulli(sample_matrix)
            else:
                return sample_matrix
        
        bs, n, m, t, d = x.shape
        self.fitting_model.train()
        self.data_pred_optimizer.zero_grad()
        
        # graph_no_self = self.graph.clone()
        # for i in range(graph_no_self.shape[0]):
        #     graph_no_self[i,i,:] = torch.ones_like(graph_no_self[i,i,:]) * -1000
        if hasattr(self.args.data_pred, "disable_graph") and \
            self.args.data_pred.disable_graph:
                sampled_graph = torch.ones_like(self.graph)[None].expand(bs, -1, -1, -1)
        else:
            sampled_graph = sample_graph(self.graph, bs, self.args.data_pred.prob)
            
        y_pred = self.fitting_model(x, sampled_graph)

        loss = self.data_pred_loss(y * observ_mask, y_pred * observ_mask) / torch.mean(observ_mask)
        loss.backward()
        self.data_pred_optimizer.step()
        return y_pred, loss

    def graph_discov(self, x, y, observ_mask):

        def sigmoid_gumbel_sample(graph, batch_size, tau=1):
            prob = torch.sigmoid(graph[None, :, :, :, None].expand(batch_size, -1, -1, -1, -1))
            logits = torch.concat([prob, (1-prob)], axis=-1)
            samples = gumbel_softmax(logits, tau=tau)[:, :, :, :, 0]
            return samples

        # self.fitting_model.eval()
        self.graph_optimizer.zero_grad()
        prob_graph = torch.sigmoid(self.graph[None, :, :])
        sample_graph = sigmoid_gumbel_sample(self.graph, self.args.batch_size, tau=self.gumbel_tau)

        y_pred = self.fitting_model(x, sample_graph)
        
        gs = prob_graph.shape
        loss_sparsity = torch.norm(prob_graph, p=1) / (gs[0] * gs[1] * gs[2])
        loss_data = self.data_pred_loss(y * observ_mask, y_pred * observ_mask) / torch.mean(observ_mask)
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
                if hasattr(self.args, "sample_period"):
                    sample_period = self.args.sample_period
                else:
                    sample_period = 1
                ## 
                batch_gen = batch_generater(data, observ_mask, # !!!!! TO-DO
                                            bs=self.args.batch_size, 
                                            n_nodes=self.args.n_nodes, 
                                            input_step=self.args.input_step, 
                                            pred_step=self.args.data_pred.pred_step,
                                            sample_period=sample_period)
                batch_gen = list(batch_gen)
                
                data_pred = deepcopy(data) # masked data points are predicted
                data_pred_all = deepcopy(data)
                for x, y, t, mask in batch_gen:
                    latent_pred_step += self.args.batch_size
                    y_pred, loss = self.latent_data_pred(x, y, mask)
                    data_pred[t] = (y_pred*(1-mask) + y*mask).clone().detach()[:,:,0]
                    data_pred_all[t] = y_pred.clone().detach()[:,:,0]
                    self.log.log_metrics({"latent_data_pred/pred_loss": loss.item()}, latent_pred_step)
                    pbar.set_postfix_str(f"S1 loss={loss.item():.2f}, spr=IDLE, auc={auc:.4f}")

                current_data_pred_lr = self.graph_optimizer.param_groups[0]['lr']
                self.log.log_metrics({"graph_discov/lr": current_data_pred_lr}, latent_pred_step)
                self.data_pred_scheduler.step()
                mse_pred_to_original = self.data_pred_loss(original_data, data_pred)
                mse_interp_to_original = self.data_pred_loss(original_data, data_interp)
                
                self.log.log_metrics({"latent_data_pred/mse_pred_to_original": mse_pred_to_original,
                                      "latent_data_pred/mse_interp_to_original": mse_interp_to_original}, latent_pred_step)
            
            # Graph Discovery
            if hasattr(self.args, "graph_discov"):
                # batch_gen = batch_generater(data, observ_mask, 
                #                             bs=self.args.batch_size, 
                #                             n_nodes=self.args.n_nodes, 
                #                             input_step=self.args.input_step, 
                #                             pred_step=self.args.data_pred.pred_step, 
                #                             sample_period=period)
                for x, y, t, mask in batch_gen:
                    graph_discov_step += self.args.batch_size
                    if hasattr(self.args, "disable_graph") and self.args.disable_graph:
                        pass
                    else:
                        loss, loss_sparsity, loss_data = self.graph_discov(x, y, mask)
                        self.log.log_metrics({"graph_discov/sparsity_loss": loss_sparsity.item(),
                                            "graph_discov/data_loss": loss_data.item(),
                                            "graph_discov/total_loss": loss.item()}, graph_discov_step)
                        pbar.set_postfix_str(f"S2 loss={loss_data.item():.2f}, spr={loss_sparsity.item():.2f}, auc={auc:.4f}")
                    
                self.graph_scheduler.step()
                current_graph_disconv_lr = self.graph_optimizer.param_groups[0]['lr']
                self.log.log_metrics({"graph_discov/lr": current_graph_disconv_lr}, graph_discov_step)
                self.log.log_metrics({"graph_discov/tau": self.gumbel_tau}, graph_discov_step)
                self.gumbel_tau *= self.gumbel_tau_gamma

            pbar.update(1)
            self.lambda_s *= self.lambda_gamma
                     
            calc, val = self.args.causal_thres.split("_")
            if calc == "value":
                threshold = float(val)
            else:
                raise NotImplementedError
            
            time_coef_mat = self.graph.detach().cpu().numpy()
            plot_roc = False
            if (epoch_i+1) % self.args.show_graph_every == 0:
                avg_mask = np.mean(observ_mask.cpu().numpy(), axis=(0,2))
                if np.min(avg_mask) < 1:
                    time_series_idx = int(np.argwhere(avg_mask < 1)[0])
                else:
                    time_series_idx = 0
                log_time_series(original_data.cpu()[-100:,time_series_idx], 
                                data_interp.cpu()[-100:,time_series_idx], 
                                data_pred_all.cpu()[-100:,time_series_idx], log=self.log, log_step=latent_pred_step)
                plot_causal_matrix_in_training(time_coef_mat, self.log, graph_discov_step, threshold=threshold)
                plot_roc = True
            
            # Show TPR FPR AUC ROC
            if true_cm is not None:
                time_prob_mat = torch.sigmoid(self.graph).detach().cpu().numpy()      
                auc = calc_and_log_metrics(time_prob_mat, true_cm, self.log, graph_discov_step, threshold=threshold, plot_roc=plot_roc)
                


def prepare_data(opt):
    if opt.name == "uniform_var":
        data, beta, true_cm = simulate_var(**opt.param)
    elif opt.name == "var":
        data, true_cm = simulate_var_from_links(**opt.param)
    elif opt.name == "lorenz_96":
        data, true_cm = simulate_lorenz_96_process(**opt.param)
    elif opt.name == "zeros": # for debug
        data = np.zeros([opt.param.T, opt.param.N, 1])
    elif opt.name == "netsim":
        data, true_cm = load_netsim_data(**opt.param)
    else:
        raise NotImplementedError

    T, N, D = data.shape
    print("Data shape: ", data.shape)
    data = prepross_data(data)
    
    mask = np.ones_like(data)
    if hasattr(opt.pre_sample, "period") or hasattr(opt.pre_sample, "random_period"):
        if hasattr(opt.pre_sample, "period"):
            assert N == len(opt.pre_sample.period), "opt.pre_sample.period length not matched"
            period = opt.pre_sample.period
            print("Using sampling periods: ", period)
        elif hasattr(opt.pre_sample, "random_period"):
            np.random.seed(opt.pre_sample.random_period.seed)
            period = np.random.choice(opt.pre_sample.random_period.choices, N, p=opt.pre_sample.random_period.prob)
            print("Generated presampling periods: ", period)
        mask *= 0
        for i in range(N):
            period_i = period[i]
            mask[::period_i, i] += 1
            
    elif hasattr(opt.pre_sample, "random_missing"):
        np.random.seed(opt.pre_sample.random_missing.seed)
        p = opt.pre_sample.random_missing.missing_prob
        missing_var = opt.pre_sample.random_missing.missing_var
        if isinstance(missing_var, str) and missing_var=="all":
            mask = np.random.choice([0,1], size=mask.shape, p=[p,1-p])
        else:
            for var_i in missing_var:
                mask[:,var_i] = np.random.choice([0,1], size=mask[:,var_i].shape, p=[p,1-p])
        print(f"Generated random missing with missing_prob: {p:.4f}")
    else:
        raise NotImplementedError
        

    sampled_data = data * mask
    interp_data = interp_multivar_data(sampled_data, mask, interp=opt.init_fill)
    return interp_data, mask, true_cm, data


def main(opt: CUTSopt, device="cuda"):
    reproduc(**opt.reproduc)
    timestamp = datetime.now().strftime("_%Y_%m%d_%H%M%S_%f")
    opt.task_name += timestamp
    proj_path = opj(opt.dir_name, opt.task_name)
    log = MyLogger(log_dir=proj_path, **opt.log)
    log.log_opt(opt)

    data, mask, true_cm, original_data = prepare_data(opt.data)
    # data = data / 20
    
    if true_cm is not None:
        sub_cg = plot_causal_matrix(
            true_cm, 
            figsize=[1.5*data.shape[1], 1*data.shape[1]])
        log.log_figures(name="True Graph", figure=sub_cg, iters=0)
    
    if hasattr(opt, "cuts"):
        cuts = CUTS(opt.cuts, log, device=device)
        cuts.train(data, mask, original_data, true_cm)
    
                

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    parser = argparse.ArgumentParser(description='Batch Compress')
    parser.add_argument('-opt', type=str, default=opj(opd(__file__),
                        'cuts_example.yaml'), help='yaml file path')
    parser.add_argument('-g', help='availabel gpu list', default='0', type=str)
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
