import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops

import tqdm
import numpy as np
import argparse
import multiprocessing
from omegaconf import OmegaConf
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from datetime import datetime
from einops import rearrange
import warnings

from nets.loss import Causal_Loss
from utils.misc import (
    feature_sel_txt,
    reproduc,
    plot_feature_sel,
    score_dict_2_string,
    read_dict_from_csv,
)
from utils.misc import eval_multitask_performance
from nets.prob_graph import bernonlli_sample, gumbel_sample, freeze_graph, split_dy_st, cumulative_time_graph
from utils.opt_type import MultiCADopt
from utils.logger import MyLogger
from utils.model_setup import build_net, build_optim, build_loss


class Granger_Causal_Prediction(object):
    def __init__(self, args: MultiCADopt.MultiCADargs, log, device="cuda"):
        self.log: MyLogger = log
        self.args = args
        self.device = device

        if isinstance(self.args.data_pred.pred_dim, list):
            self.task_num = len(self.args.data_pred.pred_dim)
        else:
            self.task_num = 1
        self.task_names = self.args.data_pred.task_names

        self.fitting_model, self.graph = build_net(self.args, self.device)

        self.data_pred_optimizer, self.data_pred_scheduler = build_optim(self.fitting_model, self.args.data_pred, self.args.total_epoch)

        if self.args.local_expl.enable:
            self.local_expl_loss = build_loss(self.args.local_expl.loss)

        self.data_pred_loss = build_loss(self.args.data_pred.loss)

        self.gumbel_tau = 1
        if hasattr(self.args, "graph_discov") and self.args.graph_discov != "none":
            end_tau, start_tau = (
                self.args.graph_discov.end_tau,
                self.args.graph_discov.start_tau,
            )
            self.gumbel_tau_gamma = (end_tau / start_tau) ** (1 / self.args.total_epoch)
            self.gumbel_tau = start_tau

            end_lmd, start_lmd = (
                self.args.graph_discov.lambda_s_end,
                self.args.graph_discov.lambda_s_start,
            )
            self.lambda_gamma = (end_lmd / start_lmd) ** (1 / self.args.total_epoch)
            self.lambda_s = start_lmd

            self.graph_loss = Causal_Loss(lambda_s=self.lambda_s, data_loss=self.data_pred_loss, norm_by_shape=self.args.graph_discov.norm_by_shape)
            self.graph_optimizer, self.graph_scheduler = build_optim(self.graph, self.args.graph_discov, self.args.total_epoch)

    def data_pred(self, x_dy, x_st, label, mode="train", ref="zero", show_local=False):
        x_dy, x_st = x_dy.to(self.device), x_st.to(self.device)
        label = [v.to(self.device) for v in label]

        bs, t, n_dy, d_dy = x_dy.shape
        bs, n_st, d_st = x_st.shape
        if mode == "train":
            self.fitting_model.train()
        else:
            self.fitting_model.eval()

        self.data_pred_optimizer.zero_grad()

        sampled_graph = bernonlli_sample(self.graph, bs, self.args.data_pred.prob, self.args.data_pred.hard_mask, t_length=self.args.t_length, time_cumu_type=self.args.time_graph.time_cumu_type, time_cumulative=self.args.time_graph.enable)
        if hasattr(self.args, "full_graph") and self.args.full_graph:
            sampled_graph = torch.ones_like(sampled_graph)

        sampled_dy, sampled_st = split_dy_st(
            sampled_graph,
            self.args.dy_feat_num,
            self.args.st_feat_num,
            batch_dim=True,
            time_dim=self.args.time_graph.enable,
        )
        out = self.fitting_model(x_dy, x_st, sampled_dy, sampled_st, ref=ref, tau=self.gumbel_tau, suspend_local_expl=self.epoch_i < self.args.local_expl.start_after)
        if self.args.local_expl.enable:
            y_pred, dy_local, st_local = out
            if show_local:
                for i in range(0, dy_local.shape[0], dy_local.shape[0] // 5):
                    sub_cg = plot_feature_sel(dy_local[i].detach().cpu(), figsize=[40, 10], show_text=False)
                    self.log.log_figures(sub_cg, name=f"local/dy_local_{i}", iters=self.epoch_i, exclude_logger="tblogger")
                    sub_cg = plot_feature_sel(st_local[i].detach().cpu(), figsize=[40, 10])
                    self.log.log_figures(sub_cg, name=f"local/st_local_{i}", iters=self.epoch_i, exclude_logger="tblogger")
        else:
            y_pred = out

        # for i in range(0, dy_local.shape[0], 100):
        #     sub_cg = plot_feature_sel(dy_local[i].detach().cpu(), figsize=[40, 10], show_text=False)
        #     self.log.log_figures(sub_cg, name=f"local/dy_local_{i}", iters=0)
        #     sub_cg = plot_feature_sel(st_local[i].detach().cpu(), figsize=[40, 10])
        #     self.log.log_figures(sub_cg, name=f"local/st_local_{i}", iters=0)

        if not isinstance(y_pred, list):
            y_pred = [y_pred]
        loss = self.data_pred_loss(y_pred, label)
        local_expl_loss = torch.tensor(0.0).to(self.device)
        if self.args.local_expl.enable and self.epoch_i >= self.args.local_expl.start_after:
            local_expl_loss = self.local_expl_loss(dy_local) + self.local_expl_loss(st_local)
            loss += local_expl_loss * self.args.local_expl.lambda_s

        if mode == "train":
            loss.backward()
            self.data_pred_optimizer.step()

        return y_pred, loss, local_expl_loss

    def graph_discov(self, x_dy, x_st, label, ref="zero"):
        x_dy, x_st = x_dy.to(self.device), x_st.to(self.device)
        label = [v.to(self.device) for v in label]

        bs, t, n_dy, d_dy = x_dy.shape
        bs, n_st, d_st = x_st.shape

        self.fitting_model.train()

        self.graph_optimizer.zero_grad()
        sampled_graph, prob_graph = gumbel_sample(self.graph, bs, tau=self.gumbel_tau, t_length=self.args.t_length, time_cumu_type=self.args.time_graph.time_cumu_type, time_cumulative=self.args.time_graph.enable, time_dim=self.args.time_graph.enable)

        sampled_dy, sampled_st = split_dy_st(
            sampled_graph,
            self.args.dy_feat_num,
            self.args.st_feat_num,
            batch_dim=True,
            time_dim=self.args.time_graph.enable,
        )
        out = self.fitting_model(x_dy, x_st, sampled_dy, sampled_st, ref=ref, tau=self.gumbel_tau, suspend_local_expl=self.epoch_i < self.args.local_expl.start_after)
        if self.args.local_expl.enable:
            y_pred, dy_local, st_local = out
        else:
            y_pred = out

        if not isinstance(y_pred, list):
            y_pred = [y_pred]

        dy_graph, st_graph = split_dy_st(
            prob_graph,
            self.args.dy_feat_num,
            self.args.st_feat_num,
            batch_dim=False,
            time_dim=self.args.time_graph.enable,
        )
        # dy_weight = self.args.dy_feat_num / (self.args.dy_feat_num + self.args.st_feat_num)
        dy_weight = 0.5

        loss, loss_sparsity, loss_data = self.graph_loss(y_pred, label, [dy_graph, st_graph], [dy_weight, 1 - dy_weight])

        loss.backward()
        self.graph_optimizer.step()
        return loss, loss_sparsity, loss_data

    def train(
        self,
        train_dataset: torch.utils.data.dataset.Subset,
        val_dataset: torch.utils.data.dataset.Subset,
        test_dataset,
    ):
        if hasattr(self.args, "load_graph_dir") and self.args.load_graph_dir != "none":
            self.load_graph(self.args.load_graph_dir, test_dataset)
        
        print(f"Train set num: {len(train_dataset):d}")
        print(f"Val set num: {len(val_dataset):d}")
        print(f"Test set num: {len(test_dataset):d}")

        params = {
            "num_workers": 0,
        }
        params.update(self.args.dataloader)

        val_test_params = deepcopy(params)
        val_test_params.pop("shuffle")
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            collate_fn=val_dataset.dataset.get_collate_fn(),
            **val_test_params,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            collate_fn=test_dataset.get_collate_fn(),
            **val_test_params,
        )

        dy_item = train_dataset.dataset.dynamic_items
        st_item = train_dataset.dataset.static_items
        if hasattr(train_dataset.dataset, "name_lut"):
            dy_item = [test_dataset.name_lut.get(n, n) for n in dy_item]
            st_item = [test_dataset.name_lut.get(n, n) for n in st_item]
        total_step = 0
        print("Dy items: ", str(dy_item))
        print("St items: ", str(st_item))
        # epoch_size = self.args.max_batch_num * self.args.batch_size
        # epoch_train_dataset = torch.utils.data.random_split(train_dataset, [epoch_size, len(train_dataset) - epoch_size])[0]
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            collate_fn=train_dataset.dataset.get_collate_fn(),
            **params,
        )
        batch_num = len(train_loader)
        print("Batch num: ", batch_num)

        for epoch_i in range(1, self.args.total_epoch + 1):

            self.epoch_i = epoch_i
            all_scores = {}

            if hasattr(self.args, "graph_discov") and self.args.graph_discov != "none" and epoch_i > self.args.graph_discov.start_after:
                self.args.full_graph = False
            else:
                self.args.full_graph = True

            torch.cuda.empty_cache()
            # Data Prediction
            if hasattr(self.args, "data_pred"):
                pred_epoch = []
                label_epoch = []
                print(f"Epoch {epoch_i:d} Data Prediction")
                pbar = tqdm.tqdm(total=batch_num)
                for batch_i, (dynamic_data, static_data, label) in enumerate(train_loader):
                    pred_batch, loss, local_expl_loss = self.data_pred(dynamic_data, static_data, label, mode="train", show_local=(batch_i % (batch_num // 5 + 1) == 0))
                    if len(pred_epoch) < 1000:
                        pred_epoch.append([d.detach().cpu() for d in pred_batch])
                        label_epoch.append([d.detach().cpu() for d in label])
                    total_step += 1

                    self.log.log_metrics({"latent_data_pred/pred_loss": loss.item()}, iters=total_step)
                    self.log.log_metrics({"latent_data_pred/local_expl_loss": local_expl_loss.item()}, iters=total_step)
                    pbar.set_postfix_str(f"A0 loss={loss.item():.6f}")
                    pbar.update(1)
                    if debug:
                        break

                pbar.close()
                current_data_pred_lr = self.data_pred_optimizer.param_groups[0]["lr"]
                self.log.log_metrics({"latent_data_pred/lr": current_data_pred_lr}, iters=epoch_i)
                self.data_pred_scheduler.step()

                scores, figures, _, _ = eval_multitask_performance(
                    pred_epoch,
                    label_epoch,
                    names=["train/" + n for n in self.task_names],
                    take_samples=1000,
                )
                self.log.log_metrics(scores, figures, iters=epoch_i)
                all_scores.update(scores)

            # Graph Discovery
            if hasattr(self.args, "graph_discov") and self.args.graph_discov != "none":
                if epoch_i > self.args.graph_discov.start_after:
                    print(f"Epoch {epoch_i:d} Graph Discovery")
                    pbar = tqdm.tqdm(total=batch_num)
                    for cycle_i in range(self.args.graph_discov.train_cycle):
                        for batch_i, (dynamic_data, static_data, label) in enumerate(train_loader):
                            total_step += 1
                            if hasattr(self.args, "full_graph") and self.args.full_graph:
                                pass
                            else:
                                loss, loss_sparsity, loss_data = self.graph_discov(
                                    dynamic_data,
                                    static_data,
                                    label,
                                )
                                self.log.log_metrics(
                                    {
                                        "graph_discov/sparsity_loss": loss_sparsity.item(),
                                        "graph_discov/data_loss": loss_data.item(),
                                        "graph_discov/total_loss": loss.item(),
                                    },
                                    iters=total_step,
                                )
                                pbar.set_postfix_str(f"B{cycle_i:d} loss={loss_data.item():.2f}")
                                pbar.update(1)
                            if debug:
                                break

                    pbar.close()
                    self.graph_scheduler.step()
                    current_graph_disconv_lr = self.graph_optimizer.param_groups[0]["lr"]
                    self.log.log_metrics({"graph_discov/lr": current_graph_disconv_lr}, iters=epoch_i)
                    self.log.log_metrics({"graph_discov/tau": self.gumbel_tau}, iters=epoch_i)
                    self.gumbel_tau *= self.gumbel_tau_gamma
                    self.lambda_s *= self.lambda_gamma

                if hasattr(self.args.graph_discov, "freeze_graph_after"):
                    if epoch_i == self.args.graph_discov.freeze_graph_after:
                        freeze_graph(self.args, self.graph)

            if hasattr(self.args, "data_pred") and (epoch_i) % self.args.valid_every == 0:
                # Data Prediction Validation
                pred_epoch = []
                label_epoch = []
                print(f"Epoch {epoch_i:d} Validation")
                pbar = tqdm.tqdm(total=len(val_loader))
                for dynamic_data, static_data, label in val_loader:
                    pred_batch, loss, local_expl_loss = self.data_pred(dynamic_data, static_data, label, mode="valid")
                    pred_epoch.append([d.detach().cpu() for d in pred_batch])
                    label_epoch.append([d.detach().cpu() for d in label])
                    torch.cuda.empty_cache()

                    pbar.set_postfix_str(f"VAL loss={loss.item():.6f}")
                    pbar.update(1)
                    if debug:
                        break

                pbar.close()
                scores, figures, _, _ = eval_multitask_performance(pred_epoch, label_epoch, names=["val/" + n for n in self.task_names])
                self.log.log_metrics(scores, figures, iters=epoch_i)
                all_scores.update(scores)

            if hasattr(self.args, "data_pred") and (epoch_i) % self.args.test_every == 0:
                # Data Prediction Testing
                pred_epoch = []
                label_epoch = []
                print(f"Epoch {epoch_i:d} Testing")
                pbar = tqdm.tqdm(total=len(test_loader))
                for dynamic_data, static_data, label in test_loader:
                    pred_batch, loss, local_expl_loss = self.data_pred(dynamic_data, static_data, label, mode="test")
                    pred_epoch.append([d.detach().cpu() for d in pred_batch])
                    label_epoch.append([d.detach().cpu() for d in label])
                    torch.cuda.empty_cache()

                    pbar.set_postfix_str(f"TST loss={loss.item():.6f}")
                    pbar.update(1)
                    if debug:
                        break

                pbar.close()
                scores, figures, tasks_preds, tasks_labels = eval_multitask_performance(
                    pred_epoch,
                    label_epoch,
                    names=["test/" + n for n in self.task_names],
                )

                # # List加入self.task_names转换为dict
                # tasks_preds = dict(zip(self.task_names, tasks_preds))
                # tasks_labels = dict(zip(self.task_names, tasks_labels))
                # torch.save(
                #     (tasks_preds, tasks_labels),
                #     opj(self.log.log_dir, f"iter_{epoch_i:d}", "test_predictions.pt"),
                # )
                self.log.log_metrics(scores, figures, epoch_i)
                all_scores.update(scores)

            self.log.log_txt(score_dict_2_string(all_scores), "scores.txt", epoch_i)

            if (epoch_i) % self.args.show_graph_every == 0:
                prob_graph = torch.sigmoid(self.graph)
                if self.args.time_graph.enable:
                    prob_graph = cumulative_time_graph(prob_graph, self.args.t_length, self.args.time_graph.time_cumu_type)
                prob_graph = prob_graph.detach().cpu().numpy()

                # Show Thresholded Graph
                if not isinstance(st_item, list):
                    st_item = st_item.tolist()
                if not isinstance(dy_item, list):
                    dy_item = dy_item.tolist()

                dy_graph, st_graph = split_dy_st(
                    prob_graph,
                    self.args.dy_feat_num,
                    self.args.st_feat_num,
                    batch_dim=False,
                    time_dim=self.args.time_graph.enable,
                )
                if self.args.time_graph.enable:
                    chunk_size = self.args.t_length // self.args.time_graph.time_chunk_num
                    sub_cg = plot_feature_sel(dy_graph[::chunk_size], class_names=dy_item, figsize=[40, 10])
                    self.log.log_figures(sub_cg, name="dy_graph", iters=epoch_i)
                    sub_cg = plot_feature_sel(st_graph, class_names=st_item, figsize=[40, 10])
                    self.log.log_figures(sub_cg, name="st_graph", iters=epoch_i)

                dy_log_text = feature_sel_txt(dy_graph, class_names=dy_item)
                st_log_text = feature_sel_txt(st_graph, class_names=st_item)
                self.log.log_txt(dy_log_text, name="dy_sel.csv", iters=epoch_i)
                self.log.log_txt(st_log_text, name="st_sel.csv", iters=epoch_i)

            if (epoch_i) % self.args.save_model_every == 0 or epoch_i == self.args.total_epoch - 1:
                test_path = opj(self.log.log_dir, f"iter_{epoch_i:d}", "model.pt")
                torch.save(self.fitting_model.state_dict(), test_path)
        return opd(test_path)

    def load_graph(self, load_dir, test_dataset):
        dy_items = test_dataset.dynamic_items
        st_items = test_dataset.static_items
        if hasattr(test_dataset, "name_lut"):
            dy_items = [test_dataset.name_lut.get(n, n) for n in dy_items]
            st_items = [test_dataset.name_lut.get(n, n) for n in st_items]

        # For debug
        # test_dataset.patient_data_list = test_dataset.patient_data_list[:10]
        # test_dataset.sample_each_patient = test_dataset.sample_each_patient[:10]

        print(f"Test set num: {len(test_dataset):d}")

        dy_graph_dict = read_dict_from_csv(opj(load_dir, "dy_sel.csv"))
        st_graph_dict = read_dict_from_csv(opj(load_dir, "st_sel.csv"))

        print("Dy graph: ", str(dy_graph_dict))
        print("St graph: ", str(st_graph_dict))

        dy_graph = torch.tensor([dy_graph_dict[str(dy_item)] for dy_item in dy_items]).to(self.graph.device)
        st_graph = torch.tensor([st_graph_dict[str(st_item)] for st_item in st_items]).to(self.graph.device)
        print("Graph shape: ", dy_graph.shape, st_graph.shape)
        dy_graph = dy_graph[:, 0]
        st_graph = st_graph[:, 0]
        # dy_graph = torch.ones(self.args.t_length, self.args.dy_feat_num).to(self.graph.device)
        # st_graph = torch.ones(self.args.t_length, self.args.st_feat_num).to(self.graph.device)

        thres = 0.5
        self.graph = (torch.cat([dy_graph, st_graph], dim=-1) > thres).float()  # binarize
        self.graph = self.graph * 200 - 100  # scale to -100 to 100
        self.args.data_pred.hard_mask = True  # hard thresholding instead of bernonlli sampling
        
    
    def test(self, load_dir, val_dataset, test_dataset):
        # 这个test和上面的训练过程中的test的区别是
        # 训练过程中的test是会按照概率对因果图进行采样，这里会直接以0.5为阈值进行二值化
        # 这个函数是在训练结束后进行的测试

        train_log_dir = self.log.log_dir
        self.log.log_dir = opj(load_dir)
        self.log.log_txt(load_dir + f"\n{train_log_dir}", "test_model_dir.txt")

        self.load_graph(load_dir, test_dataset)

        model_dir = opj(load_dir, "model.pt")
        self.fitting_model.load_state_dict(torch.load(model_dir))
        self.epoch_i = 0
        print("Model Loaded.")

        params = {
            "num_workers": 0,
        }
        params.update(self.args.dataloader)
        params["shuffle"] = False

        for name, dataset in {
            "test": test_dataset,
            "val": val_dataset,
        }.items():
            if dataset is None:
                print(f"No {name} dataset.")
                continue
            
            if isinstance(dataset, torch.utils.data.Subset):
                ori_dataset = dataset.dataset
            else:
                ori_dataset = dataset
            
            print(f"Testing on {name} set, length: {len(dataset):d}")
            test_loader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                collate_fn=ori_dataset.get_collate_fn(),
                **params,
            )

            all_scores = {}

            # Data Prediction Testing
            pred_epoch = []
            label_epoch = []

            for i, (dynamic_data, static_data, label) in enumerate(tqdm.tqdm(test_loader)):
                pred_batch, loss, local_expl_loss = self.data_pred(dynamic_data, static_data, label, mode="test")
                pred_epoch.append([d.detach().cpu() for d in pred_batch])
                label_epoch.append([d.detach().cpu() for d in label])
                torch.cuda.empty_cache()
                if debug:
                    break
                if i % 10 == 0:
                    # Converting to list of tasks
                    scores, figures, tasks_preds, tasks_labels = eval_multitask_performance(pred_epoch, label_epoch, names=[f"{name}/" + n for n in self.task_names])
                    print(scores)

            # Converting to list of tasks
            scores, figures, tasks_preds, tasks_labels = eval_multitask_performance(pred_epoch, label_epoch, names=[f"{name}/" + n for n in self.task_names])
            self.log.log_metrics(scores, figures, 0)
            all_scores.update(scores)

            self.log.log_txt(score_dict_2_string(all_scores), "scores.txt", 0)

            # List加入self.task_names转换为dict
            tasks_preds = dict(zip(self.task_names, tasks_preds))
            tasks_labels = dict(zip(self.task_names, tasks_labels))
            
            if isinstance(dataset, torch.utils.data.Subset):
                dataset_name = dataset.dataset.data_cache_path.split(os.sep)[-1].split(".")[0]
            else:
                dataset_name = dataset.data_cache_path.split(os.sep)[-1].split(".")[0]
            torch.save((tasks_preds, tasks_labels), opj(load_dir, f"{name}_{dataset_name}_predictions.pt"))
        self.log.log_dir = train_log_dir


def prepare_data(opt):
    if opt.name == "mimic":
        from data_prep.mimic_data.medical_dataset import MedicalDataset
        from data_prep.mimic_data.patient_data import PatientData

        if hasattr(opt, "test"):
            test_dataset = MedicalDataset(**opt.test, **opt.shared_param)
        else:
            test_dataset = None
            warnings.warn("No test dataset.")

        if hasattr(opt, "train_val"):
            train_val_dataset = MedicalDataset(**opt.train_val, **opt.shared_param)
            train_size = int(len(train_val_dataset) * 0.8)
            # Split train and val
            # train_dataset, val_dataset = torch.utils.data.random_split(
            #     train_val_dataset, [train_size, len(train_val_dataset) - train_size]
            # )
            train_dataset = torch.utils.data.Subset(train_val_dataset, range(train_size))
            val_dataset = torch.utils.data.Subset(train_val_dataset, range(train_size, len(train_val_dataset)))
        else:
            train_dataset = None
            val_dataset = None
            warnings.warn("No train_val dataset.")

        return train_dataset, val_dataset, test_dataset

    elif opt.name == "pla":
        from data_prep.pla_data.medical_dataset import MedicalDataset
        from data_prep.pla_data.patient_data import PatientData

        if hasattr(opt, "test"):
            test_dataset = MedicalDataset(**opt.test, **opt.shared_param)
        else:
            test_dataset = None
            warnings.warn("No test dataset.")

        if hasattr(opt, "train_val"):
            train_val_dataset = MedicalDataset(**opt.train_val, **opt.shared_param)
            train_size = int(len(train_val_dataset) * 0.8)
            # Split train and val
            # train_dataset, val_dataset = torch.utils.data.random_split(
            #     train_val_dataset, [train_size, len(train_val_dataset) - train_size]
            # )
            train_dataset = torch.utils.data.Subset(train_val_dataset, range(train_size))
            val_dataset = torch.utils.data.Subset(train_val_dataset, range(train_size, len(train_val_dataset)))
        else:
            train_dataset = None
            val_dataset = None
            warnings.warn("No train_val dataset.")

        return train_dataset, val_dataset, test_dataset

    else:
        raise NotImplementedError


def main(opt: MultiCADopt, device="cuda", mode="train", test_path=""):
    reproduc(**opt.reproduc)
    timestamp = datetime.now().strftime("_%Y_%m%d_%H%M%S_%f")
    opt.task_name += "_" + mode + timestamp
    proj_path = opj(opt.dir_name, opt.task_name)
    log = MyLogger(log_dir=proj_path, **opt.log)
    log.log_opt(opt)

    train_dataset, val_dataset, test_dataset = prepare_data(opt.data)

    if hasattr(opt, "gc_pred"):
        # Specify the number of features and dimensions if is set to auto
        if opt.gc_pred.dy_feat_num == "auto":
            print("Dy feat num = ", test_dataset.dy_feat_num)
            opt.gc_pred.dy_feat_num = test_dataset.dy_feat_num

        if opt.gc_pred.st_feat_num == "auto":
            print("St feat num = ", test_dataset.st_feat_num)
            opt.gc_pred.st_feat_num = test_dataset.st_feat_num

        if opt.gc_pred.dy_dim == "auto":
            print("Dy dim = ", test_dataset.dy_dim)
            opt.gc_pred.dy_dim = test_dataset.dy_dim

        if opt.gc_pred.st_dim == "auto":
            print("St dim = ", test_dataset.st_dim)
            opt.gc_pred.st_dim = test_dataset.st_dim

        if opt.gc_pred.t_length == "auto":
            print("T length = ", test_dataset.time_series_length)
            opt.gc_pred.t_length = test_dataset.time_series_length

        if opt.gc_pred.data_pred.pred_dim == "auto":
            print("Pred dim = ", test_dataset.pred_dim)
            opt.gc_pred.data_pred.pred_dim = test_dataset.pred_dim

        opt.gc_pred.data_pred.task_names = test_dataset.task_names

        if debug:
            opt.gc_pred.dataloader.num_workers = 0
            opt.gc_pred.batch_size = 8
            opt.gc_pred.total_epoch = 1
            delattr(opt.gc_pred.dataloader, "prefetch_factor")

        gcpred = Granger_Causal_Prediction(opt.gc_pred, log, device=device)

        if mode == "train":
            test_path = gcpred.train(train_dataset, val_dataset, test_dataset)
            gcpred.test(test_path, val_dataset, test_dataset)
        elif mode == "test":
            gcpred.test(test_path, val_dataset, test_dataset)
        else:
            raise NotImplementedError
    elif hasattr(opt, "compare"):
        if hasattr(opt.compare, "xgboost"):
            print("Testing XGBoost...")
            from exp.compare.xgb_pred import pred_xgb_multitask

            pred_xgb_multitask(train_dataset, val_dataset, test_dataset, opt.compare.xgboost, log)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Batch Compress")
    parser.add_argument(
        "-o",
        type=str,
        default=opj(opd(__file__), "exp/pla_exp/test_by_center/exp_by_center_data0728_xgb_4var.yaml"),
        help="yaml file path",
    )
    parser.add_argument("-g", help="availabel gpu list", default="1", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_path", type=str, default="", help="model path for testing")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    if args.g == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = "mps"
    elif args.g == "cpu":
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.g
        device = "cuda"

    debug = args.debug
    # debug = True
    # args.test = True
    # args.test_path = "outputs/mimic_train_2023_0831_004915_048332/iter_8/"

    main(
        OmegaConf.load(args.o),
        device=device,
        test_path=args.test_path,
        mode="test" if args.test else "train",
    )

    # import cProfile
    # cProfile.run("main(OmegaConf.load(args.o), device=device)", "outputs/perf_analyse")
