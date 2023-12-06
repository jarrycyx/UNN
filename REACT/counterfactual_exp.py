import os, sys
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops

if "exp" in os.getcwd():
    os.chdir("../../../")

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression as LR

from medical_pred_main import Granger_Causal_Prediction, prepare_data, main
from nets.loss import Causal_Loss
from utils.misc import (
    reproduc,
    read_dict_from_csv,
)
from utils.opt_type import MultiCADopt
from utils.logger import MyLogger


class Conterfactual_Prediction(Granger_Causal_Prediction):
    def __init__(self, args, log, device="cuda"):
        super().__init__(args, log, device)
        
    def load_model(self, load_dir, dy_items, st_items):
        
        model_dir = opj(load_dir, "model.pt")
        self.fitting_model.load_state_dict(torch.load(model_dir))

        # Data Prediction Testing
        self.dy_items = dy_items
        self.st_items = st_items
        self.all_items = dy_items + st_items

        dy_graph_dict = read_dict_from_csv(opj(load_dir, "dy_sel.csv"))
        st_graph_dict = read_dict_from_csv(opj(load_dir, "st_sel.csv"))
        # print(dy_items, st_items, dy_graph_dict, st_graph_dict)
        dy_graph = torch.tensor([float(dy_graph_dict[dy_item][0]) for dy_item in dy_items]).to(self.graph.device)
        st_graph = torch.tensor([float(st_graph_dict[st_item][0]) for st_item in st_items]).to(self.graph.device)

        thres = 0.5
        self.graph = (torch.cat([dy_graph, st_graph], dim=0) > thres).float()  # binarize
        self.graph = self.graph * 200 - 100  # scale to -100 to 100
        self.args.data_pred.hard_mask = True  # hard thresholding instead of bernonlli sampling
        self.epoch_i = 0
        
        self.valid_feat_name = [dy_item for dy_item in dy_items if float(dy_graph_dict[dy_item][0]) > thres] + \
            [st_item for st_item in st_items if float(st_graph_dict[st_item][0]) > thres]
            
    def pred_single_patient(self, dynamic_data, static_data, label):
        # 如果数据不在gpu上或者是numpy，则转换成gpu tensor
        if not isinstance(dynamic_data, torch.Tensor):
            dynamic_data = torch.tensor(dynamic_data)
            static_data = torch.tensor(static_data)
        dynamic_data = dynamic_data.to(self.device)[None]
        static_data = static_data.to(self.device)[None]
        
        if isinstance(label, dict):
            label_batch = [label]
            # dict的合并，是将每个dict对应key的item分别合并
            label_dict = []
            for key in label_batch[0].keys():
                label_dict += [torch.concat([label[key][None] for label in label_batch], axis=0),]
            label = label_dict
        
        # 预测单个样本
        pred_batch, loss, _ = self.data_pred(dynamic_data, static_data, label, mode="test")
        return [p.detach().cpu() for p in pred_batch]

    def calc_causal_effect(self, dynamic_data_list, static_data_list, label_list, task_of_interest, save_path="outputs/counterfactual_exp.csv"):
        # 测试一系列病人，计算每个变量的因果效应
        
        if isinstance(task_of_interest, str):
            task_i = self.args.data_pred.task_names.index(task_of_interest)
        
        if len(self.valid_feat_name) == 0:
            return None 
        
        # 通过计算平均值作为计算因果效应的参考值
        all_dy = torch.cat(dynamic_data_list, dim=0)[:, :, :, 0].clone()
        all_dy[all_dy == 0] = torch.nan
        all_st = torch.cat(static_data_list, dim=0)[:, :, 0].clone()
        all_st[all_st == 0] = torch.nan
        dy_ref = torch.nanmean(torch.nanmean(all_dy, dim=1), dim=0)
        st_ref = torch.nanmean(all_st, dim=0)
        ref = torch.cat([dy_ref, st_ref], dim=0)
            
        test_res = []
        print("Valid features: ", self.valid_feat_name)
        for feat_name in (self.valid_feat_name + ["None"]):
            if feat_name != "None":
                feat_index = self.all_items.index(feat_name)
                self.graph[feat_index] = -100
            else:
                feat_index = 0
            
            # print(f"Testing {feat_name} {feat_index}...")
            pred_epoch = []
            label_epoch = []
            avg_feat_epoch = []
            for i, (dynamic_data, static_data, label) in enumerate(zip(dynamic_data_list, static_data_list, label_list)):
                pred_batch, loss, _ = self.data_pred(dynamic_data, static_data, label, mode="test", ref="zero")
                pred_epoch.append(pred_batch[task_i].detach().cpu())
                label_epoch.append(label[task_i].detach().cpu())
                
                dynamic_data = dynamic_data.cpu().clone()
                dynamic_data[dynamic_data == 0] = torch.nan
                avg_dy = torch.nanmean(dynamic_data[:, :, :, 0], dim=[1])
                avg_dy[torch.isnan(avg_dy)] = dy_ref.cpu()[None].expand(avg_dy.shape[0], -1)[torch.isnan(avg_dy)]
                
                static_data = static_data.cpu().clone()
                static_data[static_data == 0] = torch.nan
                avg_st = static_data[:, :, 0]
                avg_st[torch.isnan(avg_st)] = st_ref.cpu()[None].expand(avg_st.shape[0], -1)[torch.isnan(avg_st)]
                
                avg_feat_epoch.append(torch.cat([avg_dy, avg_st], axis=1))
                torch.cuda.empty_cache()
                # break # Debug
            pred_epoch = torch.cat(pred_epoch, dim=0)
            pred_epoch = np.exp(pred_epoch[:,1])/(np.exp(pred_epoch[:,0]) + np.exp(pred_epoch[:,1]))
            avg_feat_epoch = torch.cat(avg_feat_epoch, dim=0)
            # pred_epoch = pred_epoch[:,1]
            label_epoch = torch.cat(label_epoch, dim=0)[:,1]
            
            test_res.append((feat_name, pred_epoch, label_epoch, avg_feat_epoch[:, feat_index]))
            self.graph[feat_index] = 100

        # 计算的因果效应=某个指标变化100%平均值，对预测结果的影响
        ref = ref.cpu().numpy()
        all_feat_name = [feat_name for feat_name,_,_,_ in test_res]
        test_df = pd.DataFrame({**{feat_name+"_input": avg_feat for feat_name, pred, label, avg_feat in test_res},
                                **{feat_name+"_output": pred for feat_name, pred, label, avg_feat in test_res},
                                **{"Label": test_res[0][2]}})
        
        # print(ref)
        for column in all_feat_name:
            # print("Calculating ", column)
            if column != "Label" and column != "None":
                test_df_new = pd.DataFrame({**{column+"_ref": ref[self.all_items.index(column)]},
                                            **{column+"_pert": (test_df[column+"_input"] - ref[self.all_items.index(column)]) / ref[self.all_items.index(column)]},
                                            **{column+"_ce": (test_df["None_output"] - test_df[column+"_output"])}})
                test_df = pd.concat([test_df, test_df_new], axis=1)
                
        test_df.sort_index(inplace=True, axis=1)
        test_df.to_csv(save_path)
        return test_df


def prepare_counterfact_model(opt: MultiCADopt, device="cuda", mode="train"):
    reproduc(**opt.reproduc)
    # timestamp = datetime.now().strftime("_%Y_%m%d_%H%M%S_%f")
    opt.task_name += "_" + mode + "feat_imp"
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

    # cf_pred = Conterfactual_Prediction(opt.gc_pred, log, device=device)
    return opt.gc_pred, log, test_dataset
    


def select_data(dataset, task_of_interest, cf_pred, max_sample=1000, select_mode="none", shuffle=True):

    if isinstance(task_of_interest, str):
        task_i = cf_pred.args.data_pred.task_names.index(task_of_interest)

    test_loader = DataLoader(
        dataset,
        batch_size=1000,
        shuffle=shuffle,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,
    )
    
    # 从数据中选出max_sample个正样本
    sel_patient = 0
    dynamic_data_list = []
    static_data_list = []
    label_list = []
    for i, (dynamic_data, static_data, label) in enumerate(test_loader):
        if sel_patient >= max_sample:
            break
        
        if select_mode == "pos":
            sel_index = torch.argwhere(label[task_i][:,1] == 1)[:,0]
        elif select_mode == "neg":
            sel_index = torch.argwhere(label[task_i][:,1] == 0)[:,0]
        elif select_mode == "none":
            sel_index = torch.arange(label[task_i].shape[0])
        else:
            raise NotImplementedError    
        
        print(f"Sel patient ({select_mode}): {sel_index.shape[0]} / {label[task_i].shape[0]}")
        dynamic_data_list.append(dynamic_data[sel_index[:max_sample-sel_patient]])
        static_data_list.append(static_data[sel_index[:max_sample-sel_patient]])
        label_list.append([label_task[sel_index[:max_sample-sel_patient]] for label_task in label])
        sel_patient += dynamic_data_list[-1].shape[0]

    data = (dynamic_data_list, static_data_list, label_list)
    return data