import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops


import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import numpy as np
import torch
from einops import rearrange
import omegaconf
from omegaconf import OmegaConf
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from copy import deepcopy
import csv
import tqdm
import matplotlib
# matplotlib.use('qt4agg')
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
#定义自定义字体，文件名从1.b查看系统中文字体中来
myfont = FontProperties(fname='exp/FangZhengHeiTiJianTi-1.ttf')
#解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus']=False
# from utils.logger import MyLogger



def calc_mean_std(train_dataset, sample_num=300):
    all_dy = []
    print("Calculating feature mean and std for normalization...")
    for i in tqdm.trange(0, len(train_dataset), len(train_dataset) // sample_num):
        dynamic_data, _, _ = train_dataset[i]
        all_dy.append(dynamic_data[None])
    all_dy = torch.cat(all_dy, axis=0)
    b, t, n, d = all_dy.shape
    
    feat_mean, feat_std = [], []
    for n_i in range(n):
        feat_value = rearrange(all_dy[:, :, n_i, 0], "b t -> (b t)")
        feat_absence = rearrange(all_dy[:, :, n_i, 1], "b t -> (b t)")
        feat_value_valid = feat_value[feat_absence == 0]
        mean, std = torch.mean(feat_value_valid), torch.std(feat_value_valid)
        feat_mean.append(mean if not torch.isnan(mean) else 0)
        feat_std.append(std if std > 1e-3 else 1)

    return feat_mean, feat_std


def read_list_from_csv(item_sel_path, encoding="utf-8"):
    # item_sel_path = "data_prep/files/items_sel2.csv"
    f = open(item_sel_path, "r", encoding=encoding)
    reader = csv.reader(f)
    items_sel = []
    for i, row in enumerate(reader):
        # print(row)
        items_sel.append(row[0])
    return items_sel

def read_dict_from_csv(item_sel_path, encoding="utf-8"):
    # item_sel_path = "data_prep/files/items_sel2.csv"
    f = open(item_sel_path, "r", encoding=encoding)
    reader = csv.reader(f)
    items = {}
    for i, row in enumerate(reader):
        # print(row)
        if i > 0:
            items[str(row[0])] = row[1:]
    
    for key, val in items.items():
        items[key] = [float(v) for v in val if v.strip() != ""]
    return items

def eval_classification_performance(pred, label, name=""):
    
    if pred.shape[1] == 1:
        pred = torch.cat([1-pred, pred], dim=1)
        
    pred_logits = torch.argmax(pred, dim=-1)
    label_logits = torch.argmax(label, dim=-1)
    
    scores = {}
    figures = {}
    # results[name+"_accuracy"] = float(torch.mean(((pred_logits - label_logits)==0).float()).cpu().numpy())
    if label.shape[1] == 2: # 2 class
        try:
            scores[name+"_auc"] = roc_auc_score(label[:,1].numpy(), 
                                                pred[:,1].numpy())
            scores[name+"_auprc"] = average_precision_score(label[:,1].numpy(), 
                                                            pred[:,1].numpy())
            
            fpr, tpr, thres = roc_curve(label[:,1].numpy(), 
                                        pred[:,1].numpy(), pos_label=1)
            fig = plt.figure(figsize=[4, 4])
            plt.plot(fpr, tpr)
            figures[name+"_roc"] = fig
        except Exception as e:
            print("Evaluation error: ", e)
        # tp = torch.mean(((pred_logits==1) * (label_logits==1)).float()).detach().cpu().numpy()
        # tn = torch.mean(((pred_logits==0) * (label_logits==0)).float()).detach().cpu().numpy()
        # fp = torch.mean(((pred_logits==1) * (label_logits==0)).float()).detach().cpu().numpy()
        # fn = torch.mean(((pred_logits==0) * (label_logits==1)).float()).detach().cpu().numpy()
        # results[name+"_tp"] = tp
        # results[name+"_tn"] = tn
        # results[name+"_fp"] = fp
        # results[name+"_fn"] = fn
    
    return scores, figures


def score_dict_2_string(score: dict):
    txt = ""
    for key, val in score.items():
        txt += f"{key:s}: {val:.4f}\n"
    return txt

# batchnum x tasknum -> tasknum x batchnum
def merge_multi_task(data_list):
    data_all = [[] for _ in range(len(data_list[0]))]
    for data in data_list:
        for i,data_task in enumerate(data):
            data_all[i].append(data_task)
            
    return [torch.concat(data_all_task, axis=0) for data_all_task in data_all]
    
    
def eval_multitask_performance(preds, labels, names, take_samples=1e9):
    if take_samples < len(preds):
        choose_indices = np.random.choice(len(preds), int(take_samples), replace=False)
        preds = [pred[choose_indices] for pred in preds]
        labels = [label[choose_indices] for label in labels]
        print(f"Take {take_samples:d} samples for evaluation")
    
    tasks_preds = merge_multi_task(preds)
    tasks_labels = merge_multi_task(labels)
    
    all_scores = {}
    all_figures = {}
    for pred, label, name in tqdm.tqdm(zip(tasks_preds, tasks_labels, names)):
        # 如果所有位都是0，那么认为是无效样本
        valid_mask = np.where(torch.sum(label, dim=-1) != 0)
        label_valid = label[valid_mask]
        pred_valid = pred[valid_mask]
        
        scores, figures = eval_classification_performance(pred_valid, label_valid, name)
        all_scores.update(scores)
        all_figures.update(figures)
        
        print(f"Task {name:s} {score_dict_2_string(scores):s}, valid ratio: {len(valid_mask[0])/len(label):.4f}")
        
    return all_scores, all_figures, tasks_preds, tasks_labels
        


def normalize_data(data, axis=2):
    def slice_axis(data, i, axis):
        if axis == 1:
            return data[:,i]
        elif axis == 2:
            return data[:,:,i]
        else:
            raise NotImplementedError
        
    new_data = np.zeros_like(data, dtype=float)
    for i in range(data.shape[axis]):
        node = slice_axis(data, i, axis)
        node_std = np.nanstd(node)
        node_mean = np.nanmean(node)
        if node_std < 1e-5 or np.isnan(node_std):
            print(f"Axis {axis:d} is all nan")
            node_std = 1
        if np.isnan(node_mean):
            node_mean = 0
        node = (node - node_mean) / node_std
        
    return new_data
    

def calc_expo_param_update(start, end, step):
    return (end / start) ** (1 / step)


def log_time_series(original_data, data_interp, data_pred, log, log_step):
    fig = plt.figure(figsize=[10,10])
    plt.plot(np.arange(0, original_data.shape[0], 1), original_data, label="original")
    plt.plot(np.arange(0, data_interp.shape[0], 1), data_interp, label="interp")
    plt.plot(np.arange(0, data_pred.shape[0], 1), data_pred, label="pred")
    plt.legend()
    log.log_figures(fig, name="Predicted Latent Data", iters=log_step)

def calc_and_log_metrics(time_prob_mat, true_cm, log, log_step, threshold=0.5, plot_roc=False):
    causal_graph = (np.max(time_prob_mat, axis=2) > threshold)
    tp = np.mean(causal_graph * true_cm)
    tn = np.mean((1-causal_graph) * (1-causal_graph))
    fp = np.mean(causal_graph * (1-true_cm))
    fn = np.mean((1-causal_graph) * true_cm)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    log.log_metrics({"metrics/tpr": tpr}, log_step)
    log.log_metrics({"metrics/fpr": fpr}, log_step)
    log.log_metrics({"metrics/accuracy": acc}, log_step)

    if plot_roc:
        fpr, tpr, thres = roc_curve(true_cm.reshape(-1) > 0.5, 
                                    np.max(time_prob_mat, axis=2).reshape(-1), pos_label=1)
        fig = plt.figure(figsize=[4, 4])
        plt.plot(fpr, tpr)
        log.tblogger.add_figure(tag="ROC", figure=fig, global_step=log_step)
        
        log.log_npz(name="graph", data={"true_cm":true_cm, 
                                        "pred_cm":np.max(time_prob_mat, axis=2)})

    auc = roc_auc_score(true_cm.reshape(-1)>0.5,
                        np.max(time_prob_mat, axis=2).reshape(-1))
    log.log_metrics({"metrics/auc": auc}, log_step)
    return auc

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def plot_causal_matrix_in_training(time_coef, log, log_step, threshold=0.5, plot_each_time=True):
    n, m, t = time_coef.shape
    time_graph = sigmoid(time_coef)

    # Show Discovered Graph (Probability)
    sub_cg = plot_causal_matrix(
        np.max(time_graph, axis=2),
        figsize=[1.5*n, 1*n],
        vmin=0, vmax=1)
    log.log_figures(sub_cg, name="Discovered Prob.", iters=log_step)

    # Graph for Each Time Lag
    if plot_each_time:
        for ti in range(t):
            sub_cg = plot_causal_matrix(
                time_coef[:, :, ti],
                figsize=[1.5*n, 1*n],
                vmin=0, vmax=1)
            log.log_figures(sub_cg, name=f"Discovered Prob T-{t-ti:d}", 
                            iters=log_step, exclude_logger="tblogger")

    # Show Discovered Graph (Coefficiency)
    sub_cg = plot_causal_matrix(
        np.max(time_coef, axis=2),
        figsize=[1.5*time_coef.shape[0], 1*n])
    log.log_figures(sub_cg, name="Discovered Graph Coef", iters=log_step)

    # Show Thresholded Graph
    sub_cg = plot_causal_matrix(
        np.max(time_graph, axis=2) > threshold,
        figsize=[1.5*n, 1*n])
    log.log_figures(sub_cg, name="Discovered Graph", iters=log_step)
    
    
def feature_sel_txt(prob, class_names=None, sort=True):
    txt = "Item name, prob, Mean Value (Train), Mean Value (Valid), \n"
    num_classes = prob.shape[0]
    
    if len(prob.shape) == 1:
        prob = prob[None, :]
    
    if class_names is None or type(class_names) != list:
        class_names = [str(i+1) for i in range(num_classes)]
    if sort:
        sorted_index = np.argsort(prob[0],)[::-1]
        prob = prob[:, sorted_index]
        class_names = [class_names[i] for i in sorted_index]
    for i in range(prob.shape[1]):
        list_str = ", ".join([f"{x:.4f}" for x in prob[:, i]])
        txt += str(class_names[i]) + f", {list_str}, \n"
    return txt
    
    
def show_causal_edges_txt(prob_graph, thres_percentile=99, dy_item=None, name_lut=None):
    
    thres = np.percentile(prob_graph, thres_percentile)
    print(f"Causal Graph Threshold: {thres}, percentile: {thres_percentile}")
    highest_prob_loc = np.argwhere(prob_graph > thres)
    edges = []
    for loc_i in range(highest_prob_loc.shape[0]):
        parent, child = highest_prob_loc[loc_i]
        prob = prob_graph[parent, child]
        p_name, c_name = dy_item[parent], dy_item[child]
        edges.append((p_name, c_name, prob))
        
    edges = sorted(edges, key=lambda x: x[2], reverse=True)
        
    show_txt = ""
    for p_name , c_name, prob in edges:
        if name_lut is not None:
            if p_name in name_lut and c_name in name_lut:
                p_name = name_lut[p_name]
                c_name = name_lut[c_name]
        show_txt += f"{p_name} -> {c_name}: {prob:.4f}\n"
        
    return show_txt

def plot_feature_sel(prob, class_names=None, figsize=None, vmin=None, vmax=None, show_text=True, cmap="magma"):
    """
    A function to create a colored and labeled causal matrix matplotlib figure
    given true labels and preds.
    Args:
        cmtx (ndarray): causal matrix.
        num_classes (int): total number of nodes.
        class_names (Optional[list of strs]): a list of node names.
        figsize (Optional[float, float]): the figure size of the causal matrix.
            If None, default to [6.4, 4.8].

    Returns:
        img (figure): matplotlib figure.
    """
    num_classes = prob.shape[0]
    if len(prob.shape) == 1:
        prob = prob[None, :]
    if class_names is None or type(class_names) != list:
        class_names = [str(i+1) for i in range(num_classes)]
    
    class_names = [n[:15] for n in class_names]
    
    plt.clf()
    plt.close("all")
    figure = plt.figure(figsize=figsize)
    plt.imshow(prob, interpolation="nearest", aspect="auto",
               cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title("Causal matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, size=7)
    # plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = prob.max() / 2.0
    for i, j in itertools.product(range(prob.shape[0]), range(prob.shape[1])):
        color = "white" if prob[i, j] < threshold else "black"
        if show_text:
            plt.text(j, i, format(prob[i, j], ".2f") if prob[i, j] != 0 else ".",
                    horizontalalignment="center", color=color, fontproperties=myfont, size=7)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return figure


def plot_causal_matrix(cmtx, class_names=None, figsize=None, vmin=None, vmax=None, show_text=True, cmap="magma"):
    """
    A function to create a colored and labeled causal matrix matplotlib figure
    given true labels and preds.
    Args:
        cmtx (ndarray): causal matrix.
        num_classes (int): total number of nodes.
        class_names (Optional[list of strs]): a list of node names.
        figsize (Optional[float, float]): the figure size of the causal matrix.
            If None, default to [6.4, 4.8].

    Returns:
        img (figure): matplotlib figure.
    """
    num_classes = cmtx.shape[0]
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    
    figsize[0] = 30 if figsize[0] > 30 else figsize[0]
    figsize[1] = 20 if figsize[1] > 20 else figsize[1]
    
    plt.clf()
    plt.close("all")
    figure = plt.figure(figsize=figsize)
    plt.imshow(cmtx, interpolation="nearest",
               cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title("Causal matrix")
    plt.colorbar()
    # tick_marks = np.arange(len(class_names))
    # plt.xticks(tick_marks, class_names, rotation=45)
    # plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] < threshold else "black"
        if cmtx.shape[0] < 20 and show_text:
            plt.text(j, i, format(cmtx[i, j], ".2e") if cmtx[i, j] != 0 else ".",
                    horizontalalignment="center", color=color,)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return figure


def reproduc(seed, benchmark=False, deterministic=True):
    """Make experiments reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic


def omegaconf2list(opt, prefix='', sep='.'):
    notation_list = []
    for k, v in opt.items():
        k = str(k)
        if isinstance(v, omegaconf.listconfig.ListConfig):
            notation_list.append("{}{}={}".format(prefix, k, v))
            # if k in ['iter_list','step_list']: # do not sparse list
            #     dot_notation_list.append("{}{}={}".format(prefix,k,v))
            # else:
            #     templist = []
            #     for v_ in v:
            #         templist.append('{}{}={}'.format(prefix,k,v_))
            #     dot_notation_list.append(templist)
        elif isinstance(v, (float, str, int,)):
            notation_list.append("{}{}={}".format(prefix, k, v))
        elif v is None:
            notation_list.append("{}{}=~".format(prefix, k,))
        elif isinstance(v, omegaconf.dictconfig.DictConfig):
            nested_flat_list = omegaconf2list(v, prefix + k + sep, sep=sep)
            if nested_flat_list:
                notation_list.extend(nested_flat_list)
        else:
            raise NotImplementedError
    return notation_list


def omegaconf2dotlist(opt, prefix='',):
    return omegaconf2list(opt, prefix, sep='.')


def omegaconf2dict(opt, sep):
    notation_list = omegaconf2list(opt, sep=sep)
    dict = {notation.split('=', maxsplit=1)[0]: notation.split(
        '=', maxsplit=1)[1] for notation in notation_list}
    return dict






class LabelArray(object):
    # def __init__(self, array, labels):
    #     self.arr = array
    #     self.labels = labels
    #     self.marks = dim_marks
    #     assert [len[label_list] for label_list in labels] == self.arr.shape
        
    def __init__(self, dim, labels=None):
        if labels is not None:
            if len(dim) != dim:
                raise "The length of labels has to be equal to dim if defined"
            else:
                self.labels = deepcopy(labels)
        else:
            self.labels = [[] for _ in range(dim)]
        self.arr = None
        self.update_arr()
        
    def update_arr(self):
        if self.arr is not None:
            oldarr = self.arr
            self.arr = np.zeros([len(dim) for dim in self.labels]) * np.nan
            self.arr[tuple([slice(0,sh_dim,1) for sh_dim in oldarr.shape])] = oldarr
        else:
            self.arr = np.zeros([len(dim) for dim in self.labels]) * np.nan
        self.shape = self.arr.shape
        
        
    def __getitem__(self, label_list):
        index_list = []
        for dim,label in enumerate(label_list):
            if isinstance(label, str):
                index_list.append(self.labels[dim].index(label))
            elif isinstance(label, slice):
                index_list.append(label)
            elif isinstance(label, int):
                index_list.append(label)
            else:
                raise NotImplementedError
            
        return self.arr[tuple(index_list)]
    
    def __setitem__(self, label_list, val):    
        index_list = []
        for dim,label in enumerate(label_list):
            if isinstance(label, str):
                if not label in self.labels[dim]:
                    self.labels[dim].append(label)
                    self.update_arr()
                index_list.append(self.labels[dim].index(label))
            elif isinstance(label, slice):
                index_list.append(label)
            elif isinstance(label, int):
                index_list.append(label)
            else:
                raise NotImplementedError
            
        self.arr[tuple(index_list)] = val
    
    def __str__(self):
        return str(self.arr) + "\n--------------------------\n" + str(self.labels)
    
    
    def to_np(self):
        return self.arr
    
    def from_np(self, array):
        assert [len[label_list] for label_list in self.labels] == self.arr.shape
        self.arr = array
        
