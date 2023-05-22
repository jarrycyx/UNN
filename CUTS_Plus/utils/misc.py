import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import numpy as np
import torch
import omegaconf
from sklearn.metrics import roc_curve, roc_auc_score


def log_time_series(original_data, data_interp, data_pred, log, log_step):
    fig = plt.figure(figsize=[10,10])
    plt.plot(np.arange(0, original_data.shape[0], 1), original_data, label="original")
    plt.plot(np.arange(0, data_interp.shape[0], 1), data_interp, label="interp")
    plt.plot(np.arange(0, data_pred.shape[0], 1), data_pred, label="pred")
    plt.legend()
    log.log_figures(fig, name="Predicted Latent Data", iters=log_step)


def calc_and_log_metrics(time_prob_mat, true_cm, log, log_step, threshold=0.5, plot_roc=True):
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

def plot_causal_matrix_in_training(time_coef, name, log, log_step, threshold=0.5, plot_each_time=False):
    if time_coef is None:
        return
    
    if np.max(time_coef) - np.min(time_coef) > 0.01:
        time_coef = (time_coef - np.min(time_coef)) / (np.max(time_coef) - np.min(time_coef))
    n, m, t = time_coef.shape
    
    # # Show Discovered Graph (Coefficiency)
    # sub_cg = plot_causal_matrix(
    #     np.max(time_coef, axis=2),
    #     figsize=[1.5*time_coef.shape[0], 1*n])
    # log.log_figures(sub_cg, name="Discovered Graph Coef/" + name, iters=log_step)
    
    # # Graph for Each Time Lag
    # if plot_each_time:
    #     for ti in range(t):
    #         sub_cg = plot_causal_matrix(
    #             time_coef[:, :, ti],
    #             figsize=[1.5*n, 1*n],
    #             vmin=0, vmax=1)
    #         log.log_figures(sub_cg, name=f"Discovered Prob T-{t-ti:d}", 
    #                         iters=log_step, exclude_logger="tblogger")

    # Show Discovered Graph (Probability)
    time_graph = time_coef
    sub_cg = plot_causal_matrix(
        np.max(time_graph, axis=2),
        figsize=[1.5*n, 1*n],
        vmin=0, vmax=1)
    log.log_figures(sub_cg, name="Discovered Prob/" + name, iters=log_step)

    # Show Thresholded Graph
    time_thres = np.max(time_graph, axis=2) > threshold
    sub_cg = plot_causal_matrix(
        time_thres,
        figsize=[1.5*n, 1*n])
    log.log_figures(sub_cg, name="Discovered Graph/" + name, iters=log_step)
    log.log_npz({"Discovered Graph Coef": time_coef, "Discovered Prob": time_graph, "Discovered Graph": time_thres}, 
                name="Graph.npz", iters=log_step)


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


# def read_video(video_path: str):
#     if ops(video_path)[-1] == ".tif":
#         data = tifffile.imread(video_path)
#         data = (data / np.max(data) * 255).astype(np.uint8)
#         if len(data.shape) == 3:
#             data = data[:, :, :, None]
#         return data
#     else:
#         cap = cv2.VideoCapture(video_path)
#         frames = []
#         while cap.isOpened():
#             # get a frame
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frames.append(np.array(frame)[None])

#         cap.release()
#         return np.concatenate(frames, axis=0)


# def save_video(video_path: str, data):
#     skvideo.io.vwrite(video_path, data)



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
        
