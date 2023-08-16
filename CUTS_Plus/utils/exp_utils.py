import sys
import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.dirname(__file__))

from utils.opt_type import MultiCADopt
from utils.misc import omegaconf2dict

import re
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from omegaconf import OmegaConf
import glob
import csv, tqdm
from tensorboard.backend.event_processing import event_accumulator

PROPER_NAME = {"NFGR":"BRIEF", "h265":"H.265", "h264":"H.264", "jpg":"JPEG", "aoi-2000":"AoI", "vvc":"H.266"}

MAIN_COLOR = {"NFGR":"#e64b35", "h265":"#48c9b0", "h264":"#5599c7", "jpg":"#c39bd2", "DVC":"#e6b0aa", "SGA":"#f8c370", "vvc":"#b1babb",
              "GOOD":"#239954", "BAD":"#cc6155"}
SECONDARY_COLOR = {"NFGR":"#d98880", "h265":"#76d7c3", "SGA":"#f9d7a0", "h264":"#7fb3d5", "DVC":"#e6b0aa"}


def get_time_stamp():
    return str(datetime.now().strftime("%m-%d-%H%M%S-%f"))

def show_single_scores(x_arr, y_arr, label="exp", suffix="", 
                       scatter=False, log_axis=True, 
                       xlim=None, ylim=None):
    
    '''Plotting figures'''
    cvs = FigureCanvas(name=label, figsize=[30,20])
    fig_idx = 0
    for idx,data_path in enumerate(y_arr.labels[2]): # range(score_arr.shape[2]):
        if fig_idx == 40:
            break
        elif np.isnan(np.nanmean(y_arr[:,:,data_path])):
            continue
        
        fig_idx += 1
        if log_axis:
            ax = plt.subplot(8,5,fig_idx, xscale="log")
        else:
            ax = plt.subplot(8,5,fig_idx)
        ax.set_title("DATA_{:02d}_".format(idx) + data_path[-100:-60] + "\n" + data_path[-60:])
        # ax.set_title(dim_marks[2][data_i])
        plt.set_cmap("rainbow")
        for i, dim0 in enumerate(y_arr.labels[0]):
            x_nan = x_arr[dim0,:,data_path]
            y_nan = y_arr[dim0,:,data_path]
            x = x_nan[np.isfinite(x_nan + y_nan)]
            y = y_nan[np.isfinite(x_nan + y_nan)]
            
            if len(x) > 0 and len(y) > 0:
                x, y = sort_lists(x, y)
                
                plt.plot(x, y, color=plt.get_cmap("tab20")(i), label=dim0)
                plt.scatter(x, y, color=plt.get_cmap("tab20")(i))
            
        plt.legend()
        if ylim is not None:
            plt.ylim(ylim)
    
    ax=plt.gca()
    ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            
    cvs.save_fig(suffix=suffix, time_stamp=False, save_format=".pdf")


def sort_lists(*lists):
    sorted_index = np.argsort(lists[0]).astype(int)
    results = []
    for l in lists:
        sorted_list = np.array([l[i] for i in sorted_index])
        results.append(sorted_list)
    return results


def show_averge_scores(x_arr, y_arr, label="exp", suffix="", percentile=25,
                       scatter=False, log_axis=True, figsize=[4,3],
                       xlim=None, ylim=None, legend=False, grid=False, std=False):
    cvs = FigureCanvas(name=label, figsize=figsize)
    if log_axis:
        plt.xscale("log")
        
    fig_idx = 0
    if scatter:
        for data_i,data_path in enumerate(y_arr.labels[2]):
            if np.isnan(np.nanmean(y_arr[:,:,data_path])):
                continue
            
            for i, dim0 in enumerate(y_arr.labels[0]):
                x_nan = x_arr[dim0,:,data_path]
                y_nan = y_arr[dim0,:,data_path]
                x = x_nan[np.isfinite(x_nan + y_nan)]
                y = y_nan[np.isfinite(x_nan + y_nan)]
                
                if len(x) > 0 and len(y) > 0:
                    x, y = sort_lists(x, y)
                    
                    # plt.plot(x, y, color=plt.get_cmap("tab20")(i), label=dim0, marker="v")
                    plt.scatter(x, y, color=MAIN_COLOR[dim0], alpha=0.5, marker="v", edgecolors='none',
                                s=80 if "NFGR" in dim0 else 50)
                
            # plt.legend(edgecolors='none')
            if ylim is not None:
                plt.ylim(ylim)
                
    if ylim is not None:
        full_range = ylim[1] - ylim[0]  
    else:
        full_range = np.nanmax(np.nanmean(y_arr.arr, axis=2)) - np.nanmin(np.nanmean(y_arr.arr, axis=2))
    max_std = np.nanmax(np.nanstd(y_arr["NFGR",:,:], axis=1))
            
        
    for idx, dim0 in enumerate(sorted(y_arr.labels[0], key=lambda item:item == "NFGR")):
        # if "jpg" in dim0:
        #     continue
        
        x_nan = np.nanmean(x_arr[dim0,:], axis=1)
        y_nan = np.nanmean(y_arr[dim0,:], axis=1)
        # y_l = np.nanpercentile(y_arr[dim0,:], percentile, axis=1)
        # y_u = np.nanpercentile(y_arr[dim0,:], 100-percentile, axis=1)
        y_std = np.nanstd(y_arr[dim0,:], axis=1) # / max_std * full_range * 0.07
        
        print(1 / max_std * full_range * 0.07)
        
        x = x_nan[np.isfinite(x_nan + y_nan)]
        y = y_nan[np.isfinite(x_nan + y_nan)]
        # y_l = y_l[np.isfinite(x_nan + y_nan)]
        # y_u = y_u[np.isfinite(x_nan + y_nan)]
        y_std = y_std[np.isfinite(x_nan + y_nan)]
        
        x, y, y_std = sort_lists(x, y, y_std)
        
        if std:
            plt.fill_between(x, y-y_std/2, y+y_std/2, color=MAIN_COLOR[dim0], alpha=0.25, edgecolors="none")
        plt.plot(x, y, color=MAIN_COLOR[dim0], label=name(dim0), 
                 lw=2 if "NFGR" in dim0 else 1.5)
        plt.scatter(x, y, color=MAIN_COLOR[dim0], 
                 s=30 if "NFGR" in dim0 else 20)
        
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    if legend:
        plt.legend(loc='lower left', bbox_to_anchor=(0.1, 0.1), fancybox=False)
    
    ax=plt.gca()
    if log_axis:
        ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=5))
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    
    if grid:
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        # plt.xticks(np.arange(0.4, 1.8, 0.28))
        # plt.yticks(np.arange(50, 500, 40))
        plt.grid(axis='both', c="#cacaca", which="major")
    
    cvs.save_fig(suffix=suffix, time_stamp=False, save_format=".pdf")
    
    
def name(alias):
    if alias in PROPER_NAME:
        return PROPER_NAME[alias]
    else:
        print("Cannot find proper name.")
        return alias

class FigureCanvas(object):
    
    def __init__(self, name="ex1", figsize=[14,9]):
        self.name = name
        plt.close('all')
        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
    
    def show_fig(self, save_format=".png", suffix="", time_stamp=True):
        save_path = "./exp/figs/%s/%s_%s%s"%(
            self.name, 
            get_time_stamp() if time_stamp else "plt", 
            suffix, save_format)
        if not os.path.exists(opd(save_path)):
            os.makedirs(opd(save_path))
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        
    def save_fig(self, save_format=".png", suffix="", time_stamp=True, save_root="./exp/figs/"):
        save_path = opj(save_root, "%s/%s_%s%s"%(
            self.name, 
            get_time_stamp() if time_stamp else "plt", 
            suffix, save_format))
        if not os.path.exists(opd(save_path)):
            os.makedirs(opd(save_path))
        plt.savefig(save_path, bbox_inches='tight')


def find_lineprofile_cmp(im_list):    
    for x in range(0, im_list[0].shape[0], 10):
        for y in range(0, im_list[0].shape[1], 10):
            lp = [im[x][y] for im in im_list]
            if np.max(lp) > 2500 and np.max(lp) < 3000:
                return lp


def get_decompressed_path(opt_path):
    res_root = opd(opt_path)
    max_step = 0
    for dirn in os.listdir(res_root):
        if dirn == "decompressed":
            return glob.glob(res_root + "/decompressed/*.tif")[0]
        elif "steps" in dirn:
            step_n = int(dirn[5:])
            if step_n > max_step:
                max_step = step_n
    search_list = glob.glob(res_root + "/steps" + str(max_step) + "/decompressed/*.*")
    if len(search_list) > 0:
        return search_list[0]
    return None

def load_scalars(event_path):
    try:
        event_path = glob.glob(event_path)[0]
    except:
        print("No event file found.")
        return None
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    # print("Available scalars: ", ea.scalars.Keys())
    scalars = {}
    for criterion in ea.scalars.Keys():
        val_scalar = ea.scalars.Items(criterion)
        val_curve = ([(i.step, i.value) for i in val_scalar])
        scalars[criterion] = val_curve
    return scalars
    
def load_scalars_cached(root_path, cache_dir="exp/cache", reload_data=False):
    
    if root_path[-1] == "/":
        root_path = root_path[:-1]
    
    # csv_file = glob.glob(root_path + "/*.csv")[0]
    # exp_list = load_csv(csv_file)
    root_name = "".join(re.split("/|\\\\", root_path)[-2:])
    cache_path = opj(cache_dir, root_name + ".npy")
    
    # read from cached data
    if os.path.exists(cache_path) and not reload_data:
        print("Loading cached result...")
        loaded_res = np.load(cache_path, allow_pickle=True)
    else:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        loaded_res = []
        for dirn in tqdm.tqdm(os.listdir(root_path)):
            if os.path.isdir(opj(root_path, dirn)):
                if len(glob.glob(root_path + "/%s/events.out.tfevents*"%(dirn))) > 0:
                    res_fname = glob.glob(root_path + "/%s/events.out.tfevents*"%(dirn))[0]
                    opt_fname = glob.glob(root_path + "/%s/opt.yaml"%(dirn))[0]
                    scores = load_scalars(res_fname)
                    opt: MultiCADopt = omegaconf2dict(OmegaConf.load(opt_fname), sep=".")
                    loaded_res.append((res_fname, scores, opt))
        
        np.save(cache_path, loaded_res)
    
    return loaded_res


if __name__=="__main__":
    # scalars = load_scalars("cyx_exp/experiments_outputs/ex2_1227/*/exp_00000/events.out.tfevents*")
    # print(scalars)
    
    res = load_scalars_cached("cyx_exp/experiments_outputs/ex2_1227/", reload_data=False)
    print("")