import sys
from omegaconf import OmegaConf
import os
from os.path import join as opj
import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname as opd
from typing import Dict, Union
import time
from torch.utils.tensorboard import SummaryWriter
from utils.misc import omegaconf2dict, omegaconf2list


class MyLogger():
    def __init__(self, log_dir: str, stderr: bool = True, tensorboard: bool = True, stdout: bool = True):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.logger_dict: Dict[str] = {}
        if stdout:
            stdout_handler = open(opj(self.log_dir, 'stdout.log'), 'w')
            sys.stdout = stdout_handler
        if stderr:
            stderr_handler = open(opj(self.log_dir, 'stderr.log'), 'w')
            sys.stderr = stderr_handler
        if tensorboard:
            self.tblogger = SummaryWriter(self.log_dir)
            self.logger_dict['tblogger'] = self.tblogger

    def log_opt(self, opt):
        OmegaConf.save(config=opt, f=opj(self.log_dir, 'opt.yaml'))
        opt_log = omegaconf2list(opt, sep='/')
        for logger_name in self.logger_dict.keys():
            if logger_name == 'tblogger':
                for idx, opt in enumerate(opt_log):
                    self.logger_dict[logger_name].add_text('hparam', opt, idx)

    def log_metrics(self, metrics_dict: Dict[str, float], iters):
        for logger_name in self.logger_dict.keys():
            if logger_name == 'csvlogger':
                self.logger_dict[logger_name].log_metrics(metrics_dict, iters)
                self.logger_dict[logger_name].save()
            elif logger_name == 'clearml_logger':
                for k in metrics_dict.keys():
                    self.logger_dict[logger_name].report_scalar(
                        k, k, metrics_dict[k], iters)
            elif logger_name == 'tblogger':
                for k in metrics_dict.keys():
                    self.logger_dict[logger_name].add_scalar(
                        k, metrics_dict[k], iters)
    
    def log_figures(self, figure, name="figure.png", iters=None, exclude_logger=[]):
        for logger_name in self.logger_dict.keys():
            if logger_name == 'tblogger':
                if logger_name not in exclude_logger:
                    self.logger_dict[logger_name].add_figure(tag=name, figure=figure, global_step=iters)
        
        if iters is None:
            save_path = opj(self.log_dir, "figures")
        else:
            save_path = opj(self.log_dir, f"iter_{iters:d}", name)
        os.makedirs(opd(save_path), exist_ok=True)
        figure.savefig(save_path)
        
    def log_npz(self, data: dict, name="data.npz", iters=None):
        if iters is None:
            save_path = opj(self.log_dir)
        else:
            save_path = opj(self.log_dir, f"iter_{iters:d}", name)
        os.makedirs(opd(save_path), exist_ok=True)
        np.savez(save_path, **data)

    def close(self):
        for logger_name in self.logger_dict.keys():
            if logger_name == 'tblogger':
                self.logger_dict[logger_name].close()
