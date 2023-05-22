from enum import IntFlag
from itertools import product
from prettytable import PrettyTable
import sys
import os
from typing import List
from omegaconf import OmegaConf
import omegaconf.dictconfig
import omegaconf.listconfig
import time
import logging
import pynvml
import subprocess
from subprocess import PIPE, Popen
import psutil
import datetime
from urllib import request, parse

pynvml.nvmlInit()
platform = sys.platform
try:
    USER = os.environ['USER']
except:
    USER = ""
MIAOCODE = {'yrz': 't9qfTi1',
            'bbncyrz': 't9qfTi1',
            'xtx': 'tubrHSG',
            'cyx': '',
            'qjy': '',
            'yangrunzhao': ''}  # tuLGGSK


def reminding(miao_code):
    page = request.urlopen("http://miaotixing.com/trigger?" + parse.urlencode(
        {"id": miao_code, "templ": 'pmbXPGS,0,,,,,', "type": "json"}))
    page.read()


class Task:
    def __init__(self, command: str, name: str, gpucost: float, cpucost: float, cost_variable: str = 'none', status: str = 'pending'):
        self.command = command
        self.name = name
        self.gpucost = gpucost
        self.cpucost = cpucost
        self.cost_variable = cost_variable
        self.gpuusing = 0
        self.cpuusing = 0
        self.status = status
        self.ets = 0
        self.queue = None
        self.worker = None
        self.returncode = None

    def update(self,):
        self.worker.update()
        if self.worker.returncode == 0:
            self.status = 'finish'
            self.stop()
        elif self.worker.returncode == 1:
            self.status = 'error'
            self.ets += 1

    def query_gpuusing_cpuusing(self,):
        if self.worker is not None:
            self.worker.query_gpuusing_cpuusing()
            self.gpuusing = self.worker.gpuusing
            self.cpuusing = self.worker.cpuusing

    def stop(self,):
        if self.worker is not None:
            self.worker.stop()

    def details(self,):
        pass


class Worker():
    def __init__(self, gpu_list: List[int], task: Task):
        self.gpu_list = gpu_list
        self.gpuusing = 0
        self.cpuusing = 0
        self.task = task
        self.process = None
        self.pid = 0
        self.pids = ''
        self.create_time = 0
        self.children = []  # conclude parent
        self.command_refine = task.command + \
            ' -g {}'.format(','.join([str(i) for i in gpu_list]))

    def start(self, debug=False):
        print(self.command_refine)
        if debug:
            self.process = Popen(self.command_refine, shell=True)
        else:
            self.process = Popen(self.command_refine, shell=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,)
        self.pid = self.process.pid
        self.create_time = psutil.Process(self.process.pid).create_time()

    def update(self,):
        self.returncode = self.process.poll()

    def query_children(self,):
        if psutil.pid_exists(self.pid):
            self.pids = ''
            self.children = []
            proc = psutil.Process(self.pid)
            self.children = proc.children(recursive=True)
            self.children.append(proc)
            self.pids = ','.join([str(proc.pid) for proc in self.children])

    def stop(self,):
        if psutil.pid_exists(self.pid):
            if psutil.Process(self.pid).create_time() == self.create_time:
                self.query_children()
                for proc in self.children:
                    proc.terminate()
                logging.warning('Task:{} Terminate!'.format(self.task.name,))
                gone, alive = psutil.wait_procs(self.children, timeout=1)
                for p in alive:
                    p.kill()
                    logging.warning('Task:{} Kill!'.format(self.task.name,))

    def get_gpu_memory(self, pid: int):
        info = os.popen('nvidia-smi').read()
        # print(info)
        for line in info.split('\n'):
            t = line.strip().split()
            try:
                cur_pid = int(t[4])
                if cur_pid != pid:
                    continue
                gpu_mem = float(t[7][:-3])
                return gpu_mem
            except:
                continue
        return 0

    def query_gpuusing_cpuusing(self,):
        self.gpuusing = 0
        self.cpuusing = 0
        self.query_children()
        for proc in self.children:
            self.gpuusing += self.get_gpu_memory(proc.pid)  # MBytes
            if psutil.pid_exists(proc.pid):
                # MBytes
                self.cpuusing += psutil.Process(
                    proc.pid).memory_info().rss/2**20


class Queue():
    def __init__(self, task_list: List[Task], gpu_list: List[int] = [0], max_repend_task=20):
        self.max_repend_task = max_repend_task
        self.repend_counter = 0
        self.task_list = task_list
        self.gpu_list = gpu_list  # CUDA_DEVICE_ORDER="PCI_BUS_ID"
        self.gpufree = {}
        self.cpufree = 0
        self.task_count = len(task_list)
        self.pending_list: List[Task] = task_list
        self.running_list: List[Task] = []
        self.error_list: List[Task] = []
        self.finish_list: List[Task] = []
        self.sharecost_dict = {}

    def query_gpuusing_cpuusing(self,):
        for task in self.running_list:
            task.query_gpuusing_cpuusing()

    def init_sharecost_dict(self,):
        for task in self.task_list:
            if task.cost_variable != 'none':
                if task.cost_variable in self.sharecost_dict.keys():
                    self.sharecost_dict[task.cost_variable]['task_list'] += [task]
                else:
                    self.sharecost_dict[task.cost_variable] = {
                        'cpucost': 0, 'gpucost': 0, 'task_list': [task]}
                    # self.sharecost_dict[task.cost_variable]={'cpucost':0,'gpucost':0,'task_list':[task]}

    def refine_gpucost_cpucost(self):
        # cpu
        for cv in self.sharecost_dict:
            task_list: List[Task] = self.sharecost_dict[cv]['task_list']
            if len(set(task_list) & set(self.running_list)):
                # find the max cpu using
                max_cpuusing = max([task.cpuusing for task in task_list])
                # *1.2
                new_cpucost = max(
                    self.sharecost_dict[cv]['cpucost'], max_cpuusing*1.2)
                self.sharecost_dict[cv]['cpucost'] = new_cpucost
                # refine cpucost
                for task in task_list:
                    task.cpucost = new_cpucost
        # gpu
        if 'win' in platform:
            logging.warn('refine gpucost Not supported for Windows!')
        elif 'linux' in platform:
            for cv in self.sharecost_dict:
                task_list: List[Task] = self.sharecost_dict[cv]['task_list']
                if len(set(task_list) & set(self.running_list)):
                    # find the max gpu using
                    max_gpuusing = max([task.gpuusing for task in task_list])
                    # *1.2
                    new_gpucost = max(
                        self.sharecost_dict[cv]['gpucost'], max_gpuusing*1.2)
                    self.sharecost_dict[cv]['gpucost'] = new_gpucost
                    # refine gpucost cpucost
                    if new_gpucost > 100:
                        for task in task_list:
                            task.gpucost = new_gpucost
        else:
            raise NotImplementedError

    def refine_gpufree_cpufree(self,):
        """consider the minimum gpu free cpu free
        """
        # cpu
        self.cpufree = psutil.virtual_memory().available/2**20  # MBytes
        for task in self.running_list:
            self.cpufree -= max(task.cpucost-task.cpuusing, 0)
        # gpu
        for gpuidx in self.gpu_list:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(gpuidx)
                self.gpufree[gpuidx] = pynvml.nvmlDeviceGetMemoryInfo(
                    h).free/2**20  # MBytes
            except Exception as e:
                # print("WARNING: Gpu refine not supported!")
                self.gpufree[gpuidx] = 100000  # MBytes
                
        if 'win' in platform:
            logging.warn('refine gpufree Not supported for Windows!')
        elif 'linux' in platform:
            for task in self.running_list:
                for gpuidx in task.worker.gpu_list:
                    self.gpufree[gpuidx] -= max(task.gpucost-task.gpuusing, 0)

    def check_finish(self,):
        if len(self.finish_list) == self.task_count:
            return True

    def update_error_finish_list(self,):
        """update every task.status in running_list
        """
        temp_error_task_list = []
        temp_finish_task_list = []
        for task in self.running_list:
            task.update()
            if task.status == 'error':
                logging.warning('Task:{} error!'.format(task.name,))
                temp_error_task_list.append(task)
            elif task.status == 'finish':
                logging.warning('Task:{} finish!'.format(task.name,))
                temp_finish_task_list.append(task)
        # remove from running_list & append to error_list finish_list
        for task in temp_error_task_list:
            self.running_list.remove(task)
            self.error_list.append(task)
        for task in temp_finish_task_list:
            self.running_list.remove(task)
            self.finish_list.append(task)

    def repending_error_list(self,):
        temp_task_list = []
        for task in self.error_list:
            pass
            # logging.warning('Task:{} repending!'.format(task.name,))
            # temp_task_list.append(task)
        # remove from error_list & append to pending_list
        for task in temp_task_list:
            if self.repend_counter < self.max_repend_task:
                self.error_list.remove(task)
                self.pending_list.append(task)
                self.repend_counter += 1
            else:
                logging.warning("Max repend number achieved! Ignoring upcoming error tasks.")

    def run_pending_list(self, max_task=1e8, debug=False, autogpu=True):
        """assign workers to every task in pending_list according to gpu and cpu
        """
        temp_task_list = []
        for task in self.pending_list:
            if len(temp_task_list)+len(self.running_list) >= max_task:
                # logging.debug('Achieve Max task nums:{}!'.format(max_task))
                break
            if autogpu:
                gpufree_maxidx = max(self.gpufree, key=self.gpufree.get)
                if self.gpufree[gpufree_maxidx]*0.95 >= task.gpucost and self.cpufree*0.95 >= task.cpucost:
                    self.gpufree[gpufree_maxidx] -= task.gpucost
                    self.cpufree -= task.cpucost
                    # assign worker
                    task.worker = Worker([gpufree_maxidx], task)
                    task.worker.start(debug)
                    logging.warning('Task:{} Start! Using GPU:{}'.format(
                        task.name, gpufree_maxidx))
                    temp_task_list.append(task)
                    time.sleep(1)  # keep 5 sec to avoid crowded line
            else:
                # assign worker
                task.worker = Worker(self.gpu_list, task)
                task.worker.start(debug)
                logging.warning('Task:{} Start! Using GPU:{}'.format(
                    task.name, ','.join([str(i) for i in self.gpu_list])))
                temp_task_list.append(task)
                time.sleep(1)  # keep 5 sec to avoid crowded line
        # remove from pending_list & append to running_list
        for task in temp_task_list:
            self.running_list.append(task)
            self.pending_list.remove(task)

    def log(self,):
        queue_status_table = PrettyTable()
        queue_status_table.title = 'Queue Status'
        queue_status_table.add_column('pending', [len(self.pending_list)])
        queue_status_table.add_column('running', [len(self.running_list)])
        queue_status_table.add_column('error', [len(self.error_list)])
        queue_status_table.add_column('finish', [len(self.finish_list)])
        print(queue_status_table)
        running_table = PrettyTable()
        running_table.title = 'Running Task'
        running_table.field_names = [
            'Name', 'CraeteTime', 'gpu idx', 'gpu using/cost(MB)', 'cpu using/cost(MB)', 'pids', ]
        for task in self.running_list:
            running_table.add_row([
                task.name,
                datetime.datetime.fromtimestamp(
                    task.worker.create_time).strftime("%Y-%m-%d %H:%M:%S"),
                ','.join([str(i) for i in self.gpu_list]),
                '{:.0f}/{:.0f}'.format(task.gpuusing, task.gpucost),
                '{:.0f}/{:.0f}'.format(task.cpuusing,
                                       task.cpucost), task.worker.pids
            ])
        print(running_table)
        cost_table = PrettyTable()
        cost_table.title = 'cost variable'
        cost_table.field_names = ['cost variable',
                                  'gpu cost(MB)', 'cpu cost(MB)', ]
        for cv in self.sharecost_dict:
            cost_table.add_row([
                cv, '{:.0f}'.format(self.sharecost_dict[cv]['gpucost']),
                '{:.0f}'.format(self.sharecost_dict[cv]['cpucost']),
            ])
        print(cost_table)
        sysinfo = PrettyTable()
        sysinfo.title = 'system info'
        sysinfo.add_column('cpufree(MB)', ['{:.0f}'.format(self.cpufree)])
        for idx in self.gpufree:
            sysinfo.add_column('gpu{}_free(MB)'.format(
                idx), ['{:.0f}'.format(self.gpufree[idx])])
        print(sysinfo)

    def start(self, time_interval: float, max_task: int, debug: bool = False, log=True, remind=True, autogpu=True):
        
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_list[0])
        except Exception as e:
            print("WARNING: Gpu refine not supported. Select max task properly!")
        
        try:
            while True:
                self.query_gpuusing_cpuusing()
                self.refine_gpucost_cpucost()
                self.refine_gpufree_cpufree()
                self.run_pending_list(max_task, debug, autogpu)
                self.update_error_finish_list()
                self.repending_error_list()
                if self.check_finish():
                    time.sleep(2)
                    self.stop(remind)
                    break
                if log:
                    self.log()
                time.sleep(time_interval)
        except Exception as e:
            print(e)
            self.stop(remind)

    def stop(self, remind=True):  # TODO 根据用户名提醒不同的人
        for task in self.task_list:
            task.stop()
        if remind:
            try:
                miao_code = MIAOCODE[USER]
                reminding(miao_code)
            except:
                print("can't connect to miao_reminding or miao_code not exist")
