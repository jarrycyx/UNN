import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops

import logging
import logging.config
import time

# https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
    
logger_dict = {"alt": None, "main": None}

log_path = time.strftime("data_dir/log/Medset2_ver%Y%m%d.log")

logger_dict["main"] = logging.getLogger()



# Async Logger 用来在多进程中存储日志，在进程结束之后，将日志写入主logger
# 这样做主要的好处是可以把多个进程的日志写入同一个文件
# 但是实测这样有点奇怪，因为不能实时写入，而且使用者很难理解这个逻辑
# 所以建议若非必要情况，都使用分文件写入的方式，即每个进程里面单独调用 logger_init
class AsyncLogger(object):
    def __init__(self, subprocess_name):
        self.name = subprocess_name
        self.msgs = []
        
    def debug(self, s):
        self.msgs.append((self.name + " - " + str(s), 0))
        
    def info(self, s):
        self.msgs.append((self.name + " - " + str(s), 1))
        
    def warning(self, s):
        self.msgs.append((self.name + " - " + str(s), 2))
        
    def error(self, s):
        self.msgs.append((self.name + " - " + str(s), 3))
        
    def fatal(self, s):
        self.msgs.append((self.name + " - " + str(s), 4))
        
    def log_all(self, l):
        for msg, level in self.msgs:
            if level == 0:
                l.debug(msg)
            elif level == 1:
                l.info(msg)
            elif level == 2:
                l.warning(msg)
            elif level == 3:
                l.error(msg)
            elif level == 4:
                l.fatal(msg)


def l():
    return logger_dict["main"]

def set_async_logger(sl):
    global logger_dict
    # 主logger替换为async logger
    logger_dict["alt"] = logger_dict["main"]
    logger_dict["main"] = sl
    
def reset_logger():
    global logger_dict
    
    logger_dict["main"].log_all(logger_dict["alt"])
    
    ml = logger_dict["main"]
    logger_dict["main"] = logger_dict["alt"]
    logger_dict["alt"] = ml


# conf_log = opj(opd(__file__), "pylogging_config.ini")
# logging.config.fileConfig(conf_log)
# pylog = logging.getLogger("main")

def logger_init(fname=None):
    global log_path
    if fname is None:
        fname = log_path
    # print("Logger init: ", fname)
    try:
        if not os.path.exists(opd(fname)):
            os.makedirs(opd(fname))
        if os.path.exists(fname):
            os.remove(fname)
    except:
        print("Logger init failed: ", fname)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='[%Y-%m_%d %H:%M:%S]',
                        filename=fname,
                        filemode='a')
    logger_dict["main"] = logging.getLogger()

# logger_init(log_path)