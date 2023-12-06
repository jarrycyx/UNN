import os, sys
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops

sys.path.append(opj(opd(__file__), "../../"))

import numpy as np
import time
from data_prep.data_utils.pylogging import l


decoder_str_list = {
    ("%Y/%m/%d %H:%M:%S", 0),
    ("%Y-%m-%d %H:%M:%S", 0),
    ("%Y.%m.%d %H:%M", 0),
    ("%Y/%m/%d %H:%M", 0),
    ("%Y/%m/%d %H.%M", 0),
    ("%Y.%m.%d %H/%M", 0),
    ("%Y.%m.%d %H:%M", 0),
    ("%Y.%m.%d %H.%M", 0),
    ("%Y.%m.%d %H%M", 0),
    ("%Y/%m/%d", 12*3600),
    ("%Y.%m.%d", 12*3600),
}



def decode_time(time_str):
    time_str = str(time_str)
    time_str = time_str.strip()
    # print(f"Try to decode: [{time_str:s}]")
    e = None
    time_stamp = np.nan
    if time_str == "":
        return time_stamp
    else:
        for decoder_str, offset in decoder_str_list:
            # print(f"Try decoder: {decoder_str:s}")
            try:
                time_stamp = time.mktime(time.strptime(time_str, decoder_str))
                time_stamp += offset
                return time_stamp
            except Exception as err:
                e = err
    
    l().error(str(e))
    l().error("Cannot decode: " + time_str)
    # raise NotImplementedError
            
    return time_stamp


def time2str(time_data):
    if time_data is None or np.isnan(time_data):
        return "None"
    elif isinstance(time_data, str):
        time_stamp = decode_time(time_data)
    elif isinstance(time_data, float) or isinstance(time_data, int):
        time_stamp = time_data
    else:
        raise NotImplementedError
    
    return time.strftime("%Y%m%d_%H:%M:%S", time.localtime(time_stamp))


if __name__ == "__main__":
    print(decode_time("2023.09.16 07:50 "))