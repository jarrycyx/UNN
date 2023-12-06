
import numpy as np
import re


# 对查找表进行筛选，筛选标准可以是黑名单或者白名单
def sel_names(name_list, rules=[".*急性.*"], mode="black_list"):
    if mode == "black_list":
        sel_res = name_list
        for i, name in enumerate(name_list):
            exclude = np.max([(re.match(re.compile(black_match), str(name)) is not None)
                              for black_match in rules])
            exclude += (name is None)
            exclude += (len(str(name)) == 0)
            if exclude:
                sel_res.remove(name)
                print("Excluding: ", name)
    elif mode == "white_list":
        sel_res = []
        for i, name in enumerate(name_list):
            include = np.max([(re.match(re.compile(white_match), str(name)) is not None)
                              for white_match in rules])
            if include:
                sel_res.append(name)
                print("Including: ", name)
    else:
        raise NotImplementedError

    return sel_res
