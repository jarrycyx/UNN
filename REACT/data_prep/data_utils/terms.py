import numpy as np


# define lower cases here
terms = [
    ["creatinine", "肌酐"],
    ["urea", "尿素"],
    ["urea nitrogen", "尿素氮"],
    ["uric acid", "尿酸"],
    ["u/l", "ul", "iu/l"],
    ["umol/l", "umoll", "umol l", "μmol/l"],
    ["mmol/l", "mmoll", "mmol l"],
    ["mg/dl", "mgdl", "mg dl"],
]


"""
Ignore lower and upper case
"""
def compare_name(a, b):
    if a.lower() == b.lower():
        return True
    for term in terms:
        if a.lower() in term and b.lower() in term:
            return True
    return False