
import numpy as np


# 存储病人数据的对象，包含一些工具函数、查找表
class PatientData(object):
    
    def __init__(self, cen_name, pid, vid, op_time):
        self.cen_name = cen_name
        self.pid = pid
        self.vid = vid
        self.op_time = op_time
        
        self.basic_info_items = None
        self.diagnosis_items = None
        self.dynamic_items = None
        
        self.data_dynamic = None 
        self.data_historic = None 
        self.data_after_endpoint = None 
        self.basic_info = None 
        self.diagnosis_data = None 
        self.outcomes = None
        self.prediction_endpoint_list = None
        
        self.predictions = []
        self.labels = []

        self.depreciated = False
        self.depreciated_reason = []

        
    def set_data(self,
                 basic_info_items,
                 diagnosis_items,
                 dynamic_items,
                 data_dynamic, 
                 data_historic, 
                 data_after_endpoint, 
                 basic_info, 
                 diagnosis_data, 
                 outcomes,
                 prediction_endpoint_list):
        
        
        self.basic_info_items = basic_info_items
        self.diagnosis_items = diagnosis_items
        self.dynamic_items = dynamic_items
        
        assert len(self.basic_info_items) > 0, "basic_info_items not loaded!"
        assert len(self.diagnosis_items) > 0, "diagnosis_items not loaded!"
        assert len(self.dynamic_items) > 0, "dynamic_items not loaded!"
        
        self.data_dynamic = data_dynamic 
        self.data_historic = data_historic 
        self.data_after_endpoint = data_after_endpoint 
        self.basic_info = basic_info 
        self.diagnosis_data = diagnosis_data 
        self.outcomes = outcomes
        self.prediction_endpoint_list = prediction_endpoint_list
        

    def depreciate_patient(self, reason=""):
        self.depreciated_reason.append(reason)
        if self.depreciated == False:
            self.depreciated = True


