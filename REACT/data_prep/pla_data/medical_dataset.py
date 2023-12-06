
import numpy as np
import torch
import time
from torch.utils.data import Dataset
import os
from scipy import interpolate

from data_prep.pla_data.patient_data import PatientData
from data_prep.data_utils.csv_loader import read_csv
from data_prep.data_utils.item_sel import sel_names

from data_prep.data_utils.pylogging import l


dir_name = time.strftime("Medset2_ver%Y%m%d.log")


"""
预先从数据库中读取数据，然后将数据保存到本地，以便后续使用
对于pla数据来说，内存足以保存全部数据
"""


def init_pool_processes(the_lock):
    """Initialize each process with a global variable lock."""
    global lock
    lock = the_lock


class MedicalData(object):
    def __init__(
        self,
        db_path,
        dy_sel,
        st_sel,
        data_cache_path,
        include_center=None,
        use_historic=True,
        use_after_endpoint=True,
        use_basic_info=True,
        max_patient=None,
        timeseries_endpoint="every_ending",
        time_interval=[None, None],
        time_series_length=14 * 12,
        time_series_resolution=2 * 3600,
        prediction_window=[6, 12, 24, 48],
    ):
        self.data_cache_path = data_cache_path
        self.use_historic = use_historic
        self.use_after_endpoint = use_after_endpoint
        self.use_basic_info = use_basic_info
        self.max_patient = max_patient
        self.db_path = db_path
        self.include_center = include_center
        self.dy_sel = dy_sel
        self.st_sel = st_sel
        self.time_series_length = time_series_length
        self.time_series_resolution = time_series_resolution
        self.timeseries_endpoint = timeseries_endpoint
        self.prediction_window = prediction_window

        self.time_interval = [0, time.mktime(time.strptime("30000101", "%Y%m%d"))]

        if time_interval[0] != "none":
            self.time_interval[0] = time.mktime(time.strptime(str(time_interval[0]), "%Y%m%d"))
        if time_interval[1] != "none":
            self.time_interval[1] = time.mktime(time.strptime(str(time_interval[1]), "%Y%m%d"))


    def read_patients_data_from_cache(self):
        self.patient_data_list = torch.load(self.data_cache_path)

        patient_data: PatientData = self.patient_data_list[0]
        self.dynamic_items = patient_data.dynamic_items
        self.diagnosis_items = patient_data.diagnosis_items
        self.basic_info_items = patient_data.basic_info_items


    def setup_db(self):
        # 建立所有检验指标、出院诊断的查找表
        dynamic_items = read_csv("data_prep/pla_data/item_sel/labtest_items.csv")
        diagnosis_items = read_csv("data_prep/pla_data/item_sel/diagnosis_items.csv")
        basic_info_items = read_csv("data_prep/pla_data/item_sel/basic_info_items.csv")

        l().info(f"Dynamic item num: {len(dynamic_items):d}, Diagnosis item num: {len(diagnosis_items):d}, Basic info item num: {len(basic_info_items):d}")

        # 根据预先定义的规则筛选检验指标、出院诊断
        self.dynamic_items = sel_names(dynamic_items, **self.dy_sel)
        self.diagnosis_items = sel_names(diagnosis_items, **self.st_sel)
        self.basic_info_items = sel_names(basic_info_items, **self.st_sel)



class MedicalDataset(Dataset, MedicalData):
    def __init__(
        self,
        db_path,
        include_center,
        dy_sel,
        st_sel,
        pred_tasks,
        use_data_cache=True,
        data_cache_path="data_dir/cache/medical_data.pt",
        use_historic=True,
        use_after_endpoint=True,
        use_basic_info=True,
        max_patient=None,
        multiprocess=True,
        timeseries_endpoint="every_ending",
        dy_interp_type="zero",
        st_interp_type="absence_feat",
        time_interval=[None, None],
        time_series_length=14 * 12,
        time_series_resolution=2 * 3600,
        prediction_window=[6, 12, 24, 48],
        cuda_data=True,
    ):
        MedicalData.__init__(
            self,
            db_path,
            dy_sel,
            st_sel,
            data_cache_path,
            use_historic=use_historic,
            use_after_endpoint=use_after_endpoint,
            include_center=include_center,
            max_patient=max_patient,
            use_basic_info=use_basic_info,
            timeseries_endpoint=timeseries_endpoint,
            time_interval=time_interval,
            time_series_length=time_series_length,
            time_series_resolution=time_series_resolution,
            prediction_window=prediction_window,
        )
        
        if not os.path.exists(data_cache_path):
            use_data_cache = False
            
        if use_data_cache:
            self.read_patients_data_from_cache()
        else:
            raise NotImplementedError
            
        self.cuda_data = cuda_data

        static_item = [self.diagnosis_items]
        if use_basic_info:
            static_item += [self.basic_info_items]
        if use_historic:
            static_item += [
                [item + "_historic" for item in self.dynamic_items],
            ]
        if use_after_endpoint:
            static_item += [
                ["cr_after_endpoint"],
            ]

        self.static_items = np.concatenate(static_item, axis=0).tolist()
        # print(self.static_items)

        self.st_feat_num = len(self.static_items)
        self.dy_feat_num = len(self.dynamic_items)
        self.pred_dim = [int(np.max(list(task.code.values())) + 1) for task in pred_tasks]
        self.pred_tasks = pred_tasks
        self.task_names = [task.name for task in pred_tasks]

        self.use_historic = use_historic
        self.use_after_endpoint = use_after_endpoint

        self.sample_each_patient = [len(p_d.data_dynamic) for p_d in self.patient_data_list]

        self.dy_interp_type = dy_interp_type
        self.st_interp_type = st_interp_type

        if "absence_feat" in self.dy_interp_type:
            self.dy_dim = 2
        else:
            self.dy_dim = 1
        if "absence_feat" in self.dy_interp_type:
            self.st_dim = 2
        else:
            self.st_dim = 1

        self.start_index = 0

        self.sample_lut = {}
        sample_idx = 0
        for patient_i, sample_num_of_patient_i in enumerate(self.sample_each_patient):
            for sample_i in range(sample_num_of_patient_i):
                self.sample_lut[sample_idx] = (patient_i, sample_i)
                sample_idx += 1

    def set_start_index(self, start_index):
        self.start_index = start_index

    def __len__(self):
        return int(np.sum(self.sample_each_patient))

    def interp(self, arr, intp_type="zero"):
        if intp_type == "zero":
            arr[np.isnan(arr)] = np.zeros_like(arr[np.isnan(arr)])

        elif intp_type == "absence_feat" or intp_type == "absence_feat_zeros":
            arr_val = np.zeros_like(arr)
            arr_val[(~np.isnan(arr)) & (np.isfinite(arr))] = arr[(~np.isnan(arr)) & (np.isfinite(arr))]
            arr_abs = np.isnan(np.sum(arr, axis=-1)).astype(float)
            arr_abs = np.expand_dims(arr_abs, axis=-1)
            arr = np.concatenate([arr_val, arr_abs], axis=-1)

        elif intp_type == "absence_feat_nearest":
            interp_arr = np.zeros_like(arr)
            for i in range(arr.shape[1]):
                valid_indices = np.argwhere(~np.isnan(arr[:, i]))[:, 0]
                if len(valid_indices) == 0:
                    interp_arr[:, i] = np.zeros_like(interp_arr[:, i])
                elif len(valid_indices) == 1:
                    interp_arr[:, i] = np.ones_like(interp_arr[:, i]) * np.nanmean(arr[:, i])
                else:
                    f = interpolate.interp1d(valid_indices, arr[valid_indices, i], kind="nearest", fill_value="extrapolate", axis=0)
                    interp_arr[:, i] = f(np.arange(0, arr.shape[0]))
            arr = interp_arr
            arr_abs = np.zeros_like(arr)
            arr = np.concatenate([arr, arr_abs], axis=-1)

        else:
            raise NotImplementedError

        return arr

    def arrange_labels(self, outcomes):
        def find_outcome_in_task(outcomes, task_code):
            matching = []
            for outcome, t, et in outcomes:
                if outcome in task_code and outcome not in matching:
                    matching.append(outcome)

            # assert len(matching) <= 1, f"Endings detected must match <= 1 outcome in task codes! {str(outcomes):s}) -> {str(matching):s}"
            return matching

        label_tasks = {}
        for dim, task in zip(self.pred_dim, self.pred_tasks):
            name = task.name
            outcomes_code = task.code

            outcome_name = find_outcome_in_task(outcomes, outcomes_code)

            label_onehot = np.zeros([dim])
            for outcome in outcome_name:
                label_onehot[outcomes_code[outcome]] = 1

                # 如果匹配名字里有一个正样本，那么就认为是正样本
                # 即：正样本覆盖负样本
                if outcomes_code[outcome] == 1:
                    label_onehot[0] = 0

            # 如果既没有匹配到正样本也没有匹配到负样本，那么就认为是负样本
            # 当前弃用了这种办法，为了忽略掉没有标签的样本
            # if np.sum(label_onehot) == 0:
            #     label_onehot[0] = 1

            # 似乎可以转换为bool类型减少显存占用
            label_tasks[name] = torch.from_numpy(label_onehot).float()

        return label_tasks

    def __getitem__(self, index):
        # find patient index and sample index
        patient_index, sample_index = self.sample_lut[index]

        if np.isnan(patient_index):
            print("Error in finding patient index!")
            patient_index = 0
        patient_data: PatientData = self.patient_data_list[patient_index]

        static_data = [patient_data.diagnosis_data]
        if self.use_basic_info:
            static_data += [
                patient_data.basic_info,
            ]
        if self.use_historic:
            static_data += [
                patient_data.data_historic[sample_index],
            ]
        if self.use_after_endpoint:
            static_data += [
                [patient_data.data_after_endpoint[sample_index]],
            ]
        static_data = np.concatenate(static_data, axis=0)
        static_data = self.interp(np.expand_dims(static_data, axis=-1), self.st_interp_type)
        static_data = torch.from_numpy(static_data).float()
        if self.cuda_data:
            static_data = static_data.cuda()

        dynamic_data = patient_data.data_dynamic[sample_index]

        SAVE_DENSE_TIME = False
        if not SAVE_DENSE_TIME:
            # 稀疏时间序列，转换为稠密时间序列
            (sparse_time_index, sparse_time_data, dshape) = dynamic_data
            dense_dynamic_data = np.ones(dshape) * np.nan
            dense_dynamic_data[sparse_time_index] = sparse_time_data
            dynamic_data = dense_dynamic_data

        dynamic_data = self.interp(np.expand_dims(dynamic_data, axis=-1), self.dy_interp_type)
        dynamic_data = torch.from_numpy(dynamic_data).float()

        if self.cuda_data:
            dynamic_data = dynamic_data.cuda()

        # 查找目标标签的时候比较慢，这里把已经找到的存下来
        if not hasattr(patient_data, "label"):
            patient_data.label = [None for _ in range(len(patient_data.outcomes))]
        if patient_data.label[sample_index] is None:
            label = self.arrange_labels(patient_data.outcomes[sample_index])
            patient_data.label[sample_index] = label
        else:
            label = patient_data.label[sample_index]
        
        if self.cuda_data:
            label = {k: v.cuda() for k, v in label.items()}
        
        return (dynamic_data, static_data, label)
    
    # 将单个数据合并为一个mini-batch
    def get_collate_fn(self):
        return collate_batch


def collate_batch(sample_list):
    dy_data_batch = [dy[None] for dy, st, label in sample_list]
    st_data_batch = [st[None] for dy, st, label in sample_list]

    dy_data_batch = torch.concat(dy_data_batch, axis=0)
    st_data_batch = torch.concat(st_data_batch, axis=0)

    label_batch = [label for dy, st, label in sample_list]
    # dict的合并，是将每个dict对应key的item分别合并
    label_dict = []
    for key in label_batch[0].keys():
        label_dict += [
            torch.concat([label[key][None] for label in label_batch], axis=0),
        ]

    return dy_data_batch, st_data_batch, label_dict
