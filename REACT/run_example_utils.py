import os, sys
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops

sys.path.append(opd(__file__))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import tqdm
from utils.fig_utils import *
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


from medical_pred_main import Granger_Causal_Prediction, prepare_data
from utils.misc import (
    reproduc,
    read_dict_from_csv,
)
from utils.opt_type import MultiCADopt
from utils.logger import MyLogger

# plt.rcParams["font.sans-serif"] = "Arial"


class Example_Prediction(Granger_Causal_Prediction):
    def __init__(self, args, log, device="cuda"):
        super().__init__(args, log, device)

    def load_model(self, load_dir, dy_items, st_items):
        model_dir = opj(load_dir, "model.pt")
        self.fitting_model.load_state_dict(torch.load(model_dir))

        # Data Prediction Testing
        self.dy_items = dy_items
        self.st_items = st_items
        self.all_items = dy_items + st_items

        dy_graph_dict = read_dict_from_csv(opj(load_dir, "dy_sel.csv"))
        st_graph_dict = read_dict_from_csv(opj(load_dir, "st_sel.csv"))
        # print(dy_items, st_items, dy_graph_dict, st_graph_dict)
        dy_graph = torch.tensor([float(dy_graph_dict[dy_item][0]) for dy_item in dy_items]).to(self.graph.device)
        st_graph = torch.tensor([float(st_graph_dict[st_item][0]) for st_item in st_items]).to(self.graph.device)

        thres = 0.5
        self.graph = (torch.cat([dy_graph, st_graph], dim=0) > thres).float()  # binarize
        self.graph = self.graph * 200 - 100  # scale to -100 to 100
        self.args.data_pred.hard_mask = True  # hard thresholding instead of bernonlli sampling
        self.epoch_i = 0

        self.valid_feat_name = [dy_item for dy_item in dy_items if float(dy_graph_dict[dy_item][0]) > thres] + [st_item for st_item in st_items if float(st_graph_dict[st_item][0]) > thres]

    def pred_single_patient(self, dynamic_data, static_data, label):
        # 如果数据不在gpu上或者是numpy，则转换成gpu tensor
        if not isinstance(dynamic_data, torch.Tensor):
            dynamic_data = torch.tensor(dynamic_data)
            static_data = torch.tensor(static_data)
        dynamic_data = dynamic_data.to(self.device)[None]
        static_data = static_data.to(self.device)[None]

        if isinstance(label, dict):
            label_batch = [label]
            # dict的合并，是将每个dict对应key的item分别合并
            label_dict = []
            for key in label_batch[0].keys():
                label_dict += [
                    torch.concat([label[key][None] for label in label_batch], axis=0),
                ]
            label = label_dict

        # 预测单个样本
        pred_batch, loss, _ = self.data_pred(dynamic_data, static_data, label, mode="test")
        return [p.detach().cpu() for p in pred_batch]

    def calc_causal_effect(self, dynamic_data_list, static_data_list, label_list, task_of_interest, save_path="outputs/example_exp.csv"):
        # 测试一系列病人，计算每个变量的因果效应

        if isinstance(task_of_interest, str):
            task_i = self.args.data_pred.task_names.index(task_of_interest)

        if len(self.valid_feat_name) == 0:
            return None

        # 通过计算平均值作为计算因果效应的参考值
        all_dy = torch.cat(dynamic_data_list, dim=0)[:, :, :, 0].clone()
        all_dy[all_dy == 0] = torch.nan
        all_st = torch.cat(static_data_list, dim=0)[:, :, 0].clone()
        all_st[all_st == 0] = torch.nan
        dy_ref = torch.nanmean(torch.nanmean(all_dy, dim=1), dim=0)
        st_ref = torch.nanmean(all_st, dim=0)
        ref = torch.cat([dy_ref, st_ref], dim=0)

        test_res = []
        print("Valid features: ", self.valid_feat_name)
        for feat_name in self.valid_feat_name + ["None"]:
            if feat_name != "None":
                feat_index = self.all_items.index(feat_name)
                self.graph[feat_index] = -100
            else:
                feat_index = 0

            # print(f"Testing {feat_name} {feat_index}...")
            pred_epoch = []
            label_epoch = []
            avg_feat_epoch = []
            for i, (dynamic_data, static_data, label) in enumerate(zip(dynamic_data_list, static_data_list, label_list)):
                pred_batch, loss, _ = self.data_pred(dynamic_data, static_data, label, mode="test", ref="zero")
                pred_epoch.append(pred_batch[task_i].detach().cpu())
                label_epoch.append(label[task_i].detach().cpu())

                dynamic_data = dynamic_data.cpu().clone()
                dynamic_data[dynamic_data == 0] = torch.nan
                avg_dy = torch.nanmean(dynamic_data[:, :, :, 0], dim=[1])
                avg_dy[torch.isnan(avg_dy)] = dy_ref.cpu()[None].expand(avg_dy.shape[0], -1)[torch.isnan(avg_dy)]

                static_data = static_data.cpu().clone()
                static_data[static_data == 0] = torch.nan
                avg_st = static_data[:, :, 0]
                avg_st[torch.isnan(avg_st)] = st_ref.cpu()[None].expand(avg_st.shape[0], -1)[torch.isnan(avg_st)]

                avg_feat_epoch.append(torch.cat([avg_dy, avg_st], axis=1))
                torch.cuda.empty_cache()
                # break # Debug
            pred_epoch = torch.cat(pred_epoch, dim=0)
            pred_epoch = np.exp(pred_epoch[:, 1]) / (np.exp(pred_epoch[:, 0]) + np.exp(pred_epoch[:, 1]))
            avg_feat_epoch = torch.cat(avg_feat_epoch, dim=0)
            # pred_epoch = pred_epoch[:,1]
            label_epoch = torch.cat(label_epoch, dim=0)[:, 1]

            test_res.append((feat_name, pred_epoch, label_epoch, avg_feat_epoch[:, feat_index]))
            self.graph[feat_index] = 100

        # 计算的因果效应=某个指标变化100%平均值，对预测结果的影响
        ref = ref.cpu().numpy()
        all_feat_name = [feat_name for feat_name, _, _, _ in test_res]
        test_df = pd.DataFrame({**{feat_name + "_input": avg_feat for feat_name, pred, label, avg_feat in test_res}, **{feat_name + "_output": pred for feat_name, pred, label, avg_feat in test_res}, **{"Label": test_res[0][2]}})

        # print(ref)
        for column in all_feat_name:
            # print("Calculating ", column)
            if column != "Label" and column != "None":
                test_df_new = pd.DataFrame(
                    {
                        **{column + "_ref": ref[self.all_items.index(column)]},
                        **{column + "_pert": (test_df[column + "_input"] - ref[self.all_items.index(column)]) / ref[self.all_items.index(column)]},
                        **{column + "_ce": (test_df["None_output"] - test_df[column + "_output"])},
                    }
                )
                test_df = pd.concat([test_df, test_df_new], axis=1)

        test_df.sort_index(inplace=True, axis=1)
        test_df.to_csv(save_path)
        return test_df


def prepare_example_model(opt: MultiCADopt, device="cuda", mode="train"):
    reproduc(**opt.reproduc)
    # timestamp = datetime.now().strftime("_%Y_%m%d_%H%M%S_%f")
    opt.task_name += "_" + mode + "feat_imp"
    proj_path = opj(opt.dir_name, opt.task_name)
    log = MyLogger(log_dir=proj_path, **opt.log)
    log.log_opt(opt)

    train_dataset, val_dataset, test_dataset = prepare_data(opt.data)

    if hasattr(opt, "gc_pred"):
        # Specify the number of features and dimensions if is set to auto
        if opt.gc_pred.dy_feat_num == "auto":
            print("Dy feat num = ", test_dataset.dy_feat_num)
            opt.gc_pred.dy_feat_num = test_dataset.dy_feat_num

        if opt.gc_pred.st_feat_num == "auto":
            print("St feat num = ", test_dataset.st_feat_num)
            opt.gc_pred.st_feat_num = test_dataset.st_feat_num

        if opt.gc_pred.dy_dim == "auto":
            print("Dy dim = ", test_dataset.dy_dim)
            opt.gc_pred.dy_dim = test_dataset.dy_dim

        if opt.gc_pred.st_dim == "auto":
            print("St dim = ", test_dataset.st_dim)
            opt.gc_pred.st_dim = test_dataset.st_dim

        if opt.gc_pred.t_length == "auto":
            print("T length = ", test_dataset.time_series_length)
            opt.gc_pred.t_length = test_dataset.time_series_length

        if opt.gc_pred.data_pred.pred_dim == "auto":
            print("Pred dim = ", test_dataset.pred_dim)
            opt.gc_pred.data_pred.pred_dim = test_dataset.pred_dim
            opt.gc_pred.data_pred.task_names = test_dataset.task_names

    # pred_model = Example_Prediction(opt.gc_pred, log, device=device)
    return opt.gc_pred, log, test_dataset




def select_data(dataset, task_of_interest, pred_model, max_sample=1000, select_mode="none", shuffle=True):
    if isinstance(task_of_interest, str):
        task_i = pred_model.args.data_pred.task_names.index(task_of_interest)

    test_loader = DataLoader(
        dataset,
        batch_size=1000,
        shuffle=shuffle,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,
    )

    # 从数据中选出max_sample个正样本
    sel_patient = 0
    dynamic_data_list = []
    static_data_list = []
    label_list = []
    for i, (dynamic_data, static_data, label) in enumerate(test_loader):
        if sel_patient >= max_sample:
            break

        if select_mode == "pos":
            sel_index = torch.argwhere(label[task_i][:, 1] == 1)[:, 0]
        elif select_mode == "neg":
            sel_index = torch.argwhere(label[task_i][:, 1] == 0)[:, 0]
        elif select_mode == "none":
            sel_index = torch.arange(label[task_i].shape[0])
        else:
            raise NotImplementedError

        print(f"Sel patient ({select_mode}): {sel_index.shape[0]} / {label[task_i].shape[0]}")
        dynamic_data_list.append(dynamic_data[sel_index[: max_sample - sel_patient]])
        static_data_list.append(static_data[sel_index[: max_sample - sel_patient]])
        label_list.append([label_task[sel_index[: max_sample - sel_patient]] for label_task in label])
        sel_patient += dynamic_data_list[-1].shape[0]

    data = (dynamic_data_list, static_data_list, label_list)
    return data


COLOR_CODE = {"肌酐": "B2", "尿素氮": "C3", "血清尿酸": "C2", "乳酸脱氢酶": "B1", "肌酸激酶同工酶ng/ml": "C1", "None": "B2"}
Y_LAEL_LUT = {
    "肌酐": "Creatinine (mg/dL)",
    "血清尿酸": "Uric Acid (mg/dL)",
    "尿素氮": "Blood Urea Nitrogen \n(mg/dL)",
    "直接胆红素": "Direct Bilirubin \n($\mu$mol / L)",
    "乳酸脱氢酶": "Lactate \n Dehydrogenase (U/L)",
    "肌酸激酶同工酶ng/ml": "Creatine \n Kinase-MB (ng/mL)",
}
MAX_RANGE = {"肌酐": 2, "血清尿酸": 700, "尿素氮": 15, "直接胆红素": 40, "乳酸脱氢酶": 600, "肌酸激酶同工酶ng/ml": 100}


def get_items_time_series(patient_data, items=["肌酐", "血清尿酸", "尿素氮", "乳酸脱氢酶", "肌酸激酶同工酶ng/ml"]):
    time_series_length = 14 * 12
    time_series_resolution = 2 * 3600
    operation_time = patient_data.op_time
    # focus_window_ep = patient_data.prediction_endpoint_list[example_sample_index]
    # print(example_sample_index, len(patient_data.data_dynamic))

    test_tp = {item_name: [] for item_name in items}
    test_val = {item_name: [] for item_name in items}
    # focus_test_tp = []
    # focus_test_val = []
    outcomes_tp = []
    outcomes_name = []
    pred_tp = []
    pred_val = []

    print("Sample num: ", len(patient_data.data_dynamic))
    # 因为是滑动窗口，所以需要从多个sample中组合出该patient的时间序列
    for sample_index in range(len(patient_data.data_dynamic)):
        this_window_ep = patient_data.prediction_endpoint_list[sample_index]

        # 读取预测概率和时间点
        pred_prob = patient_data.predictions[sample_index]
        pred_val.append(pred_prob)
        pred_tp.append(this_window_ep)

        # 读取结局名称和时间点
        for name, tp, _ in patient_data.outcomes[sample_index]:
            if tp not in outcomes_tp and "no_aki" not in name:
                outcomes_tp.append(tp)
                outcomes_name.append(name)

        # 从连续时间序列中读取真实时间点和数据
        time_points = np.arange(this_window_ep - time_series_resolution * time_series_length, this_window_ep, time_series_resolution)
        dynamic_data = patient_data.data_dynamic[sample_index]
        SAVE_DENSE_TIME = False
        if not SAVE_DENSE_TIME:
            # 稀疏时间序列，转换为稠密时间序列
            (sparse_time_index, sparse_time_data, dshape) = dynamic_data
            dense_dynamic_data = np.ones(dshape) * np.nan
            dense_dynamic_data[sparse_time_index] = sparse_time_data
            dynamic_data = dense_dynamic_data

        # 对于每个关心的指标，读取其时间序列
        for item_name in items:
            item_index = patient_data.dynamic_items.index(item_name)
            test_data = dynamic_data[:, item_index]
            valid_index = np.where(~np.isnan(test_data))[0]
            # print(f"Valid index for {item_name}: {len(valid_index)}")

            for vi in valid_index:
                if time_points[vi] not in test_tp[item_name]:
                    test_tp[item_name].append(time_points[vi])
                    test_val[item_name].append(test_data[vi])

            # if sample_index == example_sample_index:
            #     for vi in valid_index:
            #         if time_points[vi] not in focus_test_tp:
            #             focus_test_tp.append(time_points[vi])
            #             focus_test_val.append(test_data[vi])

    outcomes_tp = np.array(outcomes_tp)
    outcomes_name = np.array(outcomes_name)
    pred_tp = np.array(pred_tp)
    pred_val = np.array(pred_val)
    return operation_time, test_tp, test_val, outcomes_tp, outcomes_name, pred_tp, pred_val


def get_sample_lut(patient_data_list):
    # 对每个sample，计算其对应的patient和sample index
    sample_each_patient = [len(p_d.data_dynamic) for p_d in patient_data_list]
    sample_lut = {}
    sample_idx = 0
    for patient_i, sample_num_of_patient_i in enumerate(sample_each_patient):
        for sample_i in range(sample_num_of_patient_i):
            sample_lut[sample_idx] = (patient_i, sample_i)
            sample_idx += 1
    return sample_lut


def draw_patient_example(patient_data, example_sample_index):
    patient_time_series = get_items_time_series(patient_data, items=["肌酐", "血清尿酸", "尿素氮", "乳酸脱氢酶", "肌酸激酶同工酶ng/ml"])
    operation_time, test_tp, test_val, outcomes_tp, outcomes_name, pred_tp, pred_val = patient_time_series
    # print(test_tp)

    focus_window_ep = patient_data.prediction_endpoint_list[example_sample_index]
    # print(ENGLISH_LUT)

    plt.figure(figsize=(15, 7))
    time_range = [operation_time - 3 * 24 * 3600, focus_window_ep + 2 * 24 * 3600]

    for i, item_name in enumerate(test_tp.keys()):
        plt.subplot(len(test_val.keys()), 1, i + 1)
        ax = plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if i != len(test_val.keys()) - 1:
            ax.spines["bottom"].set_visible(False)

        # plt.title(english_lut.get(item_name))
        plt.plot(test_tp[item_name], test_val[item_name], marker="o", zorder=1, lw=2, markersize=3, alpha=1, color=MAIN_COLOR[COLOR_CODE[item_name]])
        # "#518FB0"

        v_min, v_max = np.min(test_val[item_name]), np.max(test_val[item_name])
        # v_min, v_max = 0, MAX_RANGE[item_name]

        plt.fill_between([focus_window_ep - 14 * 24 * 3600, focus_window_ep + 14 * 24 * 3600], v_min * 0.95, v_max * 1.05, edgecolor="face", color=SECOND_COLOR["C3"], alpha=0.3, zorder=0)
        
        plt.plot(
            [operation_time, operation_time],
            [
                v_min * 0.95,
                v_max * 1.05,
            ],
            color=MAIN_COLOR["C3"],
            linestyle="--",
            zorder=1,
        )

        # print(outcomes_name)
        plt.ylabel(Y_LAEL_LUT.get(item_name))
        plt.xlim(time_range)
        plt.xticks([])
        # plt.legend()

    plt.tight_layout()
    plt.savefig(opj(opd(__file__), f"example/outputs/data_{patient_data.pid}_{example_sample_index:d}.svg"), dpi=300)
    plt.show()


def calc_prediction(pred_model, test_dataset, start_index, end_index, interested_task="all_aki_24h"):
    data_list = []
    for i in range(start_index, end_index):
        data_list.append(test_dataset[i])

    pred_dict = {}

    # for exclude_item in ["肌酐", "血清尿酸", "尿素氮", "乳酸脱氢酶", "肌酸激酶同工酶ng/ml", "None"]:
    for exclude_item in ["None"]:
        pred_dict[exclude_item] = []
        if exclude_item != "None":
            exclude_index = test_dataset.dynamic_items.index(exclude_item)
            original_val = pred_model.graph[exclude_index].clone()
            pred_model.graph[exclude_index] = -100
            print(f"Exclude {exclude_item} changing the value from {original_val} to {pred_model.graph[exclude_index]}")
            print(f"Valid item index: {torch.argwhere(pred_model.graph > 0).detach().cpu().numpy()[:,0]}")

        task_i = pred_model.task_names.index(interested_task)
        for dynamic_data, static_data, label_batch in data_list:
            pred = pred_model.pred_single_patient(dynamic_data, static_data, label_batch)
            pred_this_task = torch.softmax(pred[task_i][0], dim=0)[1]
            pred_dict[exclude_item].append(pred_this_task)

        if exclude_item != "None":
            pred_model.graph[exclude_index] = original_val

    # print(pred_dict["None"])
    return pred_dict


def draw_prediction(patient_data, pred_dict, pid):
    YLIM = [0, 1]
    threshold = 0.112

    operation_time = patient_data.op_time
    time_range = [operation_time - 3 * 24 * 3600, operation_time + 14 * 24 * 3600]
    time_ticks = np.arange(time_range[0], time_range[1], 48 * 3600)

    plt.figure(figsize=(15, 6))

    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    for item_name in pred_dict.keys():
        plt.plot(
            patient_data.prediction_endpoint_list,
            pred_dict[item_name],
            label=f"prediction without {ENGLISH_LUT.get(item_name)}" if item_name != "None" else "Actual prediction",
            c=MAIN_COLOR.get(COLOR_CODE[item_name]),
            lw=1.5 if item_name != "None" else 3,
        )

    tp_draw = []
    aki_type_draw = []
    for outcome_list in patient_data.outcomes:
        for outcome, t, et in outcome_list:
            if "no_aki" in outcome:
                continue
            if t in tp_draw:
                continue
            if "severe" in outcome:
                aki_type_draw.append("severe")
            elif "moderate" in outcome:
                aki_type_draw.append("moderate")
            else:
                aki_type_draw.append("mild")
            tp_draw.append(t)

    markers = {"severe": "v", "moderate": "s", "mild": "o"}
    for type in ["severe", "moderate", "mild"]:
        tp_draw_type = [tp for tp, t in zip(tp_draw, aki_type_draw) if t == type]
        plt.scatter(tp_draw_type, [0.5] * len(tp_draw_type), marker=markers[type], color=MAIN_COLOR["A"], s=100, zorder=2, label=f"{type} AKI onset")

    plt.fill_between(time_range, YLIM[0], YLIM[1], edgecolor="face", color=SECOND_COLOR["C3"], alpha=0.3, zorder=0)
    plt.plot(time_range, [threshold, threshold], color=MAIN_COLOR["A"], linestyle="solid", zorder=0, label="Threshold", alpha=0.35)

    plt.xticks(time_ticks, (time_ticks - operation_time) // (24 * 3600), rotation=20)
    plt.xlim(time_range)
    plt.ylabel("Predictive risk for AKI\n within 24 hours")
    plt.xlabel("Days since operation")

    plt.legend(loc="lower center", ncol=5, fancybox=False, shadow=False, bbox_to_anchor=(0.5, -0.4), frameon=False)
    plt.tight_layout()

    plt.savefig(opj(opd(__file__), f"example/outputs/prediction_{pid}.svg"), dpi=300)
    plt.show()


# if __name__ == "__main__":
#     if "release" not in os.getcwd():
#         os.chdir(opj(os.getcwd(), "release"))
        
#     device = "cuda"
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     np.random.seed(42)

#     opt_gc_pred, log, test_dataset = prepare_example_model(OmegaConf.load(opj(os.getcwd(), "example/example.yaml")), device=device, mode="test")
#     load_dir = opj(os.getcwd(), "model_weights/6_var")
#     pred_model = Example_Prediction(opt_gc_pred, log, device=device)
#     pred_model.load_model(load_dir, test_dataset.dynamic_items, test_dataset.static_items)
#     patient_data_list = test_dataset.patient_data_list


#     for example_patient_index, patient_data in enumerate(patient_data_list):
#         example_sample_index = len(patient_data.data_dynamic) // 2
#         draw_patient_example(patient_data, example_sample_index)

#         patient_start = int(np.sum([len(p_d.data_dynamic) for p_d in patient_data_list[:example_patient_index]]))
#         pred_dict = calc_prediction(pred_model, test_dataset, patient_start, patient_start + len(patient_data.data_dynamic))
#         draw_prediction(patient_data, pred_dict, patient_data.pid)



