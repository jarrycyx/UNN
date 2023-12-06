import torch
import numpy as np
import sklearn
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import tqdm

from data_prep.pla_data.patient_data import PatientData
from data_prep.data_utils.time_format import time2str


plt.rcParams['font.sans-serif'] = "Arial"

ENGLISH_LUT = {
    "平均红细胞血红蛋白浓度": "MCHC",
    "尿素氮": "BUN",
    "肌酐": "SCr",
    "血清尿酸": "UA",
    "中性粒细胞百分比": "NEUT%",
    "血小板计数": "PLT",
    "肌钙蛋白T": "TnT",
    "天冬氨酸氨基转移酶": "AST",
    "丙氨酸氨基转移酶": "ALT",
    "直接胆红素": "DBIL",
    "C-反应蛋白测定": "CRP",
    "凝血酶时间测定": "PT",
    "AGE": "Age",
    "SEX": "Sex",
    "白细胞计数": "WBC",
    "血浆凝血酶原活动度测定": "PTA",
    "葡萄糖": "GLU",
    "平均红细胞体积": "MCV",
    "钠": "Na",
    "总蛋白": "TP",
    "血红蛋白测定": "HGB",
    "乳酸脱氢酶": "LDH",
    "肌酸激酶同工酶ng/ml": "CK-MB",
    "血清白蛋白": "ALB",
    "平均红细胞血红蛋白量": "MCHC",
    "AKI": "AKI",
    "尿70%红细胞前向散射光所在位置": "RBC70%",
    "乙型肝炎病毒E抗原": "HBeAg",
    "尿红细胞检查": "RBC",
    "镁": "Mg",
    "红细胞计数": "RBC",
    "无机磷": "P",
    "肌酸激酶": "CK",
}
ENGLISH_LUT.setdefault("None", "None")


MAIN_COLOR = {
    "A": "#C25756",
    "B1": "#61afa5",
    "B2": "#5798b9",
    "C1": "#cc7b74",
    "C2": "#dba25f",
    "C3": "#b1b1b1",
}
SECOND_COLOR = {
    "A": "#d07e7e",
    "B1": "#7abcb3",
    "B2": "#76adc6",
    "C1": "#d99a96",
    "C2": "#e1af79",
    "C3": "#bfbfbf",
}


def draw_curve(
    name,
    auc_data,
    main_color_code={"Ours": "A", "MLP": "C3", "LSTM": "B2", "Transformer": "B1", "XGBoost": "C2", 
                     "MLP (Causal Var.)": "C3", "LSTM (Causal Var.)": "B2", "Transformer (Causal Var.)": "B1", "XGBoost (Causal Var.)": "C2"},
    second_color_code={"Ours": "A", "MLP": "C3", "LSTM": "B2", "Transformer": "B1", "XGBoost": "C2", 
                       "MLP (Causal Var.)": "C3", "LSTM (Causal Var.)": "B2", "Transformer (Causal Var.)": "B1", "XGBoost (Causal Var.)": "C2"},
    show_fig=False,
):
    methods = auc_data.keys()

    plt.figure(figsize=(4.5, 4.5))

    ax = plt.gca()
    # ax.spines["right"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    plt.title("ROC Curve")

    plt.plot([0, 1], [0, 1], linestyle="--", label=f"Random guess (AUROC: 0.5)", c="#aaaaaa", lw=1, zorder=1)
    # plt.fill_between(auc_data["Ours"][1], np.zeros_like(auc_data["Ours"][1]), auc_data["Ours"][2], color=SECOND_COLOR["Ours"], alpha=0.15)
    for method in methods:
        print(f"Draw {method} curve")
        if "XGBoost" in method:
            # xgboost 只展示一个点，没有连续的曲线
            best_idx = np.argmax(auc_data[method]["roc_data"][0][:, 1] - auc_data[method]["roc_data"][0][:, 0])
            plt.scatter(
                auc_data[method]["roc_data"][0][best_idx, 0],
                auc_data[method]["roc_data"][0][best_idx, 1],
                c=SECOND_COLOR[main_color_code[method]],
                marker="v",
                edgecolors=MAIN_COLOR[main_color_code[method]],
                linewidths=0.5,
                label=f"{method}",
            )
            plt.errorbar(
                auc_data[method]["roc_data"][0][best_idx, 0]+0.015,
                auc_data[method]["roc_data"][0][best_idx, 1],
                yerr=[auc_data[method]["roc_data"][0][best_idx, 1:2]-auc_data[method]["roc_data"][1][best_idx, 1:2], 
                      auc_data[method]["roc_data"][0][best_idx, 1:2]-auc_data[method]["roc_data"][1][best_idx, 1:2], ],
                fmt="none", ecolor=MAIN_COLOR[main_color_code[method]], capsize=3 ,lw=0.5)
        else:
            plt.plot(
                auc_data[method]["roc_data"][0][:, 0],
                auc_data[method]["roc_data"][0][:, 1],
                label=f"{method} (AUROC: {auc_data[method]['auroc'][0]:.3f} [{auc_data[method]['auroc'][1]:.3f}-{auc_data[method]['auroc'][2]:.3f}])",
                lw=2,
                c=MAIN_COLOR[main_color_code[method]],
                zorder=3,
            )
            plt.fill_between(
                auc_data[method]["roc_data"][1][:, 0],
                auc_data[method]["roc_data"][1][:, 1],
                auc_data[method]["roc_data"][2][:, 1],
                color=SECOND_COLOR[second_color_code[method]],
                alpha=0.3,
                zorder=2,
                edgecolor="none",
            )
        # print(auc_data[method]["roc_data"][2][:, 1] - auc_data[method]["roc_data"][1][:, 1],)

    plt.xlabel("1 - Specificity")
    plt.ylabel("Senstivity")
    # plt.grid(color="#dddddd")
    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)
    plt.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.grid(axis="both", linestyle = '--', alpha=0.4)
    plt.savefig(f"exp/pla_exp/figs/save/curves/{name}_auc.svg", dpi=300)
    if show_fig:
        plt.show()
    else:
        plt.clf()

    plt.figure(figsize=(4.5, 4.5))

    ax = plt.gca()
    # ax.spines["right"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    plt.title("PRC Curve")
    # plt.fill_between(auc_data["Ours"][1], np.zeros_like(auc_data["Ours"][1]), auc_data["Ours"][2], color=SECOND_COLOR["Ours"], alpha=0.15)
    for method in methods:
        print(f"Draw {method} curve")
        if "XGBoost" in method:
            # xgboost 只展示一个点，没有连续的曲线
            best_idx = np.argmin(np.abs(auc_data[method]["prc_data"][0][:, 0] - 0.5))
            plt.scatter(
                auc_data[method]["prc_data"][0][best_idx, 0],
                auc_data[method]["prc_data"][0][best_idx, 1],
                c=SECOND_COLOR[main_color_code[method]],
                marker="v",
                edgecolors=MAIN_COLOR[main_color_code[method]],
                linewidths=0.5,
                label=f"{method}",
            )
            plt.errorbar(
                auc_data[method]["prc_data"][0][best_idx, 0]+0.015,
                auc_data[method]["prc_data"][0][best_idx, 1],
                yerr=[auc_data[method]["prc_data"][0][best_idx, 1:2]-auc_data[method]["prc_data"][1][best_idx, 1:2], 
                      auc_data[method]["prc_data"][0][best_idx, 1:2]-auc_data[method]["prc_data"][1][best_idx, 1:2], ],
                fmt="none", ecolor=MAIN_COLOR[main_color_code[method]], capsize=3 ,lw=0.5)
        else:
            plt.plot(
                auc_data[method]["prc_data"][0][:, 0],
                auc_data[method]["prc_data"][0][:, 1],
                label=f"{method} (AUPRC: {auc_data[method]['auprc'][0]:.3f} [{auc_data[method]['auprc'][1]:.3f}-{auc_data[method]['auprc'][2]:.3f}])",
                lw=2,
                c=MAIN_COLOR[main_color_code[method]],
                zorder=3,
            )
            plt.fill_between(
                auc_data[method]["prc_data"][1][:, 0],
                auc_data[method]["prc_data"][1][:, 1],
                auc_data[method]["prc_data"][2][:, 1],
                color=SECOND_COLOR[second_color_code[method]],
                alpha=0.3,
                zorder=2,
                edgecolor="none",
            )

    plt.xlabel("1 - Specificity")
    plt.ylabel("Senstivity")
    # plt.grid(color="#dddddd")
    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)
    plt.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.grid(axis="both", linestyle = '--', alpha=0.4)
    plt.savefig(f"exp/pla_exp/figs/save/curves/{name}_prc.svg", dpi=300)
    if show_fig:
        plt.show()
    else:
        plt.clf()


def load_preds(val_res_path, test_res_path, interested_task="all_aki_24h"):
    (val_tasks_preds, val_tasks_labels) = torch.load(val_res_path)
    val_preds = val_tasks_preds[interested_task]
    val_labels = val_tasks_labels[interested_task]
    (test_tasks_preds, test_tasks_labels) = torch.load(test_res_path)
    test_preds = test_tasks_preds[interested_task]
    test_labels = test_tasks_labels[interested_task]

    if len(val_preds.shape) == 2:
        if not isinstance(val_preds, torch.Tensor):
            val_preds = torch.tensor(val_preds)
            test_preds = torch.tensor(test_preds)
        # Pass Through Softmax
        val_preds = torch.softmax(val_preds, dim=1).numpy()
        test_preds = torch.softmax(test_preds, dim=1).numpy()
        val_labels = val_labels.numpy()
        test_labels = test_labels.numpy()
    else:
        val_preds = np.array(val_preds)[:, None]
        test_preds = np.array(test_preds)[:, None]
        val_labels = np.array(val_labels)[:, None]
        test_labels = np.array(test_labels)[:, None]
        
        val_preds = np.concatenate([1 - val_preds, val_preds], axis=1)
        test_preds = np.concatenate([1 - test_preds, test_preds], axis=1)
        val_labels = np.concatenate([1 - val_labels, val_labels], axis=1)
        test_labels = np.concatenate([1 - test_labels, test_labels], axis=1)
    
    test_valid_mask = np.where(np.sum(test_labels, axis=1) != 0)
    test_label_valid = test_labels[test_valid_mask]
    test_pred_valid = test_preds[test_valid_mask]
    
    val_valid_mask = np.where(np.sum(val_labels, axis=1) != 0)
    val_label_valid = val_labels[val_valid_mask]
    val_pred_valid = val_preds[val_valid_mask]

    return val_pred_valid, val_label_valid, test_pred_valid, test_label_valid


def bootstrap_auc_ci(label, pred, bootstraps=100, fold_size=None, n_point=200, alpha=0.95):
    if fold_size is None:
        fold_size = label.shape[0]
    
    aurocs = np.zeros(bootstraps)
    auprcs = np.zeros(bootstraps)
    roc_points = np.zeros((bootstraps, n_point, 2))
    prc_points = np.zeros((bootstraps, n_point, 2))

    x_axis_points = np.linspace(0, 1, n_point)
    for i in range(bootstraps):
        indices = np.random.choice(label.shape[0], size=fold_size, replace=True)
        pred_sample, label_sample = pred[indices], label[indices]
        auroc = roc_auc_score(label_sample, pred_sample)
        auprc = average_precision_score(label_sample, pred_sample)
        roc_fpr, roc_tpr, roc_thres = roc_curve(label_sample, pred_sample, pos_label=1)
        interp_tpr = np.interp(x_axis_points, roc_fpr, roc_tpr)
        
        prec, rec, thres = precision_recall_curve(label_sample, pred_sample, pos_label=1)
        interp_rec = np.interp(x_axis_points, prec, rec)

        roc_points[i, :, 0] = x_axis_points
        roc_points[i, :, 1] = interp_tpr
        prc_points[i, :, 0] = x_axis_points
        prc_points[i, :, 1] = interp_rec
        aurocs[i] = auroc
        auprcs[i] = auprc

    auroc_avg, auroc_l, auroc_r = (
        np.mean(aurocs),
        np.percentile(aurocs, (1 - alpha) / 2 * 100),
        np.percentile(aurocs, (1 + alpha) / 2 * 100),
    )
    auprc_avg, auprc_l, auprc_r = (
        np.mean(auprcs),
        np.percentile(auprcs, (1 - alpha) / 2 * 100),
        np.percentile(auprcs, (1 + alpha) / 2 * 100),
    )

    roc_avg, roc_l, roc_r = (
        np.mean(roc_points, axis=0),
        np.percentile(roc_points, (1 - alpha) / 2 * 100, axis=0),
        np.percentile(roc_points, (1 + alpha) / 2 * 100, axis=0),
    )
    prc_avg, prc_l, prc_r = (
        np.mean(prc_points, axis=0),
        np.percentile(prc_points, (1 - alpha) / 2 * 100, axis=0),
        np.percentile(prc_points, (1 + alpha) / 2 * 100, axis=0),
    )
    return (
        (auroc_avg, auroc_l, auroc_r),
        (auprc_avg, auprc_l, auprc_r),
        (roc_avg, roc_l, roc_r),
        (prc_avg, prc_l, prc_r),
    )


def evaluate(pred, label, bootstraps=100):
    # 如果所有位都是0，那么认为是无效样本
    valid_mask = np.where(np.sum(label, axis=1) != 0)
    label_valid = label[valid_mask]
    pred_valid = pred[valid_mask]
    print(f"Valid sample: {label_valid.shape[0]} / {label.shape[0]}", end="   | ")

    pred, label = pred_valid[:, 1], label_valid[:, 1]

    """Calculate TPR, FPR, Precision, Accuracy"""
    l, r = np.min(pred), np.max(label)
    # r = l + (r - l) * 0.2
    acc_list = []
    tpr_list = []
    fpr_list = []
    prec_list = []
    thres_list = np.arange(l, r, step=(r - l) / 100)

    # auc = roc_auc_score(label, pred)
    # roc_fpr, roc_tpr, roc_thres = roc_curve(label, pred, pos_label=1)

    # ap = average_precision_score(label, pred)
    # prc_prec, prc_recall, prc_thres = precision_recall_curve(label, pred, pos_label=1)

    (
        (auroc_avg, auroc_l, auroc_r),
        (auprc_avg, auprc_l, auprc_r),
        (roc_avg, roc_l, roc_r),
        (prc_avg, prc_l, prc_r),
    ) = bootstrap_auc_ci(
        label,
        pred,
        bootstraps=bootstraps
    )

    for thres in thres_list:
        pred_bin = (np.array(pred) > thres).astype(int)
        label_bin = np.array(label)

        acc = np.mean(pred_bin == label_bin)
        tpr = np.sum((pred_bin == 1) & (label_bin == 1)) / (np.sum(label_bin == 1) + 1e-4)
        fpr = np.sum((pred_bin == 1) & (label_bin == 0)) / (np.sum(label_bin == 0) + 1e-4)
        prec = np.sum((pred_bin == 1) & (label_bin == 1)) / (np.sum(pred_bin == 1) + 1e-4)

        acc_list.append(acc)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        prec_list.append(prec)

        # print(f"thres: {thres:.2f}, acc: {acc:.2f}, tpr: {tpr:.2f}, fpr: {fpr:.2f}, prec: {prec:.2f}")

    return (
        acc_list,
        tpr_list,
        fpr_list,
        prec_list,
        thres_list,
        (auroc_avg, auroc_l, auroc_r),
        (auprc_avg, auprc_l, auprc_r),
        (roc_avg, roc_l, roc_r),
        (prc_avg, prc_l, prc_r),
    )


def sel_by_patient(patient_data_list, criterion="age", val=[0,10], return_n_patient=False, return_patients=False):
    """
    选择符合条件的病人数据，并将其预测结果和标签数据合并。
    
    Args:
    - patient_data_list: List[PatientData]，病人数据列表。
    - criterion: str，筛选条件，可选值为"age"、"gender"、"year"、"center"、"aki_type"、"emergency"、"surgery_type"。
    - val: Union[str, List[Union[str, int]]]，筛选条件的值，当criterion为"emergency"时，val可选值为"emergency"或"elective"。
    - return_n_patient: bool，是否返回符合条件的病人数量。
    - return_patients: bool，是否返回符合条件的病人数据列表。
    
    Returns:
    - pred_data: np.ndarray，预测结果数据。
    - label_data: np.ndarray，标签数据。
    - len(selected): int，符合条件的病人数量，当return_n_patient=True时返回。
    - selected: List[PatientData]，符合条件的病人数据列表，当return_patients=True时返回。
    """
    def is_blank(val):
        if val == "nan" or val == "None" or val == "" or val == "unkown" or val == "others":
            return True
        if isinstance(val, int) or isinstance(val, float):
            if np.isnan(val):
                return True
        return False
    
    """先按照criterion筛选出patient，再将patient中的sample合并"""
    
    selected = []
    
    for patient_data in patient_data_list:
        patient_data: PatientData = patient_data
        if criterion == "age":
            patient_age = patient_data.basic_info[patient_data.basic_info_items.index("AGE")]
            if (val == "blank"):
                if is_blank(patient_age):
                    selected.append(patient_data)
            elif (patient_age >= val[0] and patient_age < val[1]):
                selected.append(patient_data)
        elif criterion == "gender":
            patient_gender = patient_data.basic_info[patient_data.basic_info_items.index("SEX")]
            if patient_gender == val or (val == "blank" and is_blank(patient_gender)):
                selected.append(patient_data)
        elif criterion == "year":
            patient_op_time = patient_data.op_time
            patient_year = int(time2str(patient_op_time)[:4])
            if patient_year >= val[0] and patient_year < val[1]:
                selected.append(patient_data)
        elif criterion == "center":
            patient_center = patient_data.cen_name
            if patient_center == val:
                selected.append(patient_data)
        elif criterion == "aki_type":
            outcome_lists = patient_data.outcomes
            all_outcomes = []
            for outcome_list in outcome_lists:
                for n,t,et in outcome_list:
                    all_outcomes.append(n)
                
            for outcome in all_outcomes:
                if val in outcome:
                    selected.append(patient_data)
                    break
        elif criterion == "emergency":
            op_time_since_admission = patient_data.op_time - patient_data.admission_time
            if val == "emergency":
                if op_time_since_admission <= 24 * 3600:
                    selected.append(patient_data)
            elif val == "elective":
                if op_time_since_admission > 24 * 3600:
                    selected.append(patient_data)
        elif criterion == "surgery_type":
            surgery_types = patient_data.surgery_type
            # surgery_types 是一个列表，里面有多个手术类型，只需要val匹配到其中一个即可
            for surgery_type in surgery_types:
                if isinstance(surgery_type, str) and val in surgery_type:
                    selected.append(patient_data)
                    break
        else:
            raise NotImplementedError
    
    pred_data = []
    label_data = []
    for patient_data in selected:
        for pred, label in zip(patient_data.predictions, patient_data.labels):
            pred_data.append(pred)
            label_data.append(label)
    
    pred_data = np.array(pred_data)
    label_data = np.array(label_data)
    
    if return_n_patient:
        return pred_data, label_data, len(selected)
    if return_patients:
        return pred_data, label_data, selected
    
    return pred_data, label_data


def sel_by_sample(patient_data_list, criterion="time_in_window", val=[0,4]):
    """
    按照criterion直接筛选出sample
    
    Args:
    - patient_data_list: List[PatientData]，患者数据列表
    - criterion: str，筛选条件，可选值为"time_in_window"
    - val: List[float]，筛选条件的取值范围
    
    Returns:
    - pred_data: np.ndarray，预测数据
    - label_data: np.ndarray，标签数据
    """
    
    pred_data = []
    label_data = []
    
    for patient_data in patient_data_list:
        patient_data: PatientData = patient_data
        for sample_i in range(len(patient_data.data_dynamic)):
            if criterion == "time_in_window":
                timeseries_endpoint = patient_data.prediction_endpoint_list[sample_i]
                closest_outcome_time = np.min([t for n,t,et in patient_data.outcomes[sample_i]])
                time_in_window = (closest_outcome_time - timeseries_endpoint) / 3600
                if time_in_window >= val[0] and time_in_window <= val[1]:
                    pred_data.append(patient_data.predictions[sample_i])
                    label_data.append(patient_data.labels[sample_i])
            else:
                raise NotImplementedError
                
    pred_data = np.array(pred_data)
    label_data = np.array(label_data)
    return pred_data, label_data
    