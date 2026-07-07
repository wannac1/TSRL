from sklearn import metrics
import numpy as np


def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str= str+ f"| {key}: "
            for k,v in value.items():
                str = str + f" {k}={v} "
            str= str+ "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key,value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str



def get_more_test_metrics(y_pred, y_true, img_names=None):
    """
    计算并返回分类模型的评估指标：准确率、精确率、召回率、F1-Score、混淆矩阵及ROC曲线相关数据。

    参数:
    - y_pred: 模型的预测概率（浮动值）。
    - y_true: 真实标签（二值）。
    - img_names: 可选，图像名称的列表。若提供，则在输出中附加每个图像的相关信息。

    返回:
    - metrics_dict: 包含所有计算结果的字典
    """

    # 假设阈值为 0.5, 将预测概率转换为二分类标签
    threshold = 0.5
    y_pred_bin = (y_pred >= threshold).astype(int)

    # 计算混淆矩阵
    cm = metrics.confusion_matrix(y_true, y_pred_bin)

    # 计算精确率（Precision）
    precision = metrics.precision_score(y_true, y_pred_bin)

    # 计算召回率（Recall）
    recall = metrics.recall_score(y_true, y_pred_bin)

    # 计算准确率（Accuracy）
    accuracy = metrics.accuracy_score(y_true, y_pred_bin)

    # 计算F1-Score
    f1 = metrics.f1_score(y_true, y_pred_bin)

    # 准备结果字典
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': cm,
    }

    return metrics_dict

def get_test_metrics(y_pred, y_true, img_names=None, more_metric=True):
    from sklearn import metrics
    import numpy as np

    def get_video_metrics(image_paths, preds, labels):
        result_dict = {}
        new_preds = []
        new_labels = []

        for path_or_list, pred, label in zip(image_paths, preds, labels):
            
            # --- ▼▼▼ 【【【核心修复】】】 ▼▼▼ ---
            # 检查 'path_or_list' 是列表 (video_level) 还是字符串 (image_level)
            if isinstance(path_or_list, list):
                # 如果是列表 (视频/clip 模式)，取第一个帧的路径
                # 因为一个 clip 里的所有帧都属于同一个视频
                if not path_or_list:
                    continue # 跳过空列表
                path = path_or_list[0] 
            else:
                # 如果是字符串 (图像模式)，直接使用
                path = path_or_list
            
            if not isinstance(path, str):
                 # 添加一个安全检查，以防传入的是 Tensor 或其他非字符串类型
                 print(f"Warning: Skipping non-string path in get_video_metrics: {type(path)}")
                 continue
            # --- ▲▲▲ 【【【修复结束】】】 ▲▲▲

            parts = path.split('\\') if '\\' in path else path.split('/') #
            video_id = parts[-2] if len(parts) >= 2 else 'unknown'

            result_dict.setdefault(video_id, []).append((float(pred), int(label)))

        for frames in result_dict.values():
            preds_video = [p for p, _ in frames]
            labels_video = [l for _, l in frames]
            new_preds.append(np.mean(preds_video))
            new_labels.append(int(round(np.mean(labels_video))))

        fpr, tpr, _ = metrics.roc_curve(new_labels, new_preds, pos_label=1)
        v_auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        v_eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
        return v_auc, v_eer

    # Sanitize input
    y_pred = np.asarray(y_pred).squeeze()
    y_true = np.asarray(y_true).squeeze()
    y_true = np.clip(y_true, 0, 1)

    # Frame-level metrics
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    ap = metrics.average_precision_score(y_true, y_pred)
    acc = ((y_pred > 0.5).astype(int) == y_true).sum() / len(y_true)

    # Video-level metrics
    if img_names is not None:
        v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
    else:
        v_auc = auc  # fallback if no image names

    # Assemble results
    res_dict = {
        'acc': acc,
        'auc': auc,
        'eer': eer,
        'ap': ap,
        'pred': y_pred,
        'v_auc': v_auc,
        'label': y_true
    }

    if more_metric:
        res_dict.update(get_more_test_metrics(y_pred, y_true))

    return res_dict
