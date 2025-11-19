from math import sqrt
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
)
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from jiwer import wer as jiwer_wer, cer as jiwer_cer


def f1_micro(preds, target, task, num_classes): return f1_score(to_labels(target), to_labels(preds), average="micro")
def f1_macro(preds, target, task, num_classes): return f1_score(to_labels(target), to_labels(preds), average="macro")
def f1_weighted(preds, target, task, num_classes): return f1_score(to_labels(target), to_labels(preds), average="weighted")

def top1_acc_micro(preds, target, task, num_classes): return accuracy_score(to_labels(target), to_labels(preds))
def top1_acc_macro(preds, target, task, num_classes):return accuracy_score(to_labels(target), to_labels(preds))
def top1_acc_weighted(preds, target, task, num_classes):return accuracy_score(to_labels(target), to_labels(preds))

def precision_micro(preds, target, task, num_classes):return precision_score(to_labels(target), to_labels(preds), average="micro", zero_division=0)
def precision_macro(preds, target, task, num_classes):return precision_score(to_labels(target), to_labels(preds), average="macro", zero_division=0)
def precision_weighted(preds, target, task, num_classes):return precision_score(target, preds, average="weighted", zero_division=0)

def recall_micro(preds, target, task, num_classes):return recall_score(to_labels(target), to_labels(preds), average="micro", zero_division=0)
def recall_macro(preds, target, task, num_classes):return recall_score(to_labels(target), to_labels(preds), average="macro", zero_division=0)
def recall_weighted(preds, target, task, num_classes):return recall_score(target, preds, average="weighted", zero_division=0)

def precision_recall_curves(preds, target, task='binary', num_classes=2):
    precision, recall, thresholds = precision_recall_curve(y_true=target, y_score=preds)
    return {'precision_curve': precision, 'recall_curve': recall, 'thresholds': thresholds}

def auc_roc_macro(preds, target, task, num_classes):return roc_auc_score(target, preds, average="macro", multi_class="ovr")
def auc_roc_weighted(preds, target, task, num_classes):return roc_auc_score(target, preds, average="weighted", multi_class="ovr")

def dice_micro(preds, target, num_classes):return f1_score(target, preds, average="micro")
def dice_macro(preds, target, num_classes):return f1_score(target, preds, average="macro")

def mae(preds, target):return mean_absolute_error(target, preds)
def rmse(preds, target): return sqrt(mean_squared_error(target, preds))

def ssim(preds, target): return sk_ssim(preds, target, data_range=target.max() - target.min())
def psnr(preds, target): return sk_psnr(preds, target, data_range=target.max() - target.min())

def wer(preds, target): return jiwer_wer(target, preds)
def cer(preds, target): return jiwer_cer(target, preds)


def map(preds, target, iou_type, box_format): return 0
def IoU(preds, target, iou_type, box_format): return intersection_over_union(preds, target, box_format, None, True)



####### classification helpers #########
def to_labels(arr):
    """
    Convert predictions or targets to integer class labels.
    """
    arr = np.array(arr)

    if arr.ndim == 1:
        if np.all(arr == np.floor(arr)): # if all integers
            return arr
        elif np.all((arr >= 0) & (arr <= 1)): # between 0 and 1 binary classification
            return (arr > 0.5).astype(int)
        else:
            raise Exception('Something went wrong')
    if arr.ndim == 2:
        if arr.shape[1] == 1: # between 0 and 1 binary classification
            return (arr[:, 0] > 0.5).astype(int)
        else: # multiclass or 2-class probability array
            return np.argmax(arr, axis=1)
    return arr  # already integer labels


####### iou helpers #######
def _to_xyxy(boxes, fmt):
    boxes = np.asarray(boxes, dtype=float)
    if fmt == "xyxy":
        return boxes
    elif fmt == "xywh":
        x, y, w, h = boxes.T
        return np.stack([x, y, x + w, y + h], axis=1)
    else:
        raise ValueError("Unknown box_format")


def _iou_matrix(preds_xyxy, gt_xyxy):
    if len(preds_xyxy) == 0 or len(gt_xyxy) == 0:
        return np.zeros((len(preds_xyxy), len(gt_xyxy)))

    preds = preds_xyxy[:, None, :]
    gt = gt_xyxy[None, :, :]

    inter_xmin = np.maximum(preds[..., 0], gt[..., 0])
    inter_ymin = np.maximum(preds[..., 1], gt[..., 1])
    inter_xmax = np.minimum(preds[..., 2], gt[..., 2])
    inter_ymax = np.minimum(preds[..., 3], gt[..., 3])

    inter_w = np.maximum(0, inter_xmax - inter_xmin)
    inter_h = np.maximum(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area_preds = (preds[..., 2] - preds[..., 0]) * (preds[..., 3] - preds[..., 1])
    area_gt = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])

    union = area_preds + area_gt - inter_area
    return inter_area / np.clip(union, 1e-9, None)


def intersection_over_union(preds, target, box_format, iou_threshold=None, class_metrics=False):

    all_ious = []
    per_class_ious = {}

    for p, t in zip(preds, target):
        pred_boxes = _to_xyxy(p["boxes"], box_format)
        pred_labels = np.asarray(p["labels"])

        gt_boxes = _to_xyxy(t["boxes"], box_format)
        gt_labels = np.asarray(t["labels"])

        iou_mat = _iou_matrix(pred_boxes, gt_boxes)

        image_ious = []

        for i, gt_label in enumerate(gt_labels):
            pred_mask = (pred_labels == gt_label)

            if np.any(pred_mask):
                best_iou = iou_mat[pred_mask, i].max()
            else:
                best_iou = 0.0

            if iou_threshold is None or best_iou >= iou_threshold:
                image_ious.append(best_iou)

            if class_metrics:
                per_class_ious.setdefault(gt_label, []).append(best_iou)

        all_ious.extend(image_ious)

    result = {}
    result["iou"] = float(np.mean(all_ious)) if all_ious else 0.0

    if class_metrics:
        for cl, vals in per_class_ious.items():
            result[f"iou/cl_{cl}"] = float(np.mean(vals)) if vals else 0.0

    return result