from math import sqrt
import numpy as np
import fairbench as fb
from fairbench.v2.core import sensitive
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve as pr_curve,
)
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from jiwer import wer as jiwer_wer, cer as jiwer_cer
from od_metrics import ODMetrics
odmetrics = ODMetrics()

def f1_micro(preds, target, task, num_classes): return roundx(f1_score(to_labels(target), to_labels(preds), average="micro"))
def f1_macro(preds, target, task, num_classes): return roundx(f1_score(to_labels(target), to_labels(preds), average="macro"))
def f1_weighted(preds, target, task, num_classes): return roundx(f1_score(to_labels(target), to_labels(preds), average="weighted"))

def top1_acc_micro(preds, target, task, num_classes): return roundx(accuracy_score(to_labels(target), to_labels(preds)))
def top1_acc_macro(preds, target, task, num_classes):return roundx(accuracy_score(to_labels(target), to_labels(preds)))
def top1_acc_weighted(preds, target, task, num_classes):return roundx(accuracy_score(to_labels(target), to_labels(preds)))

def precision_micro(preds, target, task, num_classes):return roundx(precision_score(to_labels(target), to_labels(preds), average="micro", zero_division=0))
def precision_macro(preds, target, task, num_classes):return roundx(precision_score(to_labels(target), to_labels(preds), average="macro", zero_division=0))
def precision_weighted(preds, target, task, num_classes):return roundx(precision_score(target, preds, average="weighted", zero_division=0))

def recall_micro(preds, target, task, num_classes):return roundx(recall_score(to_labels(target), to_labels(preds), average="micro", zero_division=0))
def recall_macro(preds, target, task, num_classes):return roundx(recall_score(to_labels(target), to_labels(preds), average="macro", zero_division=0))
def recall_weighted(preds, target, task, num_classes):return roundx(recall_score(target, preds, average="weighted", zero_division=0))

def _fairness_measure(schema, preds, target, task, num_classes, sensitive, unpack_categorical_columns=False):
    preds = to_labels(preds)
    target = to_labels(target)
    if not sensitive: raise TypeError("no sensitive attributes")
    sensitive_dims = fb.Dimensions({k: fb.categories @ v for k,v in sensitive.items()} if unpack_categorical_columns else sensitive)
    try:
        values: list[float] = list()
        for num_class in range(num_classes):
            value = fb.quick.__getattr__(schema)(
                predictions=preds==num_class,
                labels=target==num_class,
                sensitive=sensitive_dims)
            #value.show() # for debugging
            values.append(float(value))
            return min(values)
    except fb.core.NotComputable:
        raise TypeError("no valid fairness computation")

def min_prule(preds, target, task, num_classes, sensitive):
    return roundx(1-_fairness_measure("pairwise_maxrel_pr", preds, target, task, num_classes, sensitive))

def max_dfpr(preds, target, task, num_classes, sensitive):
    return roundx(_fairness_measure("pairwise_maxdiff_tnr", preds, target, task, num_classes, sensitive))

def max_dfnr(preds, target, task, num_classes, sensitive):
    return roundx(_fairness_measure("pairwise_maxdiff_tpr", preds, target, task, num_classes, sensitive))


def precision_recall_curve(preds, target, task='binary', num_classes=2):
    precision, recall, thresholds = pr_curve(y_true=target, y_score=preds)
    return roundx({'precision_curve': precision, 'recall_curve': recall, 'thresholds': thresholds})

def auc_roc_macro(preds, target, task, num_classes):return roundx(roc_auc_score(target, preds, average="macro", multi_class="ovr"))
def auc_roc_weighted(preds, target, task, num_classes):return roundx(roc_auc_score(target, preds, average="weighted", multi_class="ovr"))

def dice_micro(preds, target, num_classes):return roundx(f1_score(target, preds, average="micro"))
def dice_macro(preds, target, num_classes):return roundx(f1_score(target, preds, average="macro"))

def mae(preds, target):return roundx(mean_absolute_error(target, preds))
def rmse(preds, target): return roundx(sqrt(mean_squared_error(target, preds)))

def ssim(preds, target): return roundx(sk_ssim(preds, target, data_range=target.max() - target.min()))
def psnr(preds, target): return roundx(sk_psnr(preds, target, data_range=target.max() - target.min()))

def wer(preds, target): return roundx(jiwer_wer(target, preds))
def cer(preds, target): return roundx(jiwer_cer(target, preds))

def iou(): return None

def od_metrics(preds, target): 
    odmetrics = ODMetrics()
    return roundx(odmetrics.compute(target, preds))



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


def roundx(value, x=3):
    """Round to x decimals (handles floats, numpy, dicts, arrays)."""
    if isinstance(value, dict):
        return {k: roundx(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.round(value, x)
    return round(float(value), x)