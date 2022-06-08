import torch
import numpy as np


def accuracy(y_pred, y_true):
    stage, y_true = parse(y_pred, y_true)
    return np.mean(stage == y_true)


def precision(y_pred, y_true):
    stage, y_true = parse(y_pred, y_true)
    tp, fp, _, _ = separate_count(stage, y_true)
    return divide(tp, tp + fp)


def recall(y_pred, y_true):
    stage, y_true = parse(y_pred, y_true)
    tp, _, fn, _ = separate_count(stage, y_true)
    return divide(tp, tp + fn)


def f1(y_pred, y_true):
    pre = precision(y_pred, y_true)
    rec = recall(y_pred, y_true)
    return divide(2 * pre * rec, pre + rec)


def DSC(y_pred, y_true):
    stage = y_pred
    stage = stage.cpu().detach().numpy()[:, 1, :, :]
    y_true = np.array(y_true.cpu())

    return 2 * (stage * y_true).sum() / (y_true.sum() + stage.sum())


def HM(y_pred, y_true):
    stage, y_true = parse(y_pred, y_true)
    tp, fp, fn, _ = separate_count(stage, y_true)
    union = tp + fp + fn
    return divide(union - tp, union)


def IOU(y_pred, y_true):
    y_true = np.array(y_true.cpu())
    stage = y_pred
    stage = np.array(stage.detach().cpu())[:, 1, :, :]
    return divide((y_true * stage).sum(), ((y_true == 1) | (stage > 0.5)).sum())


def parse(y_pred, y_true):
    stage = y_pred
    # stage = y_pred[0]
    stage = torch.argmax(stage, dim=1)
    # 转换为np数组用于计算
    stage = np.array(stage.cpu())
    y_true = np.array(y_true.cpu())
    return stage, y_true


def separate_count(stage, y_true):
    tp = np.sum((stage == y_true) & (y_true == 1))
    fp = np.sum((stage != y_true) & (y_true == 0))
    fn = np.sum((stage != y_true) & (y_true == 1))
    tn = np.sum((stage == y_true) & (y_true == 0))
    return tp, fp, fn, tn


def divide(dividend, divisor):
    return dividend / divisor if divisor != 0 else 0.0
