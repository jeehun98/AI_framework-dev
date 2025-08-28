import numpy as np
from typing import Literal, Tuple, Optional

# -----------------------------
# Classification metrics
# -----------------------------
def accuracy_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return (y_true == y_pred).mean()

def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    lab_to_idx = {lab:i for i, lab in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        mat[lab_to_idx[t], lab_to_idx[p]] += 1
    return mat

def _precision_recall_f1_binary(y_true, y_pred, pos_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    return precision, recall, f1

def precision_score(y_true, y_pred, average: Literal["binary","macro"]="binary", pos_label=1):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if average == "binary":
        return _precision_recall_f1_binary(y_true, y_pred, pos_label)[0]
    # macro
    classes = np.unique(y_true)
    vals = [_precision_recall_f1_binary(y_true==c, y_pred==c, True)[0] for c in classes]
    return float(np.mean(vals))

def recall_score(y_true, y_pred, average: Literal["binary","macro"]="binary", pos_label=1):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if average == "binary":
        return _precision_recall_f1_binary(y_true, y_pred, pos_label)[1]
    classes = np.unique(y_true)
    vals = [_precision_recall_f1_binary(y_true==c, y_pred==c, True)[1] for c in classes]
    return float(np.mean(vals))

def f1_score(y_true, y_pred, average: Literal["binary","macro"]="binary", pos_label=1):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if average == "binary":
        return _precision_recall_f1_binary(y_true, y_pred, pos_label)[2]
    classes = np.unique(y_true)
    vals = [_precision_recall_f1_binary(y_true==c, y_pred==c, True)[2] for c in classes]
    return float(np.mean(vals))

def roc_auc_score(y_true, y_score, average: Optional[str]=None):
    """
    Binary ROC-AUC (y_true in {0,1}). y_score: 연속 점수(확률/로짓/decision_function).
    다중분류는 OvR로 macro-avg 하려면 average="macro" + y_score shape=(n, C) 지원.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # OvR (macro) for multi-class if 2D scores passed
    if y_score.ndim == 2 and average == "macro":
        C = y_score.shape[1]
        classes = np.unique(y_true)
        # align classes to column indices 0..C-1
        aucs = []
        for c in classes:
            y_bin = (y_true == c).astype(int)
            aucs.append(_roc_auc_binary(y_bin, y_score[:, int(c)]))
        return float(np.mean(aucs))

    # Binary
    return _roc_auc_binary(y_true.astype(int), y_score)

def _roc_auc_binary(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    # sort by score desc
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    # accumulate TPR/FPR
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    if P == 0 or N == 0:
        # 정의 불가 → 관습적으로 0.5 반환
        return 0.5

    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)

    tpr = tps / P
    fpr = fps / N

    # trapezoidal rule; prepend (0,0)
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    return float(np.trapz(tpr, fpr))

# -----------------------------
# Regression metrics
# -----------------------------
def mean_squared_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return float(np.mean(diff*diff))

def mean_absolute_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res/ss_tot if ss_tot != 0 else 0.0
