import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, f1_score


def compute_eer(y_true, y_scores):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    if len(fpr) < 2:
        return float(0.5)
    try:
        from scipy.optimize import brentq
        from scipy.interpolate import interp1d
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        return float(eer)
    except Exception:
        idx = int(np.nanargmin(np.abs(fnr - fpr)))
        return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_min_tdcf_stub():
    return float("nan")


def summarize(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "eer": compute_eer(y_true, y_prob),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "min_tdcf": compute_min_tdcf_stub(),
        "threshold": float(threshold),
    }
