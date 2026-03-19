"""
Evaluation metrics for the seizure prediction benchmark.

Core metrics:
    accuracy, sensitivity (preictal recall), specificity (interictal recall),
    F1 (weighted + preictal), AUROC, FPR/hour, Time-in-Warning
"""

from typing import Dict, Optional
import numpy as np

try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    window_stride_sec: float = 5.0,     # stride = window length (no overlap in pre-built benchmark)
) -> Dict[str, float]:
    """
    Parameters
    ----------
    y_true : (N,) int  — ground-truth labels (0=interictal, 1=preictal)
    y_pred : (N,) int  — predicted labels
    y_prob : (N,) float — predicted probability for class 1 (optional, for AUROC)
    window_stride_sec : float — seconds between consecutive windows (used for FPR/h)

    Returns
    -------
    dict of metric_name → float
    """
    m: Dict[str, float] = {}

    if _SKLEARN:
        m["accuracy"]    = float(accuracy_score(y_true, y_pred))
        m["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        m["f1_preictal"] = float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0))
        if y_prob is not None and len(np.unique(y_true)) > 1:
            m["auroc"] = float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        else:
            tn = fp = fn = tp = 0
    else:
        correct = int((y_true == y_pred).sum())
        m["accuracy"] = correct / max(len(y_true), 1)
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())

    m["sensitivity"]     = tp / max(tp + fn, 1)   # preictal recall
    m["specificity"]     = tn / max(tn + fp, 1)   # interictal recall
    m["time_in_warning"] = m["sensitivity"]
    m["tp"], m["fp"], m["tn"], m["fn"] = int(tp), int(fp), int(tn), int(fn)

    # FPR/hour: false positives per hour of interictal recording
    interictal_hours = (tn + fp) * window_stride_sec / 3600.0
    m["fpr_per_hour"] = fp / max(interictal_hours, 1e-8)

    return m


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    rows = [
        ("Accuracy",         "accuracy"),
        ("Sensitivity",      "sensitivity"),
        ("Specificity",      "specificity"),
        ("F1 (weighted)",    "f1_weighted"),
        ("F1 (preictal)",    "f1_preictal"),
        ("AUROC",            "auroc"),
        ("FPR / hour",       "fpr_per_hour"),
        ("Time in Warning",  "time_in_warning"),
    ]
    for label, key in rows:
        if key in metrics:
            print(f"{prefix}{label:<20}: {metrics[key]:.4f}")
