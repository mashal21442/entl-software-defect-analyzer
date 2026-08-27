from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def calculate_metrics(
    y_true,
    probabilities,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities).astype(float)
    predictions = (probabilities >= threshold).astype(int)

    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    pd_value = tp / (tp + fn) if (tp + fn) else 0.0
    pf_value = fp / (fp + tn) if (fp + tn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    g_mean = float(np.sqrt(pd_value * specificity))
    f1 = f1_score(y_true, predictions, zero_division=0)

    try:
        auc = roc_auc_score(y_true, probabilities)
    except ValueError:
        auc = float("nan")

    return {
        "PD": float(pd_value),
        "PF": float(pf_value),
        "F1-score": float(f1),
        "G-Mean": float(g_mean),
        "AUC": float(auc),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }
