"""Scores over a fixed full-evaluation class set, retained for later bootstrap."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    classes: tuple[int, ...] | None = None,
    known_classes: tuple[int, ...] | None = None,
) -> dict:
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    if (
        true.ndim != 1
        or not len(true)
        or pred.shape != true.shape
        or true.dtype.kind not in "iu"
        or pred.dtype.kind not in "iu"
    ):
        raise ValueError("Metric labels must be aligned nonempty integer vectors")
    labels = tuple(int(v) for v in np.unique(true)) if classes is None else tuple(classes)
    if not labels or tuple(sorted(set(labels))) != labels or not set(true) <= set(labels):
        raise ValueError("Fixed macro class set must contain every observed true class")
    known = (
        tuple(sorted(set(labels) | set(pred))) if known_classes is None else tuple(known_classes)
    )
    if not (set(true) | set(pred) | set(labels)) <= set(known):
        raise ValueError("Predictions or evaluation labels are outside the canonical mapping")
    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        pred,
        labels=list(labels),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(np.mean(f1)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_classes": list(labels),
        "absent_target_classes": sorted(set(known) - set(labels)),
        "per_class": [
            {
                "class": label,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(n),
            }
            for label, p, r, f, n in zip(labels, precision, recall, f1, support, strict=True)
        ],
        "confusion_matrix_classes": list(known),
        "confusion_matrix": confusion_matrix(true, pred, labels=list(known)).tolist(),
        "sample_count": len(true),
        "zero_division": 0,
        "class_policy": "present_in_full_target_test",
    }
