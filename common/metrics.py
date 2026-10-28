"""Binary classification metrics used by every baseline adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the threshold maximizing F1 on the provided validation set."""

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    denominator = precision + recall
    f1_scores = np.where(
        denominator > 0,
        2 * precision * recall / np.where(denominator > 0, denominator, 1),
        0.0,
    )
    index = int(np.argmax(f1_scores))
    return float(thresholds[index]) if index < len(thresholds) else 0.5


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute AUPRC, ROC-AUC, Recall, F1, and thresholded metrics."""

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty set")

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    metrics: dict[str, float] = {
        "threshold": float(threshold),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) == 2
        else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }
    return metrics


def save_metrics(metrics: dict[str, float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)


def save_predictions(
    ids: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    path: str | Path,
    threshold: float = 0.5,
) -> None:
    from .data import ID2LABEL

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("id,gold_label,gold,pred_label,pred,prob\n")
        for row_id, gold, pred, prob in zip(ids, y_true, predictions, y_prob):
            handle.write(
                f"{row_id},{ID2LABEL[int(gold)]},{int(gold)},"
                f"{ID2LABEL[int(pred)]},{int(pred)},{float(prob):.6f}\n"
            )
