from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float, loss: float | None = None
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.shape != probabilities.shape:
        raise ValueError("labels 与 probabilities 形状不一致。")
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    metrics: dict[str, float | int] = {
        "auc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": specificity,
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
        "samples": int(labels.size),
    }
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics


def format_metrics(split: str, metrics: dict[str, Any]) -> str:
    order = (
        "loss",
        "auc",
        "auprc",
        "mcc",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1_score",
    )
    display_names = {
        "loss": "Loss",
        "auc": "AUC",
        "auprc": "AUPRC",
        "mcc": "MCC",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "specificity": "Specificity",
        "f1_score": "F1-score",
    }
    parts = [
        f"{display_names[name]}={float(metrics[name]):.6f}" for name in order if name in metrics
    ]
    return f"{split}: " + " | ".join(parts)
