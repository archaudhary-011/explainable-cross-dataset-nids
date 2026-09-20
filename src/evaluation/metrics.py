"""
metrics.py
----------
Shared evaluation utilities for the Network IDS framework.

Using ONE metrics module everywhere guarantees that Experiment 1 (baseline),
Experiment 2 (model comparison), Experiment 5 (cross-dataset), etc. are all
measured identically and can be fairly compared in the final report.
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """
    Computes the full metric set required by the project spec:
    Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FPR, FNR,
    plus the raw confusion matrix components.

    y_proba must be the predicted probability of the POSITIVE class (DDoS=1).
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "fpr": fpr,
        "fnr": fnr,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return metrics


def evaluate_model(model, X_test, y_test, model_name: str = "model") -> dict:
    """
    Runs prediction, times it, and computes all metrics.
    Returns a single flat dict ready to be appended to a results table.
    """
    start_pred = time.time()
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback for models without predict_proba (rare in this project)
        y_proba = y_pred.astype(float)
    pred_time = time.time() - start_pred

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics["model"] = model_name
    metrics["prediction_time_sec"] = round(pred_time, 4)

    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-prints a single model's metrics block."""
    print("=" * 60)
    print(f"RESULTS: {metrics.get('model', 'model')}")
    print("=" * 60)
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"F1-score   : {metrics['f1']:.4f}")
    print(f"ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC     : {metrics['pr_auc']:.4f}")
    print(f"FPR        : {metrics['fpr']:.4f}")
    print(f"FNR        : {metrics['fnr']:.4f}")
    print("-" * 60)
    print("Confusion Matrix:")
    print(f"              Predicted BENIGN   Predicted DDoS")
    print(f"Actual BENIGN      {metrics['tn']:>8,}          {metrics['fp']:>8,}")
    print(f"Actual DDoS        {metrics['fn']:>8,}          {metrics['tp']:>8,}")
    print("-" * 60)
    if "train_time_sec" in metrics:
        print(f"Training time    : {metrics['train_time_sec']:.4f} sec")
    print(f"Prediction time  : {metrics['prediction_time_sec']:.4f} sec")
    print("=" * 60)


def results_to_dataframe(results_list: list) -> pd.DataFrame:
    """
    Converts a list of metric dicts (one per model) into a tidy comparison
    table, ordered by F1-score descending. This is the table format
    required by the project spec (Model / Accuracy / Precision / ... ).
    """
    df = pd.DataFrame(results_list)
    col_order = ["model", "accuracy", "precision", "recall", "f1",
                 "roc_auc", "pr_auc", "fpr", "fnr",
                 "train_time_sec", "prediction_time_sec"]
    col_order = [c for c in col_order if c in df.columns]
    remaining = [c for c in df.columns if c not in col_order]
    df = df[col_order + remaining]
    return df.sort_values("f1", ascending=False).reset_index(drop=True)