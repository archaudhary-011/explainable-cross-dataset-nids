"""
plots.py
--------
Standard evaluation visualizations for the Network IDS framework.
Every plot is saved to results/figures/ so it can be directly inserted
into the thesis document.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import seaborn as sns


def plot_roc_curve(y_true, y_proba, model_name: str, cfg: dict):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="firebrick", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()

    from src.utils import save_figure
    save_figure(fig, f"roc_curve_{model_name.replace(' ', '_')}.png", cfg)
    plt.show()
    return roc_auc


def plot_pr_curve(y_true, y_proba, model_name: str, cfg: dict):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    from sklearn.metrics import average_precision_score
    pr_auc = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="steelblue", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
    baseline = y_true.mean()
    ax.axhline(baseline, color="gray", lw=1, linestyle="--",
              label=f"Baseline (class balance = {baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve - {model_name}")
    ax.legend(loc="lower left")
    plt.tight_layout()

    from src.utils import save_figure
    save_figure(fig, f"pr_curve_{model_name.replace(' ', '_')}.png", cfg)
    plt.show()
    return pr_auc


def plot_confusion_matrix(y_true, y_pred, model_name: str, cfg: dict,
                          class_names=("BENIGN", "DDoS")):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=True,
               xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    from src.utils import save_figure
    save_figure(fig, f"confusion_matrix_{model_name.replace(' ', '_')}.png", cfg)
    plt.show()


def plot_model_comparison_bar(results_df, metric: str, cfg: dict):
    """
    Bar chart comparing all models on one metric (e.g. 'f1' or 'roc_auc').
    Expects results_df to have a 'model' column and the metric as a column.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    sorted_df = results_df.sort_values(metric, ascending=True)
    bars = ax.barh(sorted_df["model"], sorted_df[metric], color="teal")
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Model Comparison - {metric.upper()}")
    ax.set_xlim(min(0.99, sorted_df[metric].min() - 0.001), 1.001)
    for bar, val in zip(bars, sorted_df[metric]):
        ax.text(val, bar.get_y() + bar.get_height()/2, f" {val:.5f}",
               va="center", fontsize=9)
    plt.tight_layout()

    from src.utils import save_figure
    save_figure(fig, f"model_comparison_{metric}.png", cfg)
    plt.show()