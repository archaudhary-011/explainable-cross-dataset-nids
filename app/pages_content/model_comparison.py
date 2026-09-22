"""
model_comparison.py
--------------------
Page 2: Model Comparison.
Loads the saved comparison tables from Steps 5-7 and displays them as
tables and bar charts, plus the feature-selection before/after result.
"""

import os
import pandas as pd
import streamlit as st


def render(cfg: dict):
    st.title("2. Model Comparison")
    st.markdown(
        "Comparison of all trained models: Logistic Regression, Decision Tree "
        "(Step 4 baselines), Random Forest and XGBoost (Step 5 advanced models)."
    )

    metrics_dir = cfg["paths"]["metrics_dir"]
    all_results_path = os.path.join(metrics_dir, "all_models_comparison.csv")

    if not os.path.exists(all_results_path):
        st.error(f"Comparison results not found at:\n{all_results_path}\nRun notebooks/05_model_comparison.ipynb first.")
        return

    all_results = pd.read_csv(all_results_path)

    st.subheader("Full comparison table")
    display_cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                    "fpr", "fnr", "train_time_sec", "prediction_time_sec"]
    display_cols = [c for c in display_cols if c in all_results.columns]
    st.dataframe(
        all_results[display_cols].style.format({
            c: "{:.4f}" for c in display_cols if c not in ["model"]
        }),
        width="stretch",
    )

    st.subheader("F1-score comparison")
    f1_sorted = all_results.set_index("model")["f1"].sort_values()
    st.bar_chart(f1_sorted)

    st.subheader("Training vs prediction time (seconds)")
    time_df = all_results.set_index("model")[["train_time_sec", "prediction_time_sec"]]
    st.bar_chart(time_df)

    st.markdown("---")
    st.subheader("Feature Selection: Before vs After (Step 6)")

    fs_path = os.path.join(metrics_dir, "feature_selection_comparison.csv")
    if os.path.exists(fs_path):
        fs_results = pd.read_csv(fs_path)
        st.dataframe(
            fs_results[["model", "accuracy", "f1", "roc_auc", "train_time_sec", "prediction_time_sec"]],
            width="stretch",
        )
        st.caption(
            "Reducing from 67 to 20 features cost negligible performance "
            "while cutting training time substantially."
        )
    else:
        st.info("Feature selection comparison not found. Run notebooks/06_feature_selection.ipynb first.")

    st.markdown("---")
    st.subheader("Final Tuned Model (Step 7)")

    final_path = os.path.join(metrics_dir, "final_model_metrics.csv")
    if os.path.exists(final_path):
        final_results = pd.read_csv(final_path)
        st.dataframe(final_results, width="stretch")

        hp_path = os.path.join(cfg["paths"]["reports_dir"], "final_model_hyperparameters.json")
        if os.path.exists(hp_path):
            import json
            with open(hp_path) as f:
                hyperparams = json.load(f)
            with st.expander("Best hyperparameters found (RandomizedSearchCV)"):
                st.json(hyperparams)
    else:
        st.info("Final model metrics not found. Run notebooks/07_final_model.ipynb first.")