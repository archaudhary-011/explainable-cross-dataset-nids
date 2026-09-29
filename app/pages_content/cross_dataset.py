"""
cross_dataset.py
-----------------
Page 4: Cross-Dataset Generalization.

Displays results for ALL THREE target datasets tested (matching the
official project scope): CSE-CIC-IDS2018, UNSW-NB15, and CICDDoS2019.
Loads results/reports/cross_dataset_all_results.json - one file
containing every cross-dataset test result, saved by the corresponding
notebooks (11, 13, 15).
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def render(cfg: dict):
    st.title("4. Cross-Dataset Generalization")
    st.markdown(
        "The CICIDS2017-trained model was tested on **three independent external "
        "datasets** to measure real-world generalization, matching the official "
        "project scope (train on Dataset A, test on Datasets B, C, D)."
    )

    reports_dir = cfg["paths"]["reports_dir"]
    metrics_dir = cfg["paths"]["metrics_dir"]
    figures_dir = cfg["paths"]["figures_dir"]
    results_path = os.path.join(reports_dir, "cross_dataset_all_results.json")

    if not os.path.exists(results_path):
        st.error(f"Cross-dataset results not found at:\n{results_path}")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    in_dataset_path = os.path.join(metrics_dir, "final_model_metrics.csv")
    in_dataset_row = None
    if os.path.exists(in_dataset_path):
        in_dataset_row = pd.read_csv(in_dataset_path).iloc[0]

    dataset_order = ["cse_cic_ids2018", "unsw_nb15", "cicddos2019"]

    # --- At-a-glance F1 comparison chart: in-dataset vs all 3 targets ---
    st.subheader("At a Glance: The Generalization Cliff")

    chart_labels = []
    chart_f1 = []
    chart_colors = []

    if in_dataset_row is not None:
        chart_labels.append("CICIDS2017\n(in-dataset)")
        chart_f1.append(in_dataset_row["f1"])
        chart_colors.append("#2ca02c")  # green

    for key in dataset_order:
        if key in all_results:
            r = all_results[key]
            short_name = {
                "cse_cic_ids2018": "CSE-CIC-\nIDS2018",
                "unsw_nb15": "UNSW-\nNB15",
                "cicddos2019": "CICDDoS\n2019",
            }[key]
            chart_labels.append(short_name)
            chart_f1.append(r["f1"])
            chart_colors.append("#d62728")  # red

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(chart_labels, chart_f1, color=chart_colors)
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.set_title("F1-score: In-Dataset vs Cross-Dataset Targets")
    for bar, val in zip(bars, chart_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.4f}",
                ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Random Forest at the default 0.5 threshold (20 features; UNSW-NB15 uses the auxiliary "
        "7-feature model). Green = held-out CICIDS2017 test data. Red = unseen externa "
        "datasets."
    )

    st.markdown("---")

    # --- Summary table ---
    st.subheader("Summary Table: Full Metric Set")

    rows = []
    if in_dataset_row is not None:
        rows.append({
            "Dataset": "CICIDS2017 (in-dataset)",
            "Feature Overlap": "20/20 (100%)",
            "Accuracy": in_dataset_row["accuracy"],
            "Precision": in_dataset_row["precision"],
            "Recall": in_dataset_row["recall"],
            "F1-score": in_dataset_row["f1"],
            "ROC-AUC": in_dataset_row["roc_auc"],
            "FPR": in_dataset_row["fpr"],
        })

    for key in dataset_order:
        if key in all_results:
            r = all_results[key]
            rows.append({
                "Dataset": r["dataset_name"],
                "Feature Overlap": r["feature_overlap"],
                "Accuracy": r["accuracy"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1-score": r["f1"],
                "ROC-AUC": r["roc_auc"],
                "FPR": r["fpr"],
            })

    summary_df = pd.DataFrame(rows)
    num_cols = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "FPR"]
    st.dataframe(
        summary_df.style.format({c: "{:.4f}" for c in num_cols}),
        width="stretch",
    )

    st.error(
        "**The model failed to generalize on all three external datasets.** "
        "Every target shows FNR ≈ 1.0000 (near-total miss rate on attack traffic), "
        "regardless of feature overlap or shared tooling with the source domain."
    )

    st.markdown("---")

    # --- Detailed tabs per dataset ---
    st.subheader("Detailed Results Per Dataset")
    tab_labels = [all_results[k]["dataset_name"] for k in dataset_order if k in all_results]
    tabs = st.tabs(tab_labels)

    for tab, key in zip(tabs, [k for k in dataset_order if k in all_results]):
        r = all_results[key]
        with tab:
            row1 = st.columns(4)
            row1[0].metric("Accuracy", f"{r['accuracy']:.4f}")
            row1[1].metric("Precision", f"{r['precision']:.4f}")
            row1[2].metric("Recall", f"{r['recall']:.4f}")
            row1[3].metric("F1-score", f"{r['f1']:.4f}")

            row2 = st.columns(4)
            row2[0].metric("ROC-AUC", f"{r['roc_auc']:.4f}")
            row2[1].metric("PR-AUC", f"{r['pr_auc']:.4f}")
            row2[2].metric("FPR", f"{r['fpr']:.4f}")
            row2[3].metric("FNR", f"{r['fnr']:.4f}")

            st.markdown(
                f"**Feature overlap:** {r['feature_overlap']}  |  "
                f"**Same extraction tool:** {'Yes' if r['same_tool'] else 'No'} ({r['tool_used']})"
            )
            st.markdown(f"**Test samples:** {r['n_samples']:,}")

            cm_data = pd.DataFrame(
                [[r["tn"], r["fp"]], [r["fn"], r["tp"]]],
                index=["Actual BENIGN", "Actual Attack"],
                columns=["Predicted BENIGN", "Predicted Attack"],
            )
            st.dataframe(cm_data, width="stretch")

            st.info(f"**Note:** {r['notes']}")

    st.markdown("---")

    # --- Domain shift section: all three targets as tabs ---
    st.subheader("Why did this happen? Domain Shift Analysis (all three targets)")
    st.markdown(
        "The model relies heavily on packet-size features (see Page 3 / SHAP analysis). "
        "The Kolmogorov-Smirnov test below quantifies how far each target domain's "
        "feature distributions have shifted from the training domain's - for the "
        "attack class specifically, since that is where every model failed."
    )

    shift_files = {
        "cse_cic_ids2018": ("domain_shift_ddos.csv", "domain_shift_top_feature.png", "CSE-CIC-IDS2018"),
        "unsw_nb15": ("domain_shift_unsw_nb15.csv", "domain_shift_unsw_top_feature.png", "UNSW-NB15"),
        "cicddos2019": ("domain_shift_cicddos2019.csv", "domain_shift_cicddos2019_top_feature.png", "CICDDoS2019"),
    }

    shift_tab_labels = [v[2] for k, v in shift_files.items() if os.path.exists(os.path.join(reports_dir, v[0]))]
    if shift_tab_labels:
        shift_tabs = st.tabs(shift_tab_labels)

        tab_idx = 0
        for key, (csv_name, plot_name, label) in shift_files.items():
            csv_path = os.path.join(reports_dir, csv_name)
            if not os.path.exists(csv_path):
                continue
            with shift_tabs[tab_idx]:
                shift_df = pd.read_csv(csv_path)
                n_sig = shift_df["significant_shift"].sum()
                st.caption(f"{n_sig} of {len(shift_df)} features show statistically significant shift (KS test, p < 0.05).")
                st.dataframe(
                    shift_df[["feature", "source_mean", "target_mean", "ks_statistic",
                             "wasserstein_normalized", "significant_shift"]].head(10),
                    width="stretch",
                )
                if plot_name:
                    plot_path = os.path.join(figures_dir, plot_name)
                    if os.path.exists(plot_path):
                        st.image(plot_path, width="stretch")
            tab_idx += 1

    st.markdown("---")
    st.markdown("---")
    st.subheader("All 4 Models Compared Cross-Dataset")
    heatmap_path = os.path.join(figures_dir, "multi_model_cross_dataset_heatmap.png")
    if os.path.exists(heatmap_path):
        st.image(heatmap_path, width="stretch")
    thr_path = os.path.join(metrics_dir, "threshold_analysis_all_models.csv")
    if os.path.exists(thr_path):
        st.markdown("**Threshold analysis** (best F1 vs the trivial 'flag everything' baseline; "
                    "threshold chosen with target labels, diagnostic only):")
        st.dataframe(pd.read_csv(thr_path).drop(columns=["best_threshold"]), width="stretch")
    st.subheader("Research Interpretation")
    st.markdown(
        """
        The final Random Forest reaches F1 = 0.9999 in-dataset but F1 = 0.0000 on all three
        external datasets at the default threshold (FNR = 1.0).

        A multi-model and threshold analysis (4 models; oracle threshold, diagnostic only) refines this:

        - **CICDDoS2019 and UNSW-NB15:** no model, at any threshold, meaningfully beats labelling
        everything as attack (gain over baseline at most 0.02). No usable transferable signal was found.
        - **CSE-CIC-IDS2018:** Random Forest and Decision Tree gain nothing, but Logistic Regression
        (best F1 0.991) and XGBoost (0.943) keep strong ranking ability (ROC-AUC 0.986 / 0.960)
        against a baseline F1 of 0.527. Their failure at the default threshold is largely miscalibration.

        High in-dataset accuracy did not carry over to any external dataset at the default operating
        point, and for the final Random Forest no threshold recovers it. Why some models transfer to 2018
        and others do not was not determined in this study.
        """
    )
