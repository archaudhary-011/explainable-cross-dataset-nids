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
    results_path = os.path.join(reports_dir, "cross_dataset_all_results.json")

    if not os.path.exists(results_path):
        st.error(f"Cross-dataset results not found at:\n{results_path}")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    in_dataset_path = os.path.join(metrics_dir, "final_model_metrics.csv")
    in_dataset_f1 = None
    if os.path.exists(in_dataset_path):
        in_dataset_f1 = pd.read_csv(in_dataset_path).iloc[0]["f1"]

    # --- Summary table across all three targets ---
    st.subheader("Summary: In-Dataset vs All Cross-Dataset Targets")

    rows = []
    if in_dataset_f1 is not None:
        rows.append({
            "Dataset": "CICIDS2017 (in-dataset, held-out test set)",
            "Feature Overlap": "20/20 (100%)",
            "Same Tool": "—",
            "F1-score": in_dataset_f1,
            "ROC-AUC": None,
            "FNR": None,
        })

    dataset_order = ["cse_cic_ids2018", "unsw_nb15", "cicddos2019"]
    for key in dataset_order:
        if key in all_results:
            r = all_results[key]
            rows.append({
                "Dataset": r["dataset_name"],
                "Feature Overlap": r["feature_overlap"],
                "Same Tool": "Yes" if r["same_tool"] else "No",
                "F1-score": r["f1"],
                "ROC-AUC": r["roc_auc"],
                "FNR": r["fnr"],
            })

    summary_df = pd.DataFrame(rows)
    st.dataframe(
        summary_df.style.format({"F1-score": "{:.4f}", "ROC-AUC": "{:.4f}", "FNR": "{:.4f}"}, na_rep="—"),
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
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("F1-score", f"{r['f1']:.4f}")
            col2.metric("ROC-AUC", f"{r['roc_auc']:.4f}")
            col3.metric("FPR", f"{r['fpr']:.4f}")
            col4.metric("FNR", f"{r['fnr']:.4f}")

            st.markdown(f"**Feature overlap:** {r['feature_overlap']}  |  **Same extraction tool:** {'Yes' if r['same_tool'] else 'No'} ({r['tool_used']})")
            st.markdown(f"**Test samples:** {r['n_samples']:,}")

            cm_data = pd.DataFrame(
                [[r["tn"], r["fp"]], [r["fn"], r["tp"]]],
                index=["Actual BENIGN", "Actual Attack"],
                columns=["Predicted BENIGN", "Predicted Attack"],
            )
            st.dataframe(cm_data, width="stretch")

            st.info(f"**Note:** {r['notes']}")

    st.markdown("---")
    st.subheader("Why did this happen? Domain Shift Analysis (all three targets)")
    st.markdown(
        "The model relies heavily on packet-size features (see Page 3 / SHAP analysis). "
        "The Kolmogorov-Smirnov test below quantifies how far each target domain's "
        "feature distributions have shifted from the training domain's - for the "
        "attack class specifically, since that is where every model failed."
    )

    figures_dir = cfg["paths"]["figures_dir"]

    shift_files = {
        "cse_cic_ids2018": ("domain_shift_ddos.csv", "domain_shift_top_feature.png", "CSE-CIC-IDS2018"),
        "unsw_nb15": ("domain_shift_unsw_nb15.csv", None, "UNSW-NB15"),
        "cicddos2019": ("domain_shift_cicddos2019.csv", None, "CICDDoS2019"),
    }

    shift_tab_labels = [v[2] for k, v in shift_files.items() if os.path.exists(os.path.join(reports_dir, v[0]))]
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
    st.subheader("Research Interpretation")
    st.markdown(
        """
        This project tested cross-dataset generalization against **three independent
        external datasets**, deliberately chosen to vary in feature-extraction tooling
        and attack methodology:

        - **CSE-CIC-IDS2018**: same tool (CICFlowMeter), 100% feature overlap — isolates
          the effect of *attack-signature drift alone*.
        - **UNSW-NB15**: different tool (Argus/Bro-IDS), only 35% feature overlap, with
          2 of those 7 "common" features later found to be semantically mismatched —
          isolates the effect of *tooling and measurement heterogeneity*.
        - **CICDDoS2019**: same tool, 100% feature overlap — a second same-tool test;
          some rank-ordering signal survived (ROC-AUC = 0.698) but no usable operating
          point existed at any decision threshold.

        **All three failed completely** (FNR ≈ 1.0, best achievable F1 ≈ 0 in every case),
        despite the model achieving 99.99% F1 in-dataset. This demonstrates that high
        in-dataset accuracy provides no guarantee of real-world generalization — a
        finding that holds regardless of whether the failure mode is attack-signature
        drift, tooling incompatibility, or a combination of both.
        """
    )