"""
cross_dataset.py
-----------------
Page 4: Cross-Dataset Generalization.

Displays the Step 11 results: in-dataset vs cross-dataset performance,
the generalization gap, and the domain shift analysis (KS test) that
explains WHY performance collapsed on the target domain.
"""

import os
import json
import pandas as pd
import streamlit as st


def render(cfg: dict):
    st.title("4. Cross-Dataset Generalization")
    st.markdown(
        "Testing the CICIDS2017-trained model on **CSE-CIC-IDS2018** "
        "(Wednesday-21-02-2018: DDoS-HOIC + DDoS-LOIC-UDP)."
    )

    metrics_dir = cfg["paths"]["metrics_dir"]
    reports_dir = cfg["paths"]["reports_dir"]

    in_dataset_path = os.path.join(metrics_dir, "final_model_metrics.csv")
    if not os.path.exists(in_dataset_path):
        st.error("In-dataset metrics not found. Run notebooks/07_final_model.ipynb first.")
        return

    in_dataset = pd.read_csv(in_dataset_path).iloc[0]

    # Cross-dataset metrics were printed but not saved to CSV in Step 11 -
    # hardcoded here from the actual executed result (see notebooks/11_cross_dataset_evaluation.ipynb)
    st.subheader("In-Dataset vs Cross-Dataset Performance")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("In-Dataset F1 (2017 test set)", f"{in_dataset['f1']:.4f}")
    with col2:
        st.metric("Cross-Dataset F1 (2018)", "0.0000", delta="-0.9999", delta_color="inverse")
    with col3:
        st.metric("Generalization Gap", "0.9999")

    st.error(
        "**The model failed completely on the target domain.** "
        "False Negative Rate = 1.0000 — every single DDoS flow in the 2018 "
        "dataset was misclassified as BENIGN."
    )

    st.subheader("Cross-Dataset Confusion Matrix")
    cm_data = pd.DataFrame(
        [[360519, 0], [200591, 0]],
        index=["Actual BENIGN", "Actual DDoS"],
        columns=["Predicted BENIGN", "Predicted DDoS"],
    )
    st.dataframe(cm_data, width="stretch")

    st.markdown("---")
    st.subheader("Why did this happen? Domain Shift Analysis")
    st.markdown(
        "The model relies heavily on packet-size features (see Page 3 / SHAP analysis). "
        "The Kolmogorov-Smirnov test below shows these exact features are the most "
        "statistically shifted between the 2017 (source) and 2018 (target) domains."
    )

    shift_ddos_path = os.path.join(reports_dir, "domain_shift_ddos.csv")
    if os.path.exists(shift_ddos_path):
        shift_ddos = pd.read_csv(shift_ddos_path)
        st.markdown("**Top 10 most-shifted features (DDoS class, source vs target):**")
        st.dataframe(
            shift_ddos[["feature", "source_mean", "target_mean", "ks_statistic",
                       "wasserstein_normalized", "significant_shift"]].head(10),
            width="stretch",
        )

        n_sig = shift_ddos["significant_shift"].sum()
        st.caption(f"{n_sig} of {len(shift_ddos)} features show statistically significant shift (KS test, p < 0.05).")
    else:
        st.info("Domain shift report not found. Run notebooks/11_cross_dataset_evaluation.ipynb first.")

    figures_dir = cfg["paths"]["figures_dir"]
    shift_plot_path = os.path.join(figures_dir, "domain_shift_top_feature.png")
    if os.path.exists(shift_plot_path):
        st.subheader("Distribution shift for the most-shifted feature")
        st.image(shift_plot_path, width="stretch")

    st.markdown("---")
    st.subheader("Research Interpretation")
    st.markdown(
        """
        This result demonstrates a well-documented limitation in network intrusion detection
        research: a model trained on one attack tool's traffic signature (2017's DDoS tool,
        characterized by very small packet sizes) does not automatically detect a different
        attack tool's version of the "same" attack category (2018's HOIC/LOIC-UDP tools,
        which produce much larger packets).

        **This is the central research finding of this project**: high in-dataset accuracy
        (99.99%) does not imply real-world generalization, and explainability analysis (SHAP)
        combined with domain shift statistics (KS test) can *predict and explain* this failure
        before it is even observed empirically.
        """
    )