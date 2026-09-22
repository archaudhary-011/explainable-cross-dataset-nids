"""
overview.py
-----------
Page 1: Dataset Overview.
Loads the cleaned CICIDS2017 parquet (Step 2 output) and displays
class balance, feature counts, and basic structural facts.
"""

import os
import pandas as pd
import streamlit as st


def render(cfg: dict):
    st.title("1. Dataset Overview")
    st.markdown(
        "Source dataset: **CICIDS2017** (Friday-Afternoon DDoS capture). "
        "Scope: binary classification, BENIGN vs DDoS."
    )

    processed_dir = cfg["paths"]["processed_dir"]
    parquet_path = os.path.join(processed_dir, "cicids2017_ddos_cleaned.parquet")

    if not os.path.exists(parquet_path):
        st.error(
            f"Cleaned dataset not found at:\n{parquet_path}\n\n"
            "Run notebooks/02_cleaning.ipynb first."
        )
        return

    df = pd.read_parquet(parquet_path)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total samples", f"{len(df):,}")
    col2.metric("Total features", f"{df.shape[1] - 2}")  # exclude Label + label_binary
    col3.metric("Classes", "2 (BENIGN / DDoS)")

    st.subheader("Class distribution")
    class_counts = df["Label"].value_counts()
    class_props = (df["Label"].value_counts(normalize=True) * 100).round(2)

    dist_df = pd.DataFrame({
        "Count": class_counts,
        "Percentage": class_props.astype(str) + "%",
    })
    st.dataframe(dist_df, width="stretch")

    st.bar_chart(class_counts)

    st.subheader("Sample rows (first 10)")
    st.dataframe(df.drop(columns=["label_binary"]).head(10), width="stretch")

    with st.expander("Data cleaning summary (from Step 2)"):
        report_path = os.path.join(cfg["paths"]["reports_dir"], "cleaning_report.json")
        if os.path.exists(report_path):
            import json
            with open(report_path) as f:
                report = json.load(f)
            st.json(report)
        else:
            st.info("Cleaning report not found. Run notebooks/02_cleaning.ipynb first.")