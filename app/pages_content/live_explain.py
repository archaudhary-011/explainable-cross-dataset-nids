"""
live_explain.py
----------------
Page 3: Live Prediction + SHAP Explanation.

Lets the user pick a sample (from the saved test set, or via CSV upload)
and see, live:
  - the model's prediction and confidence
  - a SHAP waterfall plot explaining exactly why

Uses the final tuned Random Forest (Step 7) with the 20 selected
features (Step 6). The model and SHAP explainer are cached so they are
built once per session, not on every interaction.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st


@st.cache_resource
def load_model_and_features(models_dir: str, reports_dir: str):
    model = joblib.load(os.path.join(models_dir, "final_model_random_forest.joblib"))
    with open(os.path.join(reports_dir, "selected_features.json")) as f:
        selection_info = json.load(f)
    selected_features = selection_info["selected_features"]
    return model, selected_features


@st.cache_resource
def build_explainer(_model):
    # Leading underscore tells Streamlit not to hash this argument
    # (it's a fitted sklearn model, not hashable in a meaningful way)
    return shap.TreeExplainer(_model)


@st.cache_data
def load_test_data(processed_dir: str):
    X_test = pd.read_parquet(os.path.join(processed_dir, "X_test_selected.parquet"))
    y_test = pd.read_parquet(os.path.join(processed_dir, "y_test.parquet"))["label_binary"]
    return X_test, y_test


def explain_and_display(sample_row: pd.DataFrame, model, explainer, selected_features):
    """
    Shared logic: given a single-row dataframe (already in the correct
    feature order), predicts and shows a SHAP waterfall explanation.
    """
    pred = model.predict(sample_row)[0]
    proba = model.predict_proba(sample_row)[0]

    pred_label = "DDoS" if pred == 1 else "BENIGN"
    confidence = proba[pred]

    col1, col2 = st.columns(2)
    with col1:
        if pred == 1:
            st.error(f"### Prediction: {pred_label}")
        else:
            st.success(f"### Prediction: {pred_label}")
    with col2:
        st.metric("Confidence", f"{confidence * 100:.2f}%")

    st.subheader("Feature values for this sample")
    st.dataframe(sample_row.T.rename(columns={sample_row.index[0]: "Value"}), width="stretch")

    st.subheader("Why did the model decide this? (SHAP explanation)")
    st.caption(
        "Red bars push the prediction toward DDoS. Blue bars push it toward BENIGN. "
        "The final position is the model's output for this specific sample."
    )

    raw_shap = explainer.shap_values(sample_row)
    if isinstance(raw_shap, list):
        shap_vals = np.asarray(raw_shap[1])
    else:
        shap_vals = np.asarray(raw_shap)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1]

    fig = plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals[0],
            base_values=expected_value,
            data=sample_row.values[0],
            feature_names=selected_features,
        ),
        show=False,
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render(cfg: dict):
    st.title("3. Live Prediction + SHAP Explanation")
    st.markdown(
        "Pick a network flow sample below to see the model's prediction "
        "and a live SHAP explanation of why it made that decision."
    )

    models_dir = cfg["paths"]["models_dir"]
    reports_dir = cfg["paths"]["reports_dir"]
    processed_dir = cfg["paths"]["processed_dir"]

    required_path = os.path.join(models_dir, "final_model_random_forest.joblib")
    if not os.path.exists(required_path):
        st.error("Final model not found. Run notebooks/07_final_model.ipynb first.")
        return

    model, selected_features = load_model_and_features(models_dir, reports_dir)
    explainer = build_explainer(model)

    mode = st.radio(
        "Choose input source:",
        ["Pick from test set", "Upload a CSV"],
        horizontal=True,
    )

    if mode == "Pick from test set":
        X_test, y_test = load_test_data(processed_dir)

        col_a, col_b = st.columns([3, 1])
        with col_a:
            idx = st.slider("Sample index", 0, len(X_test) - 1, 0)
        with col_b:
            if st.button("🎲 Random sample"):
                idx = int(np.random.randint(0, len(X_test)))
                st.session_state["random_idx"] = idx

        if "random_idx" in st.session_state:
            idx = st.session_state["random_idx"]

        sample_row = X_test.iloc[[idx]][selected_features]
        actual_label = "DDoS" if y_test.iloc[idx] == 1 else "BENIGN"
        st.caption(f"True label for this sample (from test set): **{actual_label}**")

        explain_and_display(sample_row, model, explainer, selected_features)

    else:
        st.markdown(f"Upload a CSV with these **{len(selected_features)} columns** (exact names required):")
        st.code(", ".join(selected_features))

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                user_df.columns = [c.strip() for c in user_df.columns]

                missing = [c for c in selected_features if c not in user_df.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                    return

                st.success(f"File loaded: {len(user_df)} row(s) found.")
                row_idx = 0
                if len(user_df) > 1:
                    row_idx = st.slider("Select row to explain", 0, len(user_df) - 1, 0)

                sample_row = user_df.iloc[[row_idx]][selected_features].apply(pd.to_numeric, errors="coerce")

                if sample_row.isna().any(axis=1).iloc[0]:
                    st.error("Selected row contains non-numeric or missing values in required columns.")
                    return

                explain_and_display(sample_row, model, explainer, selected_features)

            except Exception as e:
                st.error(f"Error reading file: {e}")