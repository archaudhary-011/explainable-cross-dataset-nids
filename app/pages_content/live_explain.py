"""
live_explain.py
----------------
Page 3: Live Prediction + SHAP Explanation.

Lets the user pick a sample (from the saved test set, a cross-dataset
target, or via CSV upload) and see, live:
  - the model's prediction and confidence
  - a SHAP waterfall plot explaining exactly why

Uses the final tuned Random Forest (Step 7) with the 20 selected
features (Step 6) for the primary modes, and the auxiliary 7-feature
model for the UNSW-NB15 cross-dataset case. Models and SHAP explainers
are cached so they are built once per session, not on every interaction.
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


@st.cache_data
def load_cross_data(x_path: str, y_path: str):
    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)["label_binary"]
    return X, y


@st.cache_resource
def load_cross_model(model_path: str):
    return joblib.load(model_path)


def explain_and_display(sample_row: pd.DataFrame, model, explainer, feature_names):
    """Predict one sample and display its SHAP waterfall explanation."""
    # Optional terminal debugging: uncomment these lines if you need to
    # inspect the feature values for the currently selected sample.
    # print("\nFeature values for the current sample:")
    # print(sample_row.T.to_string())

    pred = model.predict(sample_row)[0]
    proba = model.predict_proba(sample_row)[0]

    # Find the probability index for class 1 (attack), rather than assuming
    # class ordering without checking the fitted model.
    classes = list(getattr(model, "classes_", [0, 1]))
    positive_idx = classes.index(1) if 1 in classes else int(np.argmax(classes))
    positive_probability = float(proba[positive_idx])

    pred_label = "DDoS / Attack" if pred == 1 else "BENIGN"
    confidence = float(proba[classes.index(pred)])

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
        "Red bars push the explanation toward the attack class; blue bars push it "
        "toward BENIGN. The waterfall explains the selected class output."
    )

    try:
        # Disable SHAP's internal additivity assertion to avoid the exception
        # that can occur with some model/data/version combinations. We still
        # calculate and report the difference below instead of hiding it.
        raw_shap = explainer.shap_values(sample_row, check_additivity=False)

        if isinstance(raw_shap, list):
            # Older SHAP versions return one array per class.
            shap_vals = np.asarray(raw_shap[positive_idx])
        else:
            shap_vals = np.asarray(raw_shap)
            # Newer SHAP versions may return (samples, features, classes).
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, positive_idx]

        # Expected value may be scalar or one value per class.
        expected_value = np.asarray(explainer.expected_value)
        if expected_value.ndim > 0 and expected_value.size > 1:
            expected_value = expected_value.reshape(-1)[positive_idx]
        else:
            expected_value = expected_value.reshape(-1)[0]
        expected_value = float(expected_value)

        # Ensure one-dimensional SHAP values for the single selected row.
        shap_values_for_row = np.asarray(shap_vals[0]).reshape(-1)
        if len(shap_values_for_row) != len(feature_names):
            st.error(
                "SHAP returned a different number of feature values than expected. "
                f"SHAP values: {len(shap_values_for_row)}; features: {len(feature_names)}. "
                "Check that the model and input feature columns match."
            )
            return

        approx = float(expected_value + shap_values_for_row.sum())
        difference = abs(approx - positive_probability)
        if difference > 0.02:
            st.warning(
                f"SHAP total ({approx:.3f}) differs from the model's attack probability "
                f"({positive_probability:.3f}) by {difference:.3f}. The waterfall is an "
                "approximation for this model/output configuration; interpret it cautiously."
            )
        else:
            st.caption(
                f"SHAP total: {approx:.3f} | Model attack probability: "
                f"{positive_probability:.3f} | Absolute difference: {difference:.3f}"
            )

        fig = plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_for_row,
                base_values=expected_value,
                data=sample_row.iloc[0].to_numpy(),
                feature_names=list(feature_names),
            ),
            show=False,
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as exc:
        st.error(f"Could not create the SHAP explanation: {exc}")
        st.info(
            "The prediction above was still generated. If this error continues, "
            "copy the full error from the terminal and share it for debugging."
        )


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
        ["Pick from test set", "Cross-Dataset Sample (watch it fail)", "Upload a CSV"],
        horizontal=True,
    )

    # ------------------------------------------------------------------
    # MODE 1: Pick from the CICIDS2017 held-out test set
    # ------------------------------------------------------------------
    if mode == "Pick from test set":
        X_test, y_test = load_test_data(processed_dir)

        col_a, col_b = st.columns([3, 1])
        with col_a:
            idx = st.slider("Sample index", 0, len(X_test) - 1, 0, key="test_idx_slider")
        with col_b:
            if st.button("🎲 Random sample", key="test_random_btn"):
                idx = int(np.random.randint(0, len(X_test)))
                st.session_state["random_idx"] = idx

        if "random_idx" in st.session_state:
            idx = st.session_state["random_idx"]

        sample_row = X_test.iloc[[idx]][selected_features]
        actual_label = "DDoS" if y_test.iloc[idx] == 1 else "BENIGN"
        st.caption(f"True label for this sample (from test set): **{actual_label}**")

        explain_and_display(sample_row, model, explainer, selected_features)

    # ------------------------------------------------------------------
    # MODE 2: Cross-dataset sample - demonstrates the generalization failure
    # ------------------------------------------------------------------
    elif mode == "Cross-Dataset Sample (watch it fail)":
        st.warning(
            "This model was trained ONLY on CICIDS2017. These samples come from "
            "completely different datasets it has never seen. Watch it confidently "
            "get the wrong answer."
        )

        target_choice = st.selectbox(
            "Choose target dataset:",
            ["CSE-CIC-IDS2018", "UNSW-NB15", "CICDDoS2019"],
        )

        target_config = {
            "CSE-CIC-IDS2018": {
                "x_file": "cross_dataset_2018_X.parquet",
                "y_file": "cross_dataset_2018_y.parquet",
                "model_file": "final_model_random_forest.joblib",
                "features": selected_features,
            },
            "UNSW-NB15": {
                "x_file": "cross_dataset_unsw_X.parquet",
                "y_file": "cross_dataset_unsw_y.parquet",
                "model_file": "rf_7feature_for_unsw.joblib",
                "features": None,  # inferred from the saved parquet's own columns
            },
            "CICDDoS2019": {
                "x_file": "cross_dataset_ddos2019_X.parquet",
                "y_file": "cross_dataset_ddos2019_y.parquet",
                "model_file": "final_model_random_forest.joblib",
                "features": selected_features,
            },
        }

        conf = target_config[target_choice]
        x_path = os.path.join(processed_dir, conf["x_file"])
        y_path = os.path.join(processed_dir, conf["y_file"])
        model_path = os.path.join(models_dir, conf["model_file"])

        if not (os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(model_path)):
            st.error(
                f"Data or model not found for {target_choice}.\n\n"
                f"Expected files:\n- {x_path}\n- {y_path}\n- {model_path}\n\n"
                f"Make sure the corresponding notebook has been run and its "
                f"save-to-disk cell executed."
            )
            return

        X_cross, y_cross = load_cross_data(x_path, y_path)
        cross_model = load_cross_model(model_path)
        cross_features = conf["features"] if conf["features"] is not None else list(X_cross.columns)
        cross_explainer = build_explainer(cross_model)

        st.caption(
            f"Loaded {len(X_cross):,} samples from {target_choice}. "
            f"Model uses {len(cross_features)} feature(s): {', '.join(cross_features)}"
        )

        col_a, col_b = st.columns([3, 1])
        with col_a:
            idx = st.slider(f"Sample index ({target_choice})", 0, len(X_cross) - 1, 0, key="cross_idx_slider")
        with col_b:
            if st.button("🎲 Random sample", key="cross_random_btn"):
                idx = int(np.random.randint(0, len(X_cross)))
                st.session_state["cross_random_idx"] = idx

        if "cross_random_idx" in st.session_state:
            idx = st.session_state["cross_random_idx"]

        sample_row = X_cross.iloc[[idx]][cross_features]
        actual_label = "Attack/DDoS" if y_cross.iloc[idx] == 1 else "BENIGN"

        pred = cross_model.predict(sample_row)[0]
        pred_label = "Attack/DDoS" if pred == 1 else "BENIGN"

        if pred_label != actual_label:
            st.error(f"**MISCLASSIFIED** — True label: **{actual_label}**  |  Model predicted: **{pred_label}**")
        else:
            st.success(f"Correctly classified — True label: **{actual_label}**  |  Model predicted: **{pred_label}**")

        explain_and_display(sample_row, cross_model, cross_explainer, cross_features)

    # ------------------------------------------------------------------
    # MODE 3: Upload a CSV
    # ------------------------------------------------------------------
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
                    row_idx = st.slider("Select row to explain", 0, len(user_df) - 1, 0, key="upload_row_slider")

                sample_row = user_df.iloc[[row_idx]][selected_features].apply(pd.to_numeric, errors="coerce")

                if sample_row.isna().any(axis=1).iloc[0]:
                    st.error("Selected row contains non-numeric or missing values in required columns.")
                    return

                explain_and_display(sample_row, model, explainer, selected_features)

            except Exception as e:
                st.error(f"Error reading file: {e}")
