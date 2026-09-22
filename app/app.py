"""
app.py
------
Main Streamlit entry point for the Network IDS Framework demo UI.

This app is a VIEWER on top of already-computed results. It does not
retrain models or recompute anything expensive at runtime - it loads
saved artifacts (models, parquet files, JSON reports, CSVs) produced by
the notebooks in notebooks/01 through notebooks/11.

Run with:
    streamlit run app/app.py
(run this command from the PROJECT ROOT, not from inside app/)
"""

import sys
import os

# Add project root to path so `from src...` imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from src.utils import load_config

st.set_page_config(
    page_title="Network IDS Framework",
    page_icon="🛰️",
    layout="wide",
)

cfg = load_config()

st.sidebar.title("Network IDS Framework")
st.sidebar.markdown(
    "Explainable & Cross-Dataset Generalizable ML for Network Intrusion Detection"
)

page = st.sidebar.radio(
    "Navigate to:",
    [
        "1. Dataset Overview",
        "2. Model Comparison",
        "3. Live Prediction + SHAP Explanation",
        "4. Cross-Dataset Generalization",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Random seed: {cfg['project']['random_state']}")
st.sidebar.caption("All results loaded from results/ — nothing is recomputed live except Page 3's SHAP explanation.")

if page == "1. Dataset Overview":
    from pages_content.overview import render
    render(cfg)
elif page == "2. Model Comparison":
    from pages_content.model_comparison import render
    render(cfg)
elif page == "3. Live Prediction + SHAP Explanation":
    from pages_content.live_explain import render
    render(cfg)
elif page == "4. Cross-Dataset Generalization":
    from pages_content.cross_dataset import render
    render(cfg)