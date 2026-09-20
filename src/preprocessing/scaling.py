"""
scaling.py
----------
Leakage-safe feature scaling.

RULE: StandardScaler is FIT on X_train ONLY. X_test is transformed using
the statistics learned from training data. Fitting on the full dataset
(or on test data) would leak test-set distribution information into the
training process - a common and serious mistake in ML research.
"""

import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fits a StandardScaler using training data statistics only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms X using an already-fitted scaler.
    Returns a DataFrame (not a raw numpy array) so column names are preserved
    for later steps like feature selection and SHAP explanations.
    """
    scaled_array = scaler.transform(X)
    return pd.DataFrame(scaled_array, columns=X.columns, index=X.index)


def save_scaler(scaler: StandardScaler, path: str) -> None:
    joblib.dump(scaler, path)
    print(f"Scaler saved: {path}")


def load_scaler(path: str) -> StandardScaler:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler file not found: {path}")
    return joblib.load(path)