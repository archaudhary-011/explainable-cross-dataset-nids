"""
shap_utils.py
-------------
SHAP-based explainability for the Network IDS framework.

Uses TreeExplainer, which is fast and EXACT for tree-based models
(Random Forest / XGBoost) - unlike KernelExplainer which is slow and
approximate. This is why we chose a tree-based model as our final model.

COMPUTE NOTE: SHAP is computed on a random SAMPLE of the test set
(default 2000 rows), not all 44,616 rows. This is standard practice for
large datasets and keeps computation fast on a laptop, while still being
large enough to produce statistically stable global importance rankings.

VERSION ROBUSTNESS: newer SHAP versions return a shap.Explanation object;
older versions return raw numpy arrays (and for binary classification,
sometimes a list of two arrays - one per class). The helper functions
below normalize all of these into one consistent format: a 2D numpy
array of shape (n_samples, n_features) representing the SHAP values for
the POSITIVE class (DDoS = 1).
"""

import numpy as np
import pandas as pd
import shap


def get_shap_values_for_positive_class(explainer, X_sample: pd.DataFrame) -> np.ndarray:
    """
    Runs the explainer and returns SHAP values for the positive class (1)
    as a clean 2D numpy array (n_samples, n_features), regardless of which
    SHAP version/output format is in use.
    """
    raw = explainer.shap_values(X_sample)

    # Case 1: list of arrays, one per class (older SHAP / some sklearn RF outputs)
    if isinstance(raw, list):
        # index 1 = positive class (DDoS = 1)
        values = raw[1]
    else:
        values = raw

    values = np.asarray(values)

    # Case 2: 3D array (n_samples, n_features, n_classes) - newer SHAP versions
    # for some multiclass-capable models
    if values.ndim == 3:
        values = values[:, :, 1]  # positive class slice

    return values


def build_tree_explainer(model):
    """Creates a SHAP TreeExplainer for a fitted tree-based model."""
    return shap.TreeExplainer(model)


def sample_for_shap(X_test: pd.DataFrame, y_test: pd.Series,
                    n_samples: int = 2000, random_state: int = 42):
    """
    Draws a random, class-balanced-ish sample from the test set for SHAP
    computation. Uses the test set (never training data) since we are
    explaining the model's behavior on unseen data, consistent with how
    it will be evaluated.
    """
    n_samples = min(n_samples, len(X_test))
    sample_idx = X_test.sample(n=n_samples, random_state=random_state).index
    return X_test.loc[sample_idx], y_test.loc[sample_idx]


def pick_example_by_class(X_test: pd.DataFrame, y_test: pd.Series,
                          target_class: int, model, random_state: int = 42):
    """
    Picks one example the model correctly classifies as target_class
    (0 = BENIGN, 1 = DDoS), for local explanation. Choosing a CORRECTLY
    classified example (not a misclassified edge case) gives a clean,
    representative local explanation for the write-up.
    """
    y_pred = model.predict(X_test)
    correct_mask = (y_test.values == target_class) & (y_pred == target_class)
    candidates = X_test[correct_mask]

    if len(candidates) == 0:
        raise ValueError(f"No correctly classified examples found for class {target_class}")

    example = candidates.sample(n=1, random_state=random_state)
    return example