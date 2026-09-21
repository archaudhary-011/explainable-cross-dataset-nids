"""
feature_selection.py
--------------------
Feature reduction for the Network IDS framework.

APPROACH (deliberately simple and defensible, not "every technique for
the sake of it"):
  1. Correlation analysis - remove features that are near-duplicates of
     another feature (redundant, adds no information, can destabilize
     Logistic Regression coefficients and slow down SHAP).
  2. Random Forest importance - rank remaining features and keep the
     top-K, since RF showed the most distributed (non-collapsed)
     importance profile of all models tested (Step 5).

LEAKAGE RULE: both the correlation matrix and the importance ranking are
computed using X_train ONLY. X_test is never used to decide which
features to keep - only to evaluate the result afterward.
"""

import numpy as np
import pandas as pd


def find_correlated_features(X_train: pd.DataFrame, threshold: float = 0.95) -> dict:
    """
    Finds pairs of features with absolute Pearson correlation above the
    threshold, computed on TRAINING DATA ONLY.

    For each correlated pair, keeps the first-encountered feature and
    marks the second as redundant - a simple, transparent, deterministic
    rule (not an arbitrary manual pick).

    Returns a dict with the correlation matrix, the list of correlated
    pairs, and the list of features recommended for removal.
    """
    corr_matrix = X_train.corr().abs()

    # Upper triangle only, to avoid counting each pair twice and to avoid
    # comparing a feature against itself
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    correlated_pairs = []
    to_drop = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > threshold].tolist()
        for row in high_corr:
            correlated_pairs.append((row, col, round(upper.loc[row, col], 4)))
            to_drop.add(col)  # drop the second feature in the pair

    return {
        "correlation_matrix": corr_matrix,
        "correlated_pairs": correlated_pairs,
        "recommended_drop": sorted(to_drop),
    }


def select_top_k_by_importance(feature_importances: pd.Series, k: int) -> list:
    """
    Returns the names of the top-K features by importance score.
    feature_importances must be a pandas Series indexed by feature name
    (e.g. from a fitted RandomForestClassifier.feature_importances_),
    computed from a model trained on X_train only.
    """
    return feature_importances.sort_values(ascending=False).head(k).index.tolist()


def apply_feature_selection(X_train: pd.DataFrame, X_test: pd.DataFrame,
                            selected_features: list):
    """
    Subsets both train and test sets to the selected feature list.
    Both are subset identically and in the same column order, which
    matters for tree-based models and for consistent SHAP explanations.
    """
    X_train_sel = X_train[selected_features].copy()
    X_test_sel = X_test[selected_features].copy()
    return X_train_sel, X_test_sel


def print_selection_report(n_original: int, dropped_correlated: list,
                           n_after_correlation: int, k_selected: int,
                           final_features: list) -> None:
    print("=" * 64)
    print("FEATURE SELECTION REPORT")
    print("=" * 64)
    print(f"Original features:                {n_original}")
    print(f"Removed (correlation > 0.95):      {len(dropped_correlated)}")
    for f in dropped_correlated:
        print(f"    - {f}")
    print(f"Remaining after correlation step:  {n_after_correlation}")
    print(f"Top-K selected by RF importance:   {k_selected}")
    print("-" * 64)
    print("Final selected features:")
    for f in final_features:
        print(f"    - {f}")
    print("=" * 64)