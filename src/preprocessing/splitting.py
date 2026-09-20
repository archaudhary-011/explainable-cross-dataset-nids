"""
splitting.py
------------
Leakage-safe train/test splitting.

CRITICAL RULE: this is the LAST step that is allowed to touch the full
dataset. Everything after this file (scaling, feature selection, model
training) must be fit on X_train only, then applied to X_test.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_features_labels(df: pd.DataFrame,
                          label_col: str = "Label",
                          binary_col: str = "label_binary",
                          drop_cols: list = None):
    """
    Separates the dataframe into X (features) and y (binary target).

    Drops the original text label column and any other non-feature columns
    (the binary target itself is removed from X, obviously - keeping it in
    would be the most severe possible form of leakage).
    """
    if drop_cols is None:
        drop_cols = []

    cols_to_remove = list(set([label_col, binary_col] + drop_cols))
    cols_to_remove = [c for c in cols_to_remove if c in df.columns]

    X = df.drop(columns=cols_to_remove)
    y = df[binary_col].copy()

    return X, y


def make_train_test_split(X: pd.DataFrame, y: pd.Series, cfg: dict):
    """
    Stratified train/test split using settings from config.yaml.

    Stratification matters here because our classes are imbalanced
    (42.6% / 57.4%): without it, a random split could accidentally shift
    the class ratio between train and test, making metrics less reliable.
    """
    test_size = cfg["split"]["test_size"]
    stratify_arg = y if cfg["split"].get("stratify", True) else None
    seed = cfg["project"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_arg,
    )

    return X_train, X_test, y_train, y_test


def verify_split(X_train, X_test, y_train, y_test, verbose: bool = True) -> dict:
    """
    Sanity checks after splitting:
      - no row overlap between train and test (via index)
      - class ratio is preserved in both sets (stratification worked)
    """
    train_idx = set(X_train.index)
    test_idx = set(X_test.index)
    overlap = train_idx & test_idx

    train_ratio = y_train.value_counts(normalize=True).sort_index()
    test_ratio = y_test.value_counts(normalize=True).sort_index()

    result = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "index_overlap_count": len(overlap),
        "train_class_ratio": train_ratio.to_dict(),
        "test_class_ratio": test_ratio.to_dict(),
    }

    if verbose:
        print("=" * 64)
        print("SPLIT VERIFICATION")
        print("=" * 64)
        print(f"Train samples: {result['n_train']:,}")
        print(f"Test samples:  {result['n_test']:,}")
        print(f"Index overlap between train/test: {result['index_overlap_count']} "
              f"(MUST be 0)")
        print("-" * 64)
        print("Train class ratio:", {k: round(v, 4) for k, v in result["train_class_ratio"].items()})
        print("Test class ratio: ", {k: round(v, 4) for k, v in result["test_class_ratio"].items()})
        print("=" * 64)

        if result["index_overlap_count"] != 0:
            print("WARNING: TRAIN/TEST OVERLAP DETECTED. DO NOT PROCEED. This "
                  "indicates a leakage bug.")

    return result