"""
cross_dataset_cleaning.py
--------------------------
Cleaning and label encoding for the target-domain (CSE-CIC-IDS2018) data,
mirroring the source-domain cleaning logic (src/preprocessing/cleaning.py)
so both datasets are prepared under the same rules.

SCOPE NOTE: this 2018 file contains 'Benign', 'DDOS attack-HOIC', and
'DDOS attack-LOIC-UDP'. For a fair like-for-like comparison against the
2017 BENIGN-vs-DDoS model, we keep only these classes (no filtering
needed here since the file already contains exactly these three) and
merge both DDoS variants into a single 'DDOS' binary label, consistent
with how 2017's single-tool DDoS was encoded.
"""

import numpy as np
import pandas as pd


def clean_target_dataset(X: pd.DataFrame, y_raw: pd.Series, verbose: bool = True):
    """
    Applies the same structural cleaning as the source dataset:
    duplicates, infinities, NaN, and reports what was removed.
    Operates on an already feature-aligned X (20 columns matching 2017).
    """
    n_start = len(X)

    combined = X.copy()
    combined["_label_raw"] = y_raw.values

    n_before_dup = len(combined)
    combined = combined.drop_duplicates()
    n_duplicates = n_before_dup - len(combined)

    y_clean = combined["_label_raw"]
    features = combined.drop(columns=["_label_raw"]).apply(pd.to_numeric, errors="coerce")

    n_infinite = int(np.isinf(features.to_numpy(dtype="float64", na_value=np.nan)).sum())
    features = features.replace([np.inf, -np.inf], np.nan)

    nan_mask = features.isna().any(axis=1)
    n_nan_rows = int(nan_mask.sum())
    features = features.loc[~nan_mask]
    y_clean = y_clean.loc[~nan_mask]

    if verbose:
        print("=" * 64)
        print("TARGET DATASET (2018) CLEANING REPORT")
        print("=" * 64)
        print(f"Rows at start:                {n_start:,}")
        print(f"Duplicate rows removed:       {n_duplicates:,}")
        print(f"Infinite values found:        {n_infinite:,}")
        print(f"Rows removed (NaN/inf):       {n_nan_rows:,}")
        print(f"Final rows:                   {len(features):,}")
        print("=" * 64)

    return features.reset_index(drop=True), y_clean.reset_index(drop=True)


def encode_target_labels(y_raw: pd.Series, benign_label: str = "Benign") -> pd.Series:
    """
    Encodes 2018 labels to the same binary scheme as 2017:
    Benign -> 0, any DDoS variant -> 1.
    Raises an error on any unexpected label so nothing is silently
    mis-encoded.
    """
    y_raw = y_raw.astype(str).str.strip()
    unique_labels = set(y_raw.unique())

    expected_ddos = {"DDOS attack-HOIC", "DDOS attack-LOIC-UDP"}
    unexpected = unique_labels - {benign_label} - expected_ddos
    if unexpected:
        raise ValueError(f"Unexpected labels found: {unexpected}")

    y_binary = y_raw.apply(lambda v: 0 if v == benign_label else 1).astype(int)
    return y_binary