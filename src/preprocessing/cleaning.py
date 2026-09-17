"""
cleaning.py
-----------
Structural cleaning utilities for the Network IDS framework.

DESIGN RULE (leakage safety):
This module contains ONLY deterministic, row/column-level validity fixes:
  - column name normalisation
  - duplicate row removal
  - duplicate COLUMN removal (Fwd Header Length appears twice in the raw CSV)
  - type coercion
  - invalid value removal (inf, NaN, negative Flow Duration)
  - removal of constant (zero-variance) columns

None of these steps learn a statistic from the data, so applying them
before the train/test split does NOT cause leakage.

Anything that DOES learn from data (scalers, encoders, feature selectors,
resamplers such as SMOTE) is deliberately excluded from this file and must
be fit on the training split only (handled in Step 3).
"""

import numpy as np
import pandas as pd

# Zero-variance columns identified during EDA on the CICIDS2017 DDoS file.
ZERO_VARIANCE_COLUMNS = [
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "CWE Flag Count",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]

# Columns to drop unconditionally because they are not model features:
# 'source_file' was added by our own loader for traceability, not a real
# network feature. 'Fwd Header Length.1' is a pandas-generated duplicate
# of 'Fwd Header Length' caused by a repeated column name in the raw CSV.
NON_FEATURE_COLUMNS = ["source_file"]
KNOWN_DUPLICATE_COLUMNS = ["Fwd Header Length.1"]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strips whitespace from column names (defensive; loader already does this)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def report_missing_and_infinite(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    """
    Diagnostic only. Returns a per-column count of NaN and infinite values.
    Run this BEFORE cleaning so the 'before' state is documented in the thesis.
    """
    df = clean_column_names(df)
    drop_cols = [c for c in (label_col, *NON_FEATURE_COLUMNS) if c in df.columns]
    numeric_df = df.drop(columns=drop_cols, errors="ignore").apply(pd.to_numeric, errors="coerce")

    n_missing = numeric_df.isna().sum()
    n_infinite = np.isinf(numeric_df.to_numpy(dtype="float64", na_value=np.nan)).sum(axis=0)

    report = pd.DataFrame({
        "n_missing": n_missing.values,
        "n_infinite": n_infinite,
    }, index=numeric_df.columns)

    report = report[(report["n_missing"] > 0) | (report["n_infinite"] > 0)]
    return report.sort_values(["n_infinite", "n_missing"], ascending=False)


def clean_dataset(df: pd.DataFrame,
                  label_col: str = "Label",
                  drop_zero_variance: bool = True,
                  verbose: bool = True) -> tuple:
    """
    Main cleaning routine.

    Steps, in order:
      1. Normalise column names
      2. Drop non-feature bookkeeping columns (source_file) and the known
         duplicate column (Fwd Header Length.1)
      3. Drop exact duplicate ROWS
      4. Coerce all feature columns to numeric
      5. Replace +/-inf with NaN
      6. Drop rows containing NaN
      7. Drop rows with negative Flow Duration (physically impossible)
      8. Drop zero-variance columns

    Returns
    -------
    cleaned_df : pd.DataFrame
    report : dict   (machine-readable cleaning log)
    """
    df = clean_column_names(df)
    n_start = len(df)
    n_cols_start = df.shape[1]

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. "
            f"Available columns include: {list(df.columns)[-5:]}"
        )

    # --- Step 2: drop bookkeeping + known duplicate columns ---
    cols_to_drop_upfront = [c for c in (NON_FEATURE_COLUMNS + KNOWN_DUPLICATE_COLUMNS)
                             if c in df.columns]
    df = df.drop(columns=cols_to_drop_upfront)

    # --- Step 3: duplicate rows ---
    n_before = len(df)
    df = df.drop_duplicates()
    n_duplicates = n_before - len(df)

    # --- Step 4: split label / coerce numeric ---
    labels = df[label_col]
    features = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce")

    # --- Step 5: infinities -> NaN ---
    n_infinite = int(np.isinf(features.to_numpy(dtype="float64", na_value=np.nan)).sum())
    features = features.replace([np.inf, -np.inf], np.nan)

    # --- Step 6: drop NaN rows ---
    nan_mask = features.isna().any(axis=1)
    n_nan_rows = int(nan_mask.sum())
    features = features.loc[~nan_mask]
    labels = labels.loc[~nan_mask]

    # --- Step 7: invalid negative Flow Duration ---
    n_negative_flow = 0
    if "Flow Duration" in features.columns:
        neg_mask = features["Flow Duration"] < 0
        n_negative_flow = int(neg_mask.sum())
        features = features.loc[~neg_mask]
        labels = labels.loc[~neg_mask]

    # --- Step 8: zero-variance columns (recomputed, then compared to constant) ---
    detected_zv = []
    if drop_zero_variance:
        nunique = features.nunique(dropna=True)
        detected_zv = sorted(nunique[nunique <= 1].index.tolist())
        features = features.drop(columns=detected_zv)

    cleaned = features.copy()
    cleaned[label_col] = labels.values
    cleaned = cleaned.reset_index(drop=True)

    report = {
        "rows_start": int(n_start),
        "cols_start": int(n_cols_start),
        "columns_dropped_upfront": cols_to_drop_upfront,
        "duplicates_removed": int(n_duplicates),
        "infinite_values_found": int(n_infinite),
        "rows_removed_nan_or_inf": int(n_nan_rows),
        "rows_removed_negative_flow_duration": int(n_negative_flow),
        "zero_variance_detected": detected_zv,
        "zero_variance_matches_eda": sorted(detected_zv) == sorted(ZERO_VARIANCE_COLUMNS),
        "rows_final": int(len(cleaned)),
        "cols_final": int(cleaned.shape[1]),
        "rows_retained_pct": round(100 * len(cleaned) / n_start, 3),
    }

    if verbose:
        print("=" * 64)
        print("DATA CLEANING REPORT")
        print("=" * 64)
        print(f"Rows at start:                     {report['rows_start']:,}")
        print(f"Columns at start:                  {report['cols_start']}")
        print("-" * 64)
        print(f"Columns dropped upfront:           {cols_to_drop_upfront}")
        print(f"Duplicate rows removed:            {report['duplicates_removed']:,}")
        print(f"Infinite values found:             {report['infinite_values_found']:,}")
        print(f"Rows removed (NaN/inf):            {report['rows_removed_nan_or_inf']:,}")
        print(f"Rows removed (Flow Duration < 0):  {report['rows_removed_negative_flow_duration']:,}")
        print("-" * 64)
        print(f"Zero-variance columns detected ({len(detected_zv)}):")
        for c in detected_zv:
            print(f"    - {c}")
        print(f"Matches EDA expectation:           {report['zero_variance_matches_eda']}")
        print("-" * 64)
        print(f"Final rows:                        {report['rows_final']:,}")
        print(f"Final columns (incl. label):       {report['cols_final']}")
        print(f"Data retained:                     {report['rows_retained_pct']}%")
        print("=" * 64)

    return cleaned, report


def normalize_labels(df: pd.DataFrame, label_col: str = "Label", verbose: bool = True) -> pd.DataFrame:
    """
    Strips whitespace and standardises label casing (e.g. 'DDoS' -> 'DDOS').
    Does not merge distinct classes - only fixes formatting.
    """
    df = df.copy()
    df[label_col] = df[label_col].astype(str).str.strip().str.upper()

    if verbose:
        counts = df[label_col].value_counts()
        props = (df[label_col].value_counts(normalize=True) * 100).round(2)
        print("Label distribution after cleaning:")
        for lab in counts.index:
            print(f"    {lab:<12} {counts[lab]:>10,}   ({props[lab]:.2f}%)")
    return df


def encode_labels(df: pd.DataFrame, cfg: dict, label_col: str = "Label") -> pd.DataFrame:
    """
    Converts text labels to binary integers using the mapping in config.yaml:
    BENIGN -> 0, DDOS -> 1. Raises an error if an unexpected label appears,
    rather than silently mis-encoding it.
    """
    df = df.copy()
    benign = cfg["dataset"]["benign_label"]
    attack = cfg["dataset"]["attack_label"]

    unexpected = set(df[label_col].unique()) - {benign, attack}
    if unexpected:
        raise ValueError(
            f"Unexpected label values found: {unexpected}. "
            f"Expected only '{benign}' and '{attack}'. "
            f"Check normalize_labels() output above."
        )

    df["label_binary"] = df[label_col].map({benign: 0, attack: 1}).astype(int)
    return df