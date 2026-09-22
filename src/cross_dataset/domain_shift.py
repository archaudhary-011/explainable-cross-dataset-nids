"""
domain_shift.py
----------------
Quantifies distribution differences between source (2017) and target
(2018) domains, to explain WHY cross-dataset performance changed.

Methods used (deliberately a small, defensible set - not every possible
metric):
  - Kolmogorov-Smirnov (KS) test: per-feature, tests whether source and
    target come from the same distribution. Standard, interpretable,
    doesn't assume normality.
  - Mean/std comparison: simple, human-readable magnitude of shift.
  - Wasserstein distance: captures how far apart distributions are,
    robust to different scales across features.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


def compute_domain_shift_report(X_source: pd.DataFrame, X_target: pd.DataFrame) -> pd.DataFrame:
    """
    For each shared feature, computes:
      - source/target mean and std
      - KS statistic and p-value
      - Wasserstein distance (normalized by source std, so features on
        different scales are comparable)
    Returns a dataframe sorted by KS statistic descending (most-shifted
    features first).
    """
    rows = []
    common_features = [c for c in X_source.columns if c in X_target.columns]

    for feat in common_features:
        src = X_source[feat].dropna().values
        tgt = X_target[feat].dropna().values

        ks_stat, ks_pval = ks_2samp(src, tgt)

        src_std = src.std() if src.std() > 0 else 1e-9
        w_dist = wasserstein_distance(src, tgt) / src_std

        rows.append({
            "feature": feat,
            "source_mean": src.mean(),
            "target_mean": tgt.mean(),
            "source_std": src.std(),
            "target_std": tgt.std(),
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "significant_shift": ks_pval < 0.05,
            "wasserstein_normalized": w_dist,
        })

    report = pd.DataFrame(rows).sort_values("ks_statistic", ascending=False).reset_index(drop=True)
    return report


def print_domain_shift_summary(report: pd.DataFrame) -> None:
    n_significant = report["significant_shift"].sum()
    n_total = len(report)

    print("=" * 64)
    print("DOMAIN SHIFT SUMMARY")
    print("=" * 64)
    print(f"Features tested:                {n_total}")
    print(f"Features with significant shift (KS p<0.05): {n_significant}/{n_total}")
    print("-" * 64)
    print("Top 10 most shifted features (by KS statistic):")
    print(report[["feature", "ks_statistic", "wasserstein_normalized", "significant_shift"]].head(10).to_string(index=False))
    print("=" * 64)