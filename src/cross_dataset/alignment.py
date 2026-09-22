"""
alignment.py
------------
Feature alignment between CICIDS2017 (source, training domain) and
CSE-CIC-IDS2018 (target, external test domain).

The two datasets use different CICFlowMeter versions with different
column naming conventions for the SAME underlying features (e.g.
'Fwd Packet Length Max' in 2017 = 'Fwd Pkt Len Max' in 2018). This
module provides an explicit, documented mapping - never a silent guess.

Any feature that cannot be confidently mapped is DROPPED and REPORTED,
never silently discarded.
"""

import pandas as pd

# Explicit name mapping: 2017 name -> 2018 name.
# Built by manually matching CICFlowMeter's documented feature semantics
# across the two dataset versions. Only includes features relevant to
# our Step 6 selected feature set, but structured so it can be extended
# to the full feature set if needed later.
FEATURE_NAME_MAP_2017_TO_2018 = {
    "Fwd Packet Length Max": "Fwd Pkt Len Max",
    "Total Length of Fwd Packets": "TotLen Fwd Pkts",
    "Destination Port": "Dst Port",
    "Init_Win_bytes_forward": "Init Fwd Win Byts",
    "act_data_pkt_fwd": "Fwd Act Data Pkts",
    "Bwd Packet Length Max": "Bwd Pkt Len Max",
    "Fwd Packet Length Mean": "Fwd Pkt Len Mean",
    "Total Fwd Packets": "Tot Fwd Pkts",
    "Bwd Packet Length Min": "Bwd Pkt Len Min",
    "Init_Win_bytes_backward": "Init Bwd Win Byts",
    "Fwd IAT Mean": "Fwd IAT Mean",
    "Flow IAT Std": "Flow IAT Std",
    "Bwd IAT Total": "Bwd IAT Tot",
    "Bwd Packets/s": "Bwd Pkts/s",
    "Max Packet Length": "Pkt Len Max",
    "Packet Length Mean": "Pkt Len Mean",
    "Bwd IAT Mean": "Bwd IAT Mean",
    "URG Flag Count": "URG Flag Cnt",
    "Packet Length Variance": "Pkt Len Var",
    "Fwd Packet Length Min": "Fwd Pkt Len Min",
}


def build_alignment_report(selected_features_2017: list) -> dict:
    """
    For a given list of source-domain (2017) feature names, checks which
    have a known mapping to the target domain (2018) and which do not.
    This is the transparency report required by the project spec:
    total source features, common, unavailable, mapped, dropped, final count.
    """
    mapped = {}
    unmapped = []

    for feat in selected_features_2017:
        if feat in FEATURE_NAME_MAP_2017_TO_2018:
            mapped[feat] = FEATURE_NAME_MAP_2017_TO_2018[feat]
        else:
            unmapped.append(feat)

    report = {
        "total_source_features": len(selected_features_2017),
        "mapped_features": mapped,
        "n_mapped": len(mapped),
        "unmapped_features": unmapped,
        "n_unmapped": len(unmapped),
        "final_feature_count": len(mapped),
    }
    return report


def print_alignment_report(report: dict) -> None:
    print("=" * 64)
    print("CROSS-DATASET FEATURE ALIGNMENT REPORT")
    print("=" * 64)
    print(f"Total source (2017) features:     {report['total_source_features']}")
    print(f"Successfully mapped to 2018:      {report['n_mapped']}")
    print(f"Unmapped (would be dropped):      {report['n_unmapped']}")
    if report["unmapped_features"]:
        print(f"  Unmapped features: {report['unmapped_features']}")
    print(f"Final aligned feature count:      {report['final_feature_count']}")
    print("-" * 64)
    print("Mapping used (2017 -> 2018):")
    for k, v in report["mapped_features"].items():
        print(f"    {k:<35} -> {v}")
    print("=" * 64)


def rename_2018_to_2017_schema(df_2018: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Selects only the mapped columns from the 2018 dataframe and renames
    them back to the 2017 (source) naming convention. This means the
    SAME trained model (which expects 2017 column names/order) can be
    applied directly to the renamed 2018 data with zero changes.
    """
    reverse_map = {v: k for k, v in report["mapped_features"].items()}
    cols_2018_needed = list(reverse_map.keys())

    missing = [c for c in cols_2018_needed if c not in df_2018.columns]
    if missing:
        raise ValueError(
            f"Expected 2018 columns not found in dataframe: {missing}. "
            f"Check that the correct CSV file was loaded."
        )

    aligned = df_2018[cols_2018_needed].copy()
    aligned = aligned.rename(columns=reverse_map)

    # Reorder to match the exact order of the original 2017 selected features
    aligned = aligned[list(report["mapped_features"].keys())]

    return aligned