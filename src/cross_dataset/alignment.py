"""
alignment.py
------------
Feature alignment between CICIDS2017 (source, training domain) and
external target domains (CSE-CIC-IDS2018, UNSW-NB15, etc.).

Different datasets use different feature-extraction tools with different
naming conventions AND different underlying feature definitions:
  - CICIDS2017 / CSE-CIC-IDS2018: both use CICFlowMeter (same tool,
    different versions) -> names differ, but almost all features map 1:1.
  - UNSW-NB15: uses Argus + Bro-IDS (a DIFFERENT tool entirely) -> most
    features have no direct counterpart, only a small common subset exists.

Any feature that cannot be confidently mapped is DROPPED and REPORTED,
never silently discarded.
"""

import pandas as pd

# --- CICIDS2017 -> CSE-CIC-IDS2018 mapping (same tool, renamed columns) ---
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

# --- CICIDS2017 -> UNSW-NB15 mapping (DIFFERENT tool: Argus/Bro-IDS) ---
# Only features with a genuine conceptual counterpart are included.
# Units and exact computation may still differ even where concepts match
# (e.g. inter-packet timing granularity) - this is documented as a
# limitation, not hidden.
FEATURE_NAME_MAP_2017_TO_UNSW = {
    "Total Fwd Packets": "spkts",             # source packet count
    "Fwd Packet Length Mean": "smean",         # mean packet size, source->dest
    "Total Length of Fwd Packets": "sbytes",   # source bytes
    "Init_Win_bytes_forward": "swin",          # source TCP window size
    "Init_Win_bytes_backward": "dwin",         # destination TCP window size
    "Fwd IAT Mean": "sinpkt",                  # source inter-packet arrival time
    "Bwd IAT Mean": "dinpkt",                  # destination inter-packet arrival time
}


def build_alignment_report(selected_features_2017: list, mapping: dict = None) -> dict:
    """
    For a given list of source-domain (2017) feature names, checks which
    have a known mapping to the target domain and which do not.

    mapping: which target-domain map to use. Defaults to the 2018 map
    for backward compatibility; pass FEATURE_NAME_MAP_2017_TO_UNSW for
    the UNSW-NB15 target.
    """
    if mapping is None:
        mapping = FEATURE_NAME_MAP_2017_TO_2018

    mapped = {}
    unmapped = []

    for feat in selected_features_2017:
        if feat in mapping:
            mapped[feat] = mapping[feat]
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
    print(f"Successfully mapped to target:    {report['n_mapped']}")
    print(f"Unmapped (dropped):               {report['n_unmapped']}")
    if report["unmapped_features"]:
        print(f"  Unmapped features: {report['unmapped_features']}")
    print(f"Final aligned feature count:      {report['final_feature_count']}")
    print("-" * 64)
    print("Mapping used (2017 -> target):")
    for k, v in report["mapped_features"].items():
        print(f"    {k:<35} -> {v}")
    print("=" * 64)


def rename_target_to_source_schema(df_target: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Selects only the mapped columns from the target dataframe and renames
    them back to the 2017 (source) naming convention, so a model trained
    on 2017 column names/order can be applied directly with zero changes.
    """
    reverse_map = {v: k for k, v in report["mapped_features"].items()}
    cols_target_needed = list(reverse_map.keys())

    missing = [c for c in cols_target_needed if c not in df_target.columns]
    if missing:
        raise ValueError(
            f"Expected target columns not found in dataframe: {missing}. "
            f"Check that the correct CSV file was loaded."
        )

    aligned = df_target[cols_target_needed].copy()
    aligned = aligned.rename(columns=reverse_map)
    aligned = aligned[list(report["mapped_features"].keys())]

    return aligned


# Backward-compatible alias (used in earlier notebooks for the 2018 case)
def rename_2018_to_2017_schema(df_2018: pd.DataFrame, report: dict) -> pd.DataFrame:
    return rename_target_to_source_schema(df_2018, report)