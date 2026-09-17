"""
loader.py
---------
Dataset loading for the Network IDS framework.

Written to be dataset-agnostic: the same function will later load the
external CSE-CIC-IDS2018 CSVs during cross-dataset evaluation, so we do
not write throwaway loading code.
"""

import os
import glob
import pandas as pd


def list_available_files(directory: str, pattern: str = "*.csv") -> list:
    """
    Lists CSV files in a directory. Use this first to confirm exact
    filenames on disk (they vary slightly between CICIDS2017 mirrors).
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Directory not found: {directory}")
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    return [os.path.basename(f) for f in files]


def load_csv_files(directory: str,
                   filenames: list = None,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Loads one or more CSVs from a directory and concatenates them.

    Notes
    -----
    - Column names are stripped of whitespace immediately, because CICIDS2017
      headers contain leading spaces (e.g. ' Flow Duration'). Doing this at
      load time prevents silent KeyErrors for the rest of the project.
    - low_memory=False avoids pandas mixed-dtype warnings on these files.
    - A 'source_file' column is added so we can always trace a row back to
      its origin (useful for auditing and for the thesis appendix).
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(
            f"Dataset directory not found:\n  {directory}\n"
            f"Check the path in configs/config.yaml"
        )

    if filenames is None:
        filenames = list_available_files(directory)

    if len(filenames) == 0:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    frames = []
    for name in filenames:
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            available = list_available_files(directory)
            raise FileNotFoundError(
                f"File not found: {name}\n"
                f"Files actually present in this folder:\n  " +
                "\n  ".join(available)
            )

        df = pd.read_csv(path, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        df["source_file"] = name
        frames.append(df)

        if verbose:
            mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
            print(f"  Loaded {name}")
            print(f"    shape = {df.shape[0]:,} rows x {df.shape[1]} cols "
                  f"| memory = {mem_mb:.1f} MB")

    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    if verbose:
        total_mem = combined.memory_usage(deep=True).sum() / 1024 ** 2
        print(f"\n  COMBINED: {combined.shape[0]:,} rows x {combined.shape[1]} cols "
              f"| {total_mem:.1f} MB")

    return combined


def load_source_dataset(cfg: dict, verbose: bool = True) -> pd.DataFrame:
    """
    Convenience wrapper: loads the training-domain dataset (CICIDS2017)
    exactly as specified in configs/config.yaml.
    """
    return load_csv_files(
        directory=cfg["paths"]["cicids2017_dir"],
        filenames=cfg["dataset"]["source_files"],
        verbose=verbose,
    )


def describe_dataset(df: pd.DataFrame, label_col: str = "Label") -> None:
    """
    Prints a compact structural summary. This is the 'raw state' snapshot
    that goes into the thesis before any cleaning is applied.
    """
    print("=" * 64)
    print("RAW DATASET SUMMARY")
    print("=" * 64)
    print(f"Rows:    {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    feature_cols = [c for c in df.columns if c not in (label_col, "source_file")]
    numeric_cols = df[feature_cols].select_dtypes(include="number").columns
    non_numeric = [c for c in feature_cols if c not in numeric_cols]

    print(f"Feature columns:     {len(feature_cols)}")
    print(f"  numeric:           {len(numeric_cols)}")
    print(f"  non-numeric:       {len(non_numeric)}")
    if non_numeric:
        print(f"  non-numeric names: {non_numeric}")

    print("-" * 64)
    if label_col in df.columns:
        counts = df[label_col].value_counts()
        props = df[label_col].value_counts(normalize=True) * 100
        print("Class distribution:")
        for lab in counts.index:
            print(f"    {str(lab):<15} {counts[lab]:>10,}  ({props[lab]:.2f}%)")
        imbalance = counts.max() / counts.min()
        print(f"\n  Imbalance ratio (majority/minority): {imbalance:.2f} : 1")
    else:
        print(f"WARNING: label column '{label_col}' not found.")
        print(f"Last 3 columns are: {list(df.columns)[-3:]}")
    print("=" * 64)