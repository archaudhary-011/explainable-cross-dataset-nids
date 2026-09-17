"""
utils.py
--------
Shared helpers: config loading, path resolution, reproducibility,
and consistent directory creation.

Every notebook starts by calling load_config() so that paths, the random
seed, and dataset scope are defined in exactly ONE place (configs/config.yaml).
This is a reproducibility requirement: an examiner should be able to change
one file and re-run the whole study.
"""

import os
import random
import numpy as np
import yaml


def get_project_root() -> str:
    """
    Returns the absolute path of the project root folder.
    This file lives at <root>/src/utils.py, so the root is one level up.
    Using this instead of os.getcwd() means notebooks work no matter
    which folder Jupyter was launched from.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def load_config(config_name: str = "config.yaml") -> dict:
    """
    Loads configs/config.yaml and converts every path into an absolute path.
    Also creates the results/ subfolders if they do not exist.
    """
    root = get_project_root()
    config_path = os.path.join(root, "configs", config_name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at: {config_path}\n"
            f"Make sure configs/config.yaml exists."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Store the root, and convert all relative paths to absolute
    cfg["project_root"] = root
    cfg["paths"] = {k: os.path.join(root, v) for k, v in cfg["paths"].items()}

    # Create output folders (but never try to create the raw dataset folders)
    for key, path in cfg["paths"].items():
        if "datasets" not in path:
            os.makedirs(path, exist_ok=True)

    return cfg


def set_seed(seed: int = 42) -> None:
    """
    Fixes all random seeds we can control, so results are reproducible.
    Called once at the top of every notebook.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_plotting(cfg: dict) -> None:
    """Applies a consistent plot style across all figures in the thesis."""
    import matplotlib.pyplot as plt
    style = cfg.get("plotting", {}).get("style", "seaborn-v0_8-whitegrid")
    try:
        plt.style.use(style)
    except Exception:
        plt.style.use("default")
    plt.rcParams["figure.dpi"] = cfg.get("plotting", {}).get("dpi", 150)
    plt.rcParams["savefig.bbox"] = "tight"


def save_figure(fig, filename: str, cfg: dict) -> str:
    """
    Saves a matplotlib figure into results/figures with consistent settings.
    Returns the saved path so it can be referenced in the report.
    """
    path = os.path.join(cfg["paths"]["figures_dir"], filename)
    fig.savefig(path, dpi=cfg["plotting"]["dpi"], bbox_inches="tight")
    print(f"Figure saved: {path}")
    return path