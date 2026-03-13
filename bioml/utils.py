"""
bioml.utils
-----------
Shared utilities used across all classifiers:
  - Config loading
  - Preprocessing (correlation filter, scaling, train/test split)
  - ROC/AUC plotting with confidence intervals
  - Feature importance output
  - Run directory management
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import yaml
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def load_config(user_config: dict | str | Path | None = None) -> dict:
    """
    Load the default YAML config, then deep-merge any user overrides.

    Parameters
    ----------
    user_config:
        A dict of overrides, a path to a YAML file, or None to use defaults.

    Returns
    -------
    dict
    """
    with open(_DEFAULT_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    if user_config is None:
        return cfg

    if isinstance(user_config, (str, Path)):
        with open(user_config) as f:
            overrides = yaml.safe_load(f)
    else:
        overrides = user_config

    return _deep_merge(cfg, overrides)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------

def make_run_dir(base_dir: str | Path = "runs", label: str = "") -> Path:
    """
    Create a timestamped output directory.

    Example:  runs/20240315_142301_xgb/
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{label}" if label else ts
    run_dir = Path(base_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)
    return run_dir


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(run_dir: Path, level: int = logging.INFO) -> None:
    """Configure root logger to write to console and a file inside run_dir."""
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "run.log"),
        ],
    )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def load_data(path: str | Path) -> pd.DataFrame:
    """Load CSV or Excel into a DataFrame."""
    path = Path(path)
    if path.suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def preprocess(
    df: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """
    Full preprocessing pipeline:
      1. Drop configured columns
      2. Separate outcome
      3. Drop correlated features
      4. Optionally scale
      5. Train / test split

    Returns
    -------
    X_train, X_test, y_train, y_test, dropped_features
    """
    data_cfg = cfg["data"]
    outcome_col: str = data_cfg["outcome_col"]
    drop_cols: list[str] = data_cfg.get("drop_cols", [])
    corr_thresh: float = data_cfg.get("correlation_threshold", 0.75)
    test_size: float = data_cfg.get("test_size", 0.33)
    random_state: int = data_cfg.get("random_state", 42)
    scale: bool = cfg["preprocessing"].get("scale", True)

    df = df.copy()

    # Drop specified columns that exist in the dataframe
    existing_drops = [c for c in drop_cols if c in df.columns]
    df.drop(columns=existing_drops, inplace=True)

    # Separate outcome
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found. "
                         f"Available: {list(df.columns)}")
    y = df.pop(outcome_col)
    X = df

    # Correlation filter on numeric columns only
    numeric_cols = X.select_dtypes(include="number").columns
    corr_matrix = X[numeric_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    dropped = [col for col in upper.columns if any(upper[col] >= corr_thresh)]
    X.drop(columns=dropped, inplace=True)
    logger.info("Correlation filter dropped %d features: %s", len(dropped), dropped)

    # Scale
    if scale:
        numeric = X.select_dtypes(include="number").columns
        X[numeric] = StandardScaler().fit_transform(X[numeric])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Split: %d train / %d test samples, %d features",
        len(X_train), len(X_test), X_train.shape[1],
    )
    return X_train, X_test, y_train, y_test, dropped


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Compute AUC with a normal-approximation confidence interval.

    Returns
    -------
    auc_score, ci_lower, ci_upper, fpr, tpr
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * np.sqrt((auc_score * (1 - auc_score)) / len(y_true))
    ci_lower = max(0.0, auc_score - margin)
    ci_upper = min(1.0, auc_score + margin)
    return auc_score, ci_lower, ci_upper, fpr, tpr


def plot_roc(
    train_auc_data: tuple,
    test_auc_data: tuple,
    run_dir: Path,
    title: str = "ROC Curve",
    filename: str = "roc.png",
) -> None:
    """
    Save a ROC curve plot with train and test curves.

    Parameters
    ----------
    train_auc_data : (auc, ci_lo, ci_hi, fpr, tpr)
    test_auc_data  : (auc, ci_lo, ci_hi, fpr, tpr)
    """
    train_auc, _, _, fpr_tr, tpr_tr = train_auc_data
    test_auc, ci_lo, ci_hi, fpr_te, tpr_te = test_auc_data

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.plot(fpr_tr, tpr_tr, color="steelblue",
            label=f"Train  AUC = {train_auc:.3f}")
    ax.plot(fpr_te, tpr_te, color="darkorange",
            label=f"Test   AUC = {test_auc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
    ax.plot([0, 1], [0, 1], "r--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    fig.savefig(run_dir / filename)
    plt.close(fig)
    logger.info("ROC plot saved: %s", run_dir / filename)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def save_feature_importance(
    features: list[str],
    importances: list[float],
    run_dir: Path,
    filename: str = "feature_importance.csv",
    plot: bool = True,
) -> pd.DataFrame:
    """
    Save feature importances to CSV and optionally a bar chart.
    """
    fi = (
        pd.DataFrame({"feature": features, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi.to_csv(run_dir / filename, index=False)
    logger.info("Feature importance saved: %s", run_dir / filename)

    if plot:
        fig, ax = plt.subplots(figsize=(8, max(4, len(fi) * 0.35)), dpi=150)
        fi_plot = fi.head(20)  # top 20 for readability
        ax.barh(fi_plot["feature"][::-1], fi_plot["importance"][::-1],
                color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance (top 20)")
        fig.tight_layout()
        fig.savefig(run_dir / filename.replace(".csv", ".png"))
        plt.close(fig)

    return fi


def save_surrogate_markers(dropped: list[str], run_dir: Path) -> None:
    """Save the list of features dropped by correlation filtering."""
    if dropped:
        pd.DataFrame({"dropped_feature": dropped}).to_csv(
            run_dir / "surrogate_markers.csv", index=False
        )
        logger.info("%d surrogate markers saved.", len(dropped))
