"""
Shared plotting utilities: ROC/AUC curves with confidence intervals,
and feature importance bar charts.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import roc_curve, auc
from typing import Optional

logger = logging.getLogger(__name__)


def plot_roc(
    y_train,
    train_pred_proba,
    y_test,
    test_pred_proba,
    run_dir: str,
    model_name: str,
    label: str = "",
    confidence: float = 0.95,
) -> dict:
    """
    Plot ROC curves for train and test sets with confidence interval on test AUC.
    Saves figure to run_dir. Returns dict of AUC scores and CI bounds.
    """
    fpr_train, tpr_train, _ = roc_curve(y_train, train_pred_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, test_pred_proba)
    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)

    # Wilson-style CI on test AUC
    z = stats.norm.ppf((1 + confidence) / 2)
    ci_len = z * np.sqrt((auc_test * (1 - auc_test)) / len(y_test))
    ci_lower = max(0.0, auc_test - ci_len)
    ci_upper = min(1.0, auc_test + ci_len)

    title = f"{model_name}" + (f" — {label}" if label else "")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.plot(fpr_train, tpr_train, color="blue",
            label=f"Train (AUC={auc_train:.3f})")
    ax.plot(fpr_test, tpr_test, color="darkorange",
            label=f"Test (AUC={auc_test:.3f} [{ci_lower:.3f}, {ci_upper:.3f}])")
    ax.plot([0, 1], [0, 1], "r--", label="Chance")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {title}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fname = f"{model_name}.AUC.png" if not label else f"{model_name}.{label}.AUC.png"
    fpath = os.path.join(run_dir, fname)
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC plot saved: {fpath} | Test AUC={auc_test:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")

    return {
        "auc_train": auc_train,
        "auc_test": auc_test,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def plot_feature_importance(
    feature_names,
    importances,
    run_dir: str,
    model_name: str,
    label: str = "",
    top_n: int = 20,
    importance_col: str = "Importance",
) -> pd.DataFrame:
    """
    Bar chart of top_n most important features. Saves PNG and Excel to run_dir.
    Returns a DataFrame of all feature importances sorted descending.
    """
    fi_df = pd.DataFrame({
        "Feature": list(feature_names),
        importance_col: list(importances)
    }).sort_values(importance_col, ascending=False).reset_index(drop=True)

    top = fi_df.head(top_n)
    fname_base = f"{model_name}.FI" if not label else f"{model_name}.{label}.FI"

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)), dpi=150)
    ax.barh(top["Feature"][::-1], top[importance_col][::-1], color="steelblue")
    ax.set_xlabel(importance_col)
    ax.set_title(f"Feature Importance (Top {top_n}) — {model_name}" + (f" {label}" if label else ""))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    png_path = os.path.join(run_dir, f"{fname_base}.png")
    xlsx_path = os.path.join(run_dir, f"{fname_base}.xlsx")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    fi_df.to_excel(xlsx_path, index=False)
    logger.info(f"Feature importance saved: {png_path}, {xlsx_path}")

    return fi_df


def save_surrogate_markers(surrogate_markers: list, run_dir: str, model_name: str) -> None:
    """Save list of surrogate marker (dropped correlated feature) names to Excel."""
    if not surrogate_markers:
        return
    path = os.path.join(run_dir, f"{model_name}.SurrogateMarkers.xlsx")
    pd.DataFrame(surrogate_markers, columns=["Feature"]).to_excel(path, index=False)
    logger.info(f"Surrogate markers saved: {path}")
