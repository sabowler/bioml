"""
Shared ROC/AUC plotting with 95% confidence intervals.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)


def compute_auc_ci(y_true, y_prob, confidence: float = 0.95):
    """
    Compute AUC and confidence interval using normal approximation.

    Returns
    -------
    score : float
    ci_lower : float
    ci_upper : float
    fpr : array
    tpr : array
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    score = auc(fpr, tpr)
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * np.sqrt((score * (1 - score)) / len(y_true))
    ci_lower = max(score - margin, 0.0)
    ci_upper = min(score + margin, 1.0)
    return score, ci_lower, ci_upper, fpr, tpr


def plot_roc(
    y_train,
    train_prob,
    y_test,
    test_prob,
    model_name: str,
    output_path: Path,
    confidence: float = 0.95,
):
    """
    Plot ROC curves for train and test sets with confidence interval on test AUC.

    Parameters
    ----------
    y_train, train_prob : train labels and predicted probabilities
    y_test, test_prob : test labels and predicted probabilities
    model_name : str used in title
    output_path : Path to save PNG
    confidence : CI level, default 0.95
    """
    train_auc, _, _, fpr_train, tpr_train = compute_auc_ci(y_train, train_prob, confidence)
    test_auc, ci_lo, ci_hi, fpr_test, tpr_test = compute_auc_ci(y_test, test_prob, confidence)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.plot(fpr_train, tpr_train, color="blue", label=f"Train (AUC={train_auc:.3f})")
    ax.plot(
        fpr_test,
        tpr_test,
        color="darkorange",
        label=f"Test (AUC={test_auc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}])",
    )
    ax.plot([0, 1], [0, 1], "r--", label="Chance")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    logger.info(
        f"ROC saved to {output_path} | Train AUC={train_auc:.3f} | "
        f"Test AUC={test_auc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]"
    )
    return {"train_auc": train_auc, "test_auc": test_auc, "ci_lower": ci_lo, "ci_upper": ci_hi}
