"""Tests for ROC/AUC plotting utilities."""
import numpy as np
import pytest
from bioml.plotting.roc import compute_auc_ci


def test_auc_ci_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0.1, 0.2, 0.8, 0.9])
    auc_score, lower, upper, fpr, tpr = compute_auc_ci(y, pred)
    assert auc_score == pytest.approx(1.0)
    assert 0.0 <= lower <= upper <= 1.0


def test_auc_ci_chance():
    np.random.seed(0)
    y = np.array([0, 1] * 50)
    pred = np.random.rand(100)
    auc_score, lower, upper, fpr, tpr = compute_auc_ci(y, pred)
    assert 0.0 <= auc_score <= 1.0
    assert lower <= auc_score <= upper


def test_ci_bounds_clamped():
    """CI should never go outside [0, 1]."""
    y = np.array([0, 1])
    pred = np.array([0.0, 1.0])
    _, lower, upper, fpr, tpr = compute_auc_ci(y, pred)
    assert lower >= 0.0
    assert upper <= 1.0
