"""Tests for ROC and feature importance plotting."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from bioml.plotting.roc import compute_auc_ci, plot_roc
from bioml.plotting.importance import plot_feature_importance


def test_compute_auc_ci_perfect():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    score, ci_lo, ci_hi, fpr, tpr = compute_auc_ci(y, prob)
    assert score == pytest.approx(1.0)
    assert 0.0 <= ci_lo <= ci_hi <= 1.0


def test_compute_auc_ci_chance():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 100)
    prob = rng.random(100)
    score, ci_lo, ci_hi, fpr, tpr = compute_auc_ci(y, prob)
    assert 0.0 <= score <= 1.0
    assert ci_lo <= score <= ci_hi


def test_plot_roc_saves_file(tmp_path):
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    out = tmp_path / "test_roc.png"
    result = plot_roc(y, prob, y, prob, "TestModel", out)
    assert out.exists()
    assert "train_auc" in result
    assert "test_auc" in result


def test_plot_feature_importance_saves_files(tmp_path):
    features = ["feat_a", "feat_b", "feat_c"]
    importances = [0.5, 0.3, 0.2]
    fi = plot_feature_importance(features, importances, "TestModel", tmp_path)
    assert (tmp_path / "TestModel.FeatureImportance.xlsx").exists()
    assert (tmp_path / "TestModel.FeatureImportance.png").exists()
    assert list(fi["Feature"]) == features
