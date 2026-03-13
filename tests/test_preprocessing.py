"""Tests for shared preprocessing module."""
import pytest
import pandas as pd
import numpy as np
from bioml.utils.preprocessing import preprocess


def make_df(n=50, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "feat_a": rng.normal(0, 1, n),
        "feat_b": rng.normal(0, 1, n),
        "feat_c": rng.normal(0, 1, n),
        "id_col": range(n),
        "outcome": rng.integers(0, 2, n),
    })
    # Make feat_b highly correlated with feat_a
    df["feat_b"] = df["feat_a"] + rng.normal(0, 0.01, n)
    return df


def test_outcome_extracted():
    df = make_df()
    X, y, _ = preprocess(df, outcome_col="outcome", drop_cols=["id_col"])
    assert "outcome" not in X.columns
    assert len(y) == len(df)


def test_drop_cols_removed():
    df = make_df()
    X, y, _ = preprocess(df, outcome_col="outcome", drop_cols=["id_col"])
    assert "id_col" not in X.columns


def test_correlation_filter():
    df = make_df()
    X, y, surrogates = preprocess(df, outcome_col="outcome", drop_cols=["id_col"], corr_threshold=0.75)
    # feat_b should be dropped as a surrogate of feat_a
    assert "feat_b" in surrogates


def test_scaling_applied():
    df = make_df()
    X, y, _ = preprocess(df, outcome_col="outcome", drop_cols=["id_col"], scale=True)
    # Scaled features should have near-zero mean
    assert abs(X["feat_a"].mean()) < 0.1


def test_missing_outcome_raises():
    df = make_df()
    with pytest.raises(ValueError, match="Outcome column"):
        preprocess(df, outcome_col="nonexistent")


def test_missing_drop_col_warns(caplog):
    df = make_df()
    import logging
    with caplog.at_level(logging.WARNING):
        preprocess(df, outcome_col="outcome", drop_cols=["not_a_column"])
    assert "not_a_column" in caplog.text
