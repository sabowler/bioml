"""
Shared preprocessing utilities used across all classifiers.
"""
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def preprocess(
    df: pd.DataFrame,
    outcome_col: str,
    drop_cols: list = None,
    corr_threshold: float = 0.75,
    scale: bool = True,
):
    """
    Preprocess a dataframe for binary classification.

    Parameters
    ----------
    df : pd.DataFrame
    outcome_col : str
        Name of the binary outcome column.
    drop_cols : list, optional
        Columns to drop before modeling (e.g. patient IDs, dates).
    corr_threshold : float
        Pearson correlation threshold above which features are removed.
    scale : bool
        Whether to apply StandardScaler.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    surrogate_markers : list[str]
        Features dropped due to high correlation.
    """
    df = df.copy()

    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found.")
    y = df[outcome_col].copy()
    df = df.drop(columns=[outcome_col])

    if drop_cols:
        existing = [c for c in drop_cols if c in df.columns]
        missing = [c for c in drop_cols if c not in df.columns]
        if missing:
            logger.warning(f"drop_cols not found and skipped: {missing}")
        df = df.drop(columns=existing)
        logger.info(f"Dropped {len(existing)} user-specified columns.")

    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.warning(f"Dropping non-numeric columns: {non_numeric}")
        df = df.drop(columns=non_numeric)

    surrogate_markers = []
    if corr_threshold < 1.0:
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        surrogate_markers = [col for col in upper.columns if any(upper[col] >= corr_threshold)]
        df = df.drop(columns=surrogate_markers)
        logger.info(f"Removed {len(surrogate_markers)} correlated features (threshold={corr_threshold})")

    if scale:
        df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns, index=df.index)
        logger.info("Applied StandardScaler normalization.")

    return df, y, surrogate_markers
