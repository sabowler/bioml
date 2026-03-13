"""YAML config loader with sensible defaults for each model."""
import os
import yaml
from typing import Any, Dict


DEFAULTS: Dict[str, Dict[str, Any]] = {
    "xgb": {
        "test_size": 0.33,
        "random_state": 42,
        "corr_threshold": 0.75,
        "drop_cols": ["Event", "Patient ID", "Case ID"],
        "n_estimators": 150,
        "eta": [0.01, 0.1, 1, 10],
        "min_child_weight": [1, 3, 5],
        "max_depth": [1, 3, 5],
        "gamma": [0, 1, 3, 5],
        "subsample": [0.5, 0.75, 1],
        "reg_lambda": [0, 1, 3, 5],
        "reg_alpha": [0, 1, 3, 5],
        "n_iter": 8,
        "min_features": 2,
        "output_dir": "./results",
    },
    "svm": {
        "test_size": 0.33,
        "random_state": 42,
        "corr_threshold": 0.65,
        "drop_cols": ["Event", "Patient ID", "Case ID", "Age"],
        "cv_folds": 5,
        "kernels": ["rbf", "poly", "linear", "sigmoid"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "gamma": [0.001, 0.01, 0.1, 1, 10, 100],
        "output_dir": "./results",
    },
    "knn": {
        "test_size": 0.33,
        "random_state": 42,
        "corr_threshold": 0.65,
        "drop_cols": ["Event", "Patient ID", "Case ID"],
        "k_features": [3, 4, 5, 6, 7, 8, 9, 10],
        "n_neighbors": [2, 3, 4],
        "output_dir": "./results",
    },
    "nn": {
        "test_size": 0.33,
        "random_state": 42,
        "corr_threshold": 0.75,
        "drop_cols": ["Event", "Patient ID", "Case ID"],
        "epochs": 150,
        "batch_size": 10,
        "optimizer": "adam",
        "output_dir": "./results",
    },
}


def load_config(model: str, config_path: str = None) -> Dict[str, Any]:
    """
    Load config for a given model, optionally overriding defaults with a YAML file.

    Parameters
    ----------
    model : str
        One of 'xgb', 'svm', 'knn', 'nn'.
    config_path : str, optional
        Path to a YAML config file. Keys override defaults.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    cfg = DEFAULTS.get(model, {}).copy()
    if config_path and os.path.isfile(config_path):
        with open(config_path) as f:
            overrides = yaml.safe_load(f) or {}
        cfg.update(overrides)
    return cfg
