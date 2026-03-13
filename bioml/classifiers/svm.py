"""
Support Vector Classifier with RFECV feature selection and GridSearchCV tuning.

Based on original work by Scott A. Bowler, Ndhlovu Lab, Weill Cornell Medicine.
"""
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFECV

from bioml.utils.io import load_data, setup_run_dir, load_config
from bioml.utils.preprocessing import preprocess
from bioml.plotting.roc import plot_roc
from bioml.plotting.importance import plot_feature_importance

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "test_size": 0.33,
    "random_state": 42,
    "corr_threshold": 0.75,
    "drop_cols": [],
    "outcome_col": "Case-control status",
    "output_dir": "results",
    "cv_folds": 5,
    "param_grid": {
        "estimator__kernel": ["rbf", "poly", "linear", "sigmoid"],
        "estimator__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "estimator__gamma": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
}


def run_svm(
    data_path: str,
    outcome_col: str = None,
    drop_cols: list = None,
    config: str = None,
    output_dir: str = "results",
) -> dict:
    """
    Run SVM classifier with RFECV feature selection and GridSearchCV tuning.

    Parameters
    ----------
    data_path : str
        Path to CSV or Excel input file.
    outcome_col : str, optional
        Binary outcome column name.
    drop_cols : list, optional
        Columns to exclude before modeling.
    config : str, optional
        Path to YAML config file.
    output_dir : str
        Base directory for timestamped output folder.

    Returns
    -------
    dict with keys: run_dir, train_auc, test_auc, ci_lower, ci_upper,
                    best_params, selected_features, model
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(load_config(config))
    if outcome_col:
        cfg["outcome_col"] = outcome_col
    if drop_cols:
        cfg["drop_cols"] = drop_cols

    run_dir = setup_run_dir(output_dir, "svm")
    logger.info("=== SVM Run Started ===")

    df = load_data(data_path)
    X, y, surrogates = preprocess(
        df,
        outcome_col=cfg["outcome_col"],
        drop_cols=cfg["drop_cols"],
        corr_threshold=cfg["corr_threshold"],
    )

    if surrogates:
        pd.DataFrame(surrogates, columns=["Feature"]).to_csv(
            run_dir / "surrogate_markers.csv", index=False
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["random_state"]
    )

    logger.info("Running RFECV + GridSearchCV...")
    estimator = SVC(probability=True)
    selector = RFECV(estimator, step=1, cv=StratifiedKFold(cfg["cv_folds"]), scoring="accuracy", n_jobs=-1)
    clf = GridSearchCV(selector, param_grid=cfg["param_grid"], cv=cfg["cv_folds"], n_jobs=-1).fit(
        X_train, y_train
    )

    logger.info(f"Best params: {clf.best_params_}")
    logger.info(f"Features selected: {clf.best_estimator_.n_features_}")

    model_path = run_dir / "svm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    # Feature importance via coefficients (linear kernel only)
    selected_features = list(X_train.columns[clf.best_estimator_.support_])
    try:
        coefs = clf.best_estimator_.estimator_.coef_[0].tolist()
        fi = plot_feature_importance(selected_features, coefs, "SVM", run_dir)
    except AttributeError:
        logger.warning("Coefficient-based importance not available for this kernel. Skipping FI plot.")
        fi = pd.DataFrame({"Feature": selected_features})

    X_train_sel = clf.best_estimator_.transform(X_train)
    X_test_sel = clf.best_estimator_.transform(X_test)
    train_prob = clf.best_estimator_.estimator_.predict_proba(X_train_sel)[:, 1]
    test_prob = clf.best_estimator_.estimator_.predict_proba(X_test_sel)[:, 1]

    auc_results = plot_roc(y_train, train_prob, y_test, test_prob, "SVM", run_dir / "SVM.ROC.png")

    logger.info("=== SVM Run Complete ===")
    return {
        "run_dir": str(run_dir),
        "model": clf,
        "best_params": clf.best_params_,
        "selected_features": selected_features,
        "feature_importance": fi,
        **auc_results,
    }
