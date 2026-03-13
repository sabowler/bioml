"""
XGBoost classifier with cross-validated hyperparameter tuning and SHAP-based RFE.

Based on original work by Scott A. Bowler, Ndhlovu Lab, Weill Cornell Medicine.
"""
import logging
import pickle
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split

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
    "param_grid": {
        "n_estimators": [50, 100, 150, 200],
        "eta": [0.001, 0.01, 0.1, 1],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 1, 3, 5],
        "subsample": [0.5, 0.75, 1.0],
        "lambda": [0, 3, 5],
        "alpha": [0, 3, 5],
    },
    "cv_folds": 10,
    "early_stopping_rounds": 6,
}


def run_xgb(
    data_path: str,
    outcome_col: str = None,
    drop_cols: list = None,
    config: str = None,
    output_dir: str = "results",
) -> dict:
    """
    Run XGBoost classifier with cross-validated hyperparameter tuning.

    Parameters
    ----------
    data_path : str
        Path to CSV or Excel input file.
    outcome_col : str, optional
        Binary outcome column name. Overrides config if provided.
    drop_cols : list, optional
        Columns to exclude (e.g. IDs). Overrides config if provided.
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

    run_dir = setup_run_dir(output_dir, "xgb")
    logger.info("=== XGBoost Run Started ===")

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

    # Cross-validated hyperparameter search
    logger.info("Running cross-validated hyperparameter search...")
    pg = cfg["param_grid"]
    grid_rows = []
    import itertools
    keys = list(pg.keys())
    for combo in itertools.product(*pg.values()):
        params = dict(zip(keys, combo))
        params["eval_metric"] = "auc"
        params["n_estimators"] = params.get("n_estimators", 100)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        cv_result = xgb.cv(
            dtrain=dtrain,
            params=params,
            nfold=cfg["cv_folds"],
            metrics="auc",
            seed=cfg["random_state"],
            verbose_eval=False,
        )
        row = params.copy()
        row["valid_auc_mean"] = cv_result["test-auc-mean"].iloc[-1]
        row["valid_auc_std"] = cv_result["test-auc-std"].iloc[-1]
        grid_rows.append(row)

    grid_df = pd.DataFrame(grid_rows).sort_values(
        ["valid_auc_mean", "valid_auc_std"], ascending=[False, True]
    )
    grid_df.to_excel(run_dir / "xgb_cv_results.xlsx", index=False)
    best_params = {k: v for k, v in grid_df.iloc[0].items()
                   if k not in ("valid_auc_mean", "valid_auc_std")}
    logger.info(f"Best params: {best_params}")

    model = xgb.XGBClassifier(**best_params, random_state=cfg["random_state"]).fit(
        X_train, y_train
    )

    # Save model
    model_path = run_dir / "xgb_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Evaluate
    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    auc_results = plot_roc(y_train, train_prob, y_test, test_prob, "XGBoost", run_dir / "XGBoost.ROC.png")

    # Feature importance
    fi = plot_feature_importance(
        list(X_train.columns),
        model.feature_importances_.tolist(),
        "XGBoost",
        run_dir,
    )

    logger.info("=== XGBoost Run Complete ===")
    return {
        "run_dir": str(run_dir),
        "model": model,
        "best_params": best_params,
        "selected_features": list(X_train.columns),
        "feature_importance": fi,
        **auc_results,
    }
