"""
K-Nearest Neighbors classifier with SelectKBest feature selection and GridSearchCV.

Based on original work by Scott A. Bowler, Ndhlovu Lab, Weill Cornell Medicine.
"""
import logging
import pickle
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

from bioml.utils.io import load_data, setup_run_dir, load_config
from bioml.utils.preprocessing import preprocess
from bioml.plotting.roc import plot_roc
from bioml.plotting.importance import plot_feature_importance

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "test_size": 0.33,
    "random_state": 42,
    "corr_threshold": 0.65,
    "drop_cols": [],
    "outcome_col": "Case-control status",
    "output_dir": "results",
    "param_grid": {
        "feature_selection__k": [3, 4, 5, 6, 7, 8, 9, 10],
        "KNN__n_neighbors": [2, 3, 4, 5],
    },
}


def run_knn(
    data_path: str,
    outcome_col: str = None,
    drop_cols: list = None,
    config: str = None,
    output_dir: str = "results",
) -> dict:
    """
    Run KNN classifier with SelectKBest feature selection and GridSearchCV tuning.

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

    run_dir = setup_run_dir(output_dir, "knn")
    logger.info("=== KNN Run Started ===")

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

    logger.info("Running SelectKBest + GridSearchCV pipeline...")
    pipeline = Pipeline([
        ("feature_selection", SelectKBest()),
        ("KNN", KNeighborsClassifier()),
    ])
    best_model = GridSearchCV(pipeline, param_grid=cfg["param_grid"]).fit(X_train, y_train)
    logger.info(f"Best params: {best_model.best_params_}")

    model_path = run_dir / "knn_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    selected_features = list(
        best_model.best_estimator_.named_steps["feature_selection"].get_feature_names_out()
    )
    scores = best_model.best_estimator_.named_steps["feature_selection"].scores_
    selected_scores = [scores[i] for i, name in enumerate(X_train.columns) if name in selected_features]
    fi = plot_feature_importance(selected_features, selected_scores, "KNN", run_dir)

    train_prob = best_model.predict_proba(X_train)[:, 1]
    test_prob = best_model.predict_proba(X_test)[:, 1]
    auc_results = plot_roc(y_train, train_prob, y_test, test_prob, "KNN", run_dir / "KNN.ROC.png")

    logger.info("=== KNN Run Complete ===")
    return {
        "run_dir": str(run_dir),
        "model": best_model,
        "best_params": best_model.best_params_,
        "selected_features": selected_features,
        "feature_importance": fi,
        **auc_results,
    }
