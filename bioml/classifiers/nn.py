"""
Neural Network classifier using TensorFlow/Keras.

Based on original work by Scott A. Bowler, Ndhlovu Lab, Weill Cornell Medicine.
"""
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from bioml.utils.io import load_data, setup_run_dir, load_config
from bioml.utils.preprocessing import preprocess
from bioml.plotting.roc import plot_roc

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "test_size": 0.33,
    "random_state": 42,
    "corr_threshold": 0.75,
    "drop_cols": [],
    "outcome_col": "Case-control status",
    "output_dir": "results",
    "epochs": 150,
    "batch_size": 10,
    "optimizer": "adam",
    "loss": "binary_crossentropy",
}


def run_nn(
    data_path: str,
    outcome_col: str = None,
    drop_cols: list = None,
    config: str = None,
    output_dir: str = "results",
) -> dict:
    """
    Run a fully-connected Neural Network binary classifier.

    Architecture: input -> input_dim -> input_dim -> input_dim/3 -> input_dim/6 -> 1 (sigmoid)

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
    dict with keys: run_dir, train_auc, test_auc, ci_lower, ci_upper, model
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
    except ImportError:
        raise ImportError("TensorFlow is required for run_nn(). Install with: pip install tensorflow")

    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(load_config(config))
    if outcome_col:
        cfg["outcome_col"] = outcome_col
    if drop_cols:
        cfg["drop_cols"] = drop_cols

    run_dir = setup_run_dir(output_dir, "nn")
    logger.info("=== Neural Network Run Started ===")

    df = load_data(data_path)
    X, y, surrogates = preprocess(
        df,
        outcome_col=cfg["outcome_col"],
        drop_cols=cfg["drop_cols"],
        corr_threshold=cfg["corr_threshold"],
        scale=True,
    )

    if surrogates:
        pd.DataFrame(surrogates, columns=["Feature"]).to_csv(
            run_dir / "surrogate_markers.csv", index=False
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["random_state"]
    )

    n_features = X_train.shape[1]
    logger.info(f"Building network with input_dim={n_features}")

    model = Sequential([
        Dense(n_features, activation="relu", input_shape=(n_features,)),
        Dense(n_features, activation="relu"),
        Dense(max(1, n_features // 3), activation="relu"),
        Dense(max(1, n_features // 6), activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        loss=cfg["loss"],
        optimizer=cfg["optimizer"],
        metrics=["accuracy"],
    )

    logger.info(f"Training for {cfg['epochs']} epochs, batch_size={cfg['batch_size']}")
    model.fit(
        X_train, y_train,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        verbose=0,
    )

    model.save(run_dir / "nn_model.keras")

    train_prob = model.predict(X_train).flatten()
    test_prob = model.predict(X_test).flatten()
    auc_results = plot_roc(y_train, train_prob, y_test, test_prob, "Neural Network", run_dir / "NN.ROC.png")

    logger.info("=== Neural Network Run Complete ===")
    return {
        "run_dir": str(run_dir),
        "model": model,
        **auc_results,
    }
