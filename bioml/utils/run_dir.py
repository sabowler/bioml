"""Creates a timestamped output directory for each run."""
import os
from datetime import datetime


def make_run_dir(base: str, model_name: str) -> str:
    """
    Create and return a timestamped run directory.

    Parameters
    ----------
    base : str
        Root output directory (e.g. "./results").
    model_name : str
        Name of the model (e.g. "XGB", "SVM").

    Returns
    -------
    str
        Path to the created directory, e.g. "./results/XGB_2025-03-07_14-22-01"
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base, f"{model_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
