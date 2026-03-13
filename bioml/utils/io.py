"""
I/O utilities: data loading, run directory creation, config loading.
"""
import logging
import os
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    """Load CSV or Excel file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .csv or .xlsx")
    df = df.fillna(0)
    logger.info(f"Loaded {len(df)} rows x {len(df.columns)} columns from {path.name}")
    return df


def setup_run_dir(base_dir: str, model_name: str) -> Path:
    """
    Create a timestamped output directory for a run.

    e.g. results/xgb_2025-03-07_14-30-00/
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_dir) / f"{model_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir)
    logger.info(f"Run directory: {run_dir}")
    return run_dir


def _setup_logging(run_dir: Path):
    """Configure logging to both console and a run-specific log file."""
    log_path = run_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
        force=True,
    )


def load_config(config_path: str) -> dict:
    """Load a YAML config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {path.name}")
    return config
