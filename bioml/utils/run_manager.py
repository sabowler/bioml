"""
Manages timestamped output directories for each classifier run.
"""

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def create_run_dir(base_output_dir: str, model_name: str) -> str:
    """
    Create a timestamped output directory for a classifier run.
    Structure: <base_output_dir>/<model_name>_YYYYMMDD_HHMMSS/
    Returns the path to the created directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"{model_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Run output directory: {run_dir}")
    return run_dir


def setup_logging(run_dir: str, model_name: str) -> None:
    """Configure logging to both console and a per-run log file."""
    log_path = os.path.join(run_dir, f"{model_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger.info(f"Logging initialised. Log file: {log_path}")
