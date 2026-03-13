"""
Run directory management — creates a timestamped output folder per run.
"""
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def create_run_dir(base_dir: str = "runs", model_name: str = "model") -> Path:
    """Create a timestamped output directory for a single run.

    Args:
        base_dir:   Root directory for all runs.
        model_name: Model identifier included in folder name.

    Returns:
        Path to the created run directory.

    Example output:
        runs/xgb_2025-03-07_14-32-01/
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_dir) / f"{model_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run output directory: {run_dir}")
    return run_dir


def setup_logging(run_dir: Path, level: int = logging.INFO) -> None:
    """Configure logging to both console and a file in the run directory."""
    log_file = run_dir / "run.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
        force=True,
    )
