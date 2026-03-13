"""
RunContext: manages timestamped output directories and logging for each run.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


class RunContext:
    """
    Creates a timestamped output directory for a single classifier run
    and wires up a logger that writes to both the console and a log file.

    Example
    -------
    with RunContext("xgb", base_dir="outputs") as ctx:
        ctx.logger.info("Training started")
        model.save(ctx.path("model.pkl"))
    """

    def __init__(self, model_name: str, base_dir: str = "outputs"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / f"{model_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.logger = self._build_logger()

    def path(self, filename: str) -> Path:
        """Return a full path inside the run directory."""
        return self.run_dir / filename

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"bioml.{self.model_name}.{id(self)}")
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        # File handler
        fh = logging.FileHandler(self.path("run.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        return logger

    def __enter__(self):
        self.logger.info(f"Run directory: {self.run_dir}")
        return self

    def __exit__(self, *_):
        self.logger.info("Run complete.")
        # Remove duplicate handlers to avoid log bleed in long sessions
        self.logger.handlers.clear()
