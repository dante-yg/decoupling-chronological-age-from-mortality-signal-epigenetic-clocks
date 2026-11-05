"""Centralized path and environment configuration for the EpiClock repository.

All scripts and notebooks import path definitions from this module.
Paths can be customized by setting environment variables:
  - EPICLOCK_ROOT: Base project directory (defaults to repository root)
  - EPICLOCK_DATA_DIR: Processed data directory (defaults to PROJECT_ROOT / "data")
  - EPICLOCK_PREDICTIONS_DIR: Model and clock predictions (defaults to PROJECT_ROOT / "predictions")
  - EPICLOCK_SAVED_MODELS_DIR: Trained model artifacts (defaults to PROJECT_ROOT / "saved_models")
  - EPICLOCK_SCALERS_DIR: Preprocessing scalers (defaults to PROJECT_ROOT / "scalers")
  - EPICLOCK_FIGURES_DIR: Generated figures and tables (defaults to PROJECT_ROOT / "figures")
  - EPICLOCK_LOGS_DIR: Execution log files (defaults to PROJECT_ROOT / "logs")
  - EPICLOCK_REVIEWS_DIR: Review/revision deliverables (defaults to PROJECT_ROOT / "reviews")
  - EPICLOCK_RAW_DIR: Raw dbGaP datasets (defaults to DATA_DIR / "raw")
"""
import os
import sys
from pathlib import Path

# Resolve project root: check env variable first, fallback to repository root (directory of config.py)
PROJECT_ROOT = Path(os.environ.get("EPICLOCK_ROOT", Path(__file__).resolve().parent))

# Standard directories
DATA_DIR = Path(os.environ.get("EPICLOCK_DATA_DIR", PROJECT_ROOT / "data"))
PREDICTIONS_DIR = Path(os.environ.get("EPICLOCK_PREDICTIONS_DIR", PROJECT_ROOT / "predictions"))
SAVED_MODELS_DIR = Path(os.environ.get("EPICLOCK_SAVED_MODELS_DIR", PROJECT_ROOT / "saved_models"))
SCALERS_DIR = Path(os.environ.get("EPICLOCK_SCALERS_DIR", PROJECT_ROOT / "scalers"))
FIGURES_DIR = Path(os.environ.get("EPICLOCK_FIGURES_DIR", PROJECT_ROOT / "figures"))
LOGS_DIR = Path(os.environ.get("EPICLOCK_LOGS_DIR", PROJECT_ROOT / "logs"))
REVIEWS_DIR = Path(os.environ.get("EPICLOCK_REVIEWS_DIR", PROJECT_ROOT / "reviews"))

# Raw dbGaP datasets (controlled-access)
RAW_DIR = Path(os.environ.get("EPICLOCK_RAW_DIR", DATA_DIR / "raw"))

# Additional paths
PLOTS_DIR = Path(os.environ.get("EPICLOCK_PLOTS_DIR", PROJECT_ROOT / "plots"))
PYAGING_DIR = Path(os.environ.get("EPICLOCK_PYAGING_DIR", PROJECT_ROOT / "pyaging_data"))

# Ensure all writable output directories exist automatically
for _directory in [PREDICTIONS_DIR, SAVED_MODELS_DIR, SCALERS_DIR, FIGURES_DIR, LOGS_DIR, REVIEWS_DIR, PLOTS_DIR]:
    _directory.mkdir(parents=True, exist_ok=True)
