"""Configuration constants, file paths, and hyperparameter grids for fraud detection."""

from pathlib import Path
from typing import Any, Dict
import numpy as np

# Project root directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Reproducibility & Splitting
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
TARGET_COLUMN: str = "Class"

# Directory & File Paths
DATA_RAW_PATH: Path = PROJECT_ROOT / "data" / "raw" / "dataset.arff"
DATA_PROCESSED_PATH: Path = PROJECT_ROOT / "data" / "processed" / "eda.csv"
MODEL_DIR: Path = PROJECT_ROOT / "models"
BEST_MODEL_PATH: Path = MODEL_DIR / "best_fraud_model.pkl"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
BOXPLOTS_DIR: FIGURES_DIR / "boxplots"
TABLES_DIR: Path = REPORTS_DIR / "tables"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"
SUBMISSIONS_DIR: Path = OUTPUTS_DIR / "submissions"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Hyperparameter search grids for base models
PARAM_GRID: Dict[str, Dict[str, Any]] = {
    "LogisticRegression": {
        "model__C": [0.01, 0.1, 1],
        "model__solver": ["lbfgs", "liblinear"],
        "model__class_weight": ["balanced"],
    },
    "DecisionTree": {
        "model__max_depth": [10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 5],
        "model__class_weight": ["balanced"],
    },
    "RandomForest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__class_weight": ["balanced"],
    },
    "XGBoost": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
    },
    "LightGBM": {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__num_leaves": [31, 50, 100],
    },
    "CatBoost": {
        "model__iterations": [100, 200, 300],
        "model__depth": [4, 6, 8],
        "model__learning_rate": [0.01, 0.05, 0.1],
    },
}


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Set random seed for reproducibility across libraries.

    Args:
        seed: Integer random seed.
    """
    np.random.seed(seed)
