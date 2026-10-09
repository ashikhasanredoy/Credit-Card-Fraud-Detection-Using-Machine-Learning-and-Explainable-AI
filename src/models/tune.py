"""Hyperparameter optimization using RandomizedSearchCV."""

from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger

logger = get_logger("fraud_detection.tune")


def random_search_tune(
    pipeline: Pipeline,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 10,
    cv: int = 3,
    scoring: str = "roc_auc",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[Any, Dict[str, Any]]:
    """Execute RandomizedSearchCV tuning on a given pipeline.

    Args:
        pipeline: Estimator or Pipeline to tune.
        param_grid: Hyperparameter distribution dictionary.
        X_train: Training features DataFrame.
        y_train: Training labels Series.
        n_iter: Number of parameter settings sampled. Defaults to 10.
        cv: Cross-validation generator or fold count. Defaults to 3.
        scoring: Evaluation metric string. Defaults to "roc_auc".
        random_state: Random state seed. Defaults to 42.
        n_jobs: Number of parallel jobs. Defaults to -1.

    Returns:
        Tuple[Any, Dict[str, Any]]: Best estimator pipeline and best parameters dict.
    """
    logger.info(
        f"Starting RandomizedSearchCV tuning: n_iter={n_iter}, cv={cv}, scoring={scoring}..."
    )
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    search.fit(X_train, y_train)
    logger.info(f"RandomizedSearchCV completed. Best score: {search.best_score_:.4f}")
    logger.info(f"Best Parameters: {search.best_params_}")
    print(f"Best Parameters: {search.best_params_}")
    return search.best_estimator_, search.best_params_
