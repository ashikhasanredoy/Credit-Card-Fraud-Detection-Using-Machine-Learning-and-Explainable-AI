from typing import Any, Dict, Optional, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.models.evaluate import evaluate_model, evaluate_pipeline
from src.utils.logger import get_logger

logger = get_logger("fraud_detection.train")


def build_models() -> Dict[str, Any]:
    """Instantiate dictionary of baseline machine learning classifiers.

    Returns:
        Dict[str, Any]: Mapping of model names to classifier instances.
    """
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, class_weight="balanced"
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            verbose=0,
            auto_class_weights="Balanced",
        ),
    }


def fit_single_model(
    name: str,
    model: Any,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Construct and fit a single model pipeline with preprocessor.

    Args:
        name: Name of the model.
        model: Classifier instance.
        preprocessor: ColumnTransformer preprocessor.
        X_train: Training feature DataFrame.
        y_train: Training labels Series.

    Returns:
        Pipeline: Fitted scikit-learn Pipeline.
    """
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    return pipe


def train_all_models(
    models: Dict[str, Any],
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
    """Train all base models and evaluate their performance.

    Args:
        models: Dictionary of baseline model instances.
        preprocessor: ColumnTransformer preprocessor.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        X_val: Optional validation features.
        y_val: Optional validation labels.

    Returns:
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
            Trained pipeline dictionary and results metric dictionary (evaluating test or all splits).
    """
    trained_models: Dict[str, Pipeline] = {}
    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        logger.info(f"Training base model: {name}...")
        print(f"Training {name}...")
        pipe = fit_single_model(name, model, preprocessor, X_train, y_train)
        
        train_metrics = evaluate_pipeline(pipe, X_train, y_train)
        test_metrics = evaluate_pipeline(pipe, X_test, y_test)
        
        if X_val is not None and y_val is not None:
            val_metrics = evaluate_pipeline(pipe, X_val, y_val)
            logger.info(
                f"[{name}] Train F1: {train_metrics['f1']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f}, Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
                f"Test F1: {test_metrics['f1']:.4f}, Test ROC-AUC: {test_metrics['roc_auc']:.4f}"
            )
        else:
            logger.info(f"[{name}] Test Metrics: {test_metrics}")

        results[name] = test_metrics
        trained_models[name] = pipe
        print(f"{name}: {test_metrics}")

    return trained_models, results


def train_and_evaluate_all_splits(
    models: Dict[str, Any],
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[
    Dict[str, Pipeline],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
]:
    """Train all models and return separate score dictionaries for train, validation, and test splits.

    Args:
        models: Dictionary of baseline model instances.
        preprocessor: ColumnTransformer preprocessor.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Tuple of (trained_models, train_scores, val_scores, test_scores).
    """
    trained_models: Dict[str, Pipeline] = {}
    train_scores: Dict[str, Dict[str, float]] = {}
    val_scores: Dict[str, Dict[str, float]] = {}
    test_scores: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        logger.info(f"Fitting model pipeline: {name} on training set (N={len(X_train)})...")
        print(f"Training {name}...")
        pipe = fit_single_model(name, model, preprocessor, X_train, y_train)
        
        train_m = evaluate_pipeline(pipe, X_train, y_train)
        val_m = evaluate_pipeline(pipe, X_val, y_val)
        test_m = evaluate_pipeline(pipe, X_test, y_test)

        train_scores[name] = train_m
        val_scores[name] = val_m
        test_scores[name] = test_m
        trained_models[name] = pipe

        logger.info(
            f"Evaluated {name} -> Train F1: {train_m['f1']:.4f}, Val F1: {val_m['f1']:.4f}, Test F1: {test_m['f1']:.4f}"
        )
        print(f"  {name} | Val ROC-AUC: {val_m['roc_auc']:.4f} | Val F1: {val_m['f1']:.4f} | Test F1: {test_m['f1']:.4f}")

    return trained_models, train_scores, val_scores, test_scores
