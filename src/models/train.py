"""Base model instantiations and training procedures."""

from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.models.evaluate import evaluate_model


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
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
    """Train all base models and evaluate their performance on test data.

    Args:
        models: Dictionary of baseline model instances.
        preprocessor: ColumnTransformer preprocessor.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
            Trained pipeline dictionary and results metric dictionary.
    """
    trained_models: Dict[str, Pipeline] = {}
    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        print(f"Training {name}...")
        pipe = fit_single_model(name, model, preprocessor, X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_prob)
        results[name] = metrics
        trained_models[name] = pipe
        print(f"{name}: {metrics}")

    return trained_models, results
