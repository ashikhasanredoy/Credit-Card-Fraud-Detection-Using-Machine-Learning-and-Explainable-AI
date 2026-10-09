"""Model evaluation metrics and diagnostics."""

from typing import Any, Dict, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    y_prob: Union[np.ndarray, pd.Series, list],
) -> Dict[str, float]:
    """Calculate key classification evaluation metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted positive class probabilities.

    Returns:
        Dict[str, float]: Dictionary containing accuracy, precision, recall, f1, and roc_auc.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def evaluate_pipeline(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Evaluate a fitted pipeline or estimator on test data.

    Args:
        model: Fitted estimator or Pipeline with predict and optionally predict_proba.
        X_test: Test features DataFrame.
        y_test: Test ground truth series.

    Returns:
        Dict[str, float]: Evaluation metrics.
    """
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    return evaluate_model(y_test, y_pred, y_prob)


def print_confusion_matrix(
    y_test: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Compute and display confusion matrix.

    Args:
        y_test: True class labels.
        y_pred: Predicted class labels.

    Returns:
        np.ndarray: Computed confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    return cm


def evaluate_splits(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a fitted pipeline or estimator on train, validation, and test sets.

    Args:
        model: Fitted estimator or Pipeline.
        X_train: Training features DataFrame.
        y_train: Training ground truth series.
        X_val: Validation features DataFrame.
        y_val: Validation ground truth series.
        X_test: Test features DataFrame.
        y_test: Test ground truth series.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing metrics for 'train', 'val', and 'test'.
    """
    return {
        "train": evaluate_pipeline(model, X_train, y_train),
        "val": evaluate_pipeline(model, X_val, y_val),
        "test": evaluate_pipeline(model, X_test, y_test),
    }

