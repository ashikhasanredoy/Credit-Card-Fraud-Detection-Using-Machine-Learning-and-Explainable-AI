"""Unit tests for performance metrics evaluation functions."""

import numpy as np
from src.models.evaluate import evaluate_model


def test_evaluate_model_metrics() -> None:
    """Test evaluate_model returns expected 5 metric keys with bounded numeric values."""
    y_true = np.array([0, 1, 0, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.05, 0.85, 0.4, 0.15, 0.25, 0.95, 0.05])

    metrics = evaluate_model(y_true, y_pred, y_prob)

    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    for metric_name, value in metrics.items():
        assert isinstance(value, float), f"{metric_name} is not a float"
        assert 0.0 <= value <= 1.0, f"{metric_name} value {value} is out of bounds [0, 1]"

    assert metrics["recall"] == 0.75  # 3 detected out of 4 true frauds
    assert metrics["precision"] == 1.0  # 3 true positives out of 3 predictions
