"""Unit tests for custom exception hierarchy."""

import pytest
from src.exceptions import (
    DataLoadError,
    DataPreprocessingError,
    DataSplitError,
    FraudDetectionException,
    ModelEvaluationError,
    ModelInferenceError,
    ModelTrainingError,
)
from src.data.preprocess import stratified_train_val_test_split
import pandas as pd
import numpy as np


def test_base_fraud_detection_exception() -> None:
    """Test base exception initialization and formatting with root error."""
    root_err = ValueError("Invalid numeric value")
    try:
        raise root_err
    except ValueError as e:
        custom_err = FraudDetectionException("Failed processing feature", error_detail=e)
        assert "Failed processing feature" in str(custom_err)
        assert "ValueError" in str(custom_err)
        assert isinstance(custom_err, Exception)


def test_custom_exception_hierarchy() -> None:
    """Test inheritance of specialized exception classes from FraudDetectionException."""
    assert issubclass(DataLoadError, FraudDetectionException)
    assert issubclass(DataPreprocessingError, FraudDetectionException)
    assert issubclass(DataSplitError, FraudDetectionException)
    assert issubclass(ModelTrainingError, FraudDetectionException)
    assert issubclass(ModelEvaluationError, FraudDetectionException)
    assert issubclass(ModelInferenceError, FraudDetectionException)


def test_data_split_error_raised() -> None:
    """Test that invalid split proportions raise DataSplitError."""
    X = pd.DataFrame(np.random.randn(20, 2))
    y = pd.Series([0] * 18 + [1] * 2)

    with pytest.raises(DataSplitError) as exc_info:
        # 0.70 + 0.20 + 0.20 = 1.10 != 1.0
        stratified_train_val_test_split(X, y, train_size=0.70, val_size=0.20, test_size=0.20)

    assert "must sum to 1.0" in str(exc_info.value)
