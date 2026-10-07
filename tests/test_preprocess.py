"""Unit tests for data preprocessing and splitting routines."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    decode_bytes_columns,
    get_numeric_categorical_columns,
    remove_duplicates,
    split_features_target,
    stratified_split,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Fixture providing a sample DataFrame containing byte strings and duplicates."""
    return pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "V1": [1.1, -0.5, 0.2, 0.2, -1.2, 0.9, -0.3, 1.5, -2.1, 0.4],
            "V2": [0.5, -1.1, 0.8, 0.8, -0.4, 1.2, -0.8, 0.6, -1.0, 0.2],
            "Amount": [10.0, 25.5, 100.0, 100.0, 5.0, 80.0, 12.0, 45.0, 300.0, 15.0],
            "Class": [
                b"0",
                b"0",
                b"1",
                b"1",
                b"0",
                b"0",
                b"0",
                b"1",
                b"0",
                b"0",
            ],
        }
    )


def test_decode_bytes_columns(sample_raw_df: pd.DataFrame) -> None:
    """Test byte string decoding to standard strings."""
    cleaned = decode_bytes_columns(sample_raw_df)
    assert not any(isinstance(val, bytes) for val in cleaned["Class"])
    assert cleaned["Class"].iloc[0] == "0"


def test_remove_duplicates(sample_raw_df: pd.DataFrame) -> None:
    """Test deduplication removes duplicate records."""
    assert len(sample_raw_df) == 10
    deduped = remove_duplicates(sample_raw_df)
    assert len(deduped) == 9
    assert deduped.index.tolist() == list(range(9))


def test_split_features_target(sample_raw_df: pd.DataFrame) -> None:
    """Test separating feature matrix and target series."""
    cleaned = decode_bytes_columns(sample_raw_df)
    X, y = split_features_target(cleaned, target_col="Class")
    assert "Class" not in X.columns
    assert y.name == "Class"
    assert pd.api.types.is_integer_dtype(y)
    assert len(X) == len(y) == 10


def test_stratified_split() -> None:
    """Test stratified split maintains class balance and expected dimensions."""
    X = pd.DataFrame({"feat1": np.arange(100), "feat2": np.arange(100)})
    y = pd.Series([0] * 90 + [1] * 10)

    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=0.2, random_state=42
    )

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert (y_test == 1).sum() == 2
    assert (y_train == 1).sum() == 8


def test_get_numeric_categorical_columns() -> None:
    """Test identifying numerical and categorical features."""
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0],
            "num2": [10, 20],
            "cat1": ["A", "B"],
            "cat2": [True, False],
        }
    )
    num_cols, cat_cols = get_numeric_categorical_columns(df)
    assert set(num_cols) == {"num1", "num2"}
    assert set(cat_cols) == {"cat1", "cat2"}
