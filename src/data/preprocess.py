"""Data preprocessing, cleaning, feature splitting, and partitioning utilities."""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def decode_bytes_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Decode byte string objects into utf-8 strings for all object columns.

    Args:
        df: Input pandas DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame with decoded string columns.
    """
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=["object"]).columns:
        df_clean[col] = df_clean[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )
    return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from DataFrame and reset index.

    Args:
        df: Input pandas DataFrame.

    Returns:
        pd.DataFrame: Deduplicated DataFrame.
    """
    return df.drop_duplicates().reset_index(drop=True)


def split_features_target(
    df: pd.DataFrame, target_col: str = "Class"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features DataFrame and target series.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column. Defaults to "Class".

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and target vector y.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into train and test sets using stratified sampling on target.

    Args:
        X: Feature matrix.
        y: Target series.
        test_size: Proportion of dataset to include in test split.
        random_state: Random state seed.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def get_numeric_categorical_columns(
    X: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """Identify numeric and categorical column names from features DataFrame.

    Args:
        X: Feature matrix DataFrame.

    Returns:
        Tuple[List[str], List[str]]: List of numeric columns and categorical columns.
    """
    numeric_cols = X.select_dtypes(
        include=["int64", "int32", "float64", "float32"]
    ).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    return numeric_cols, categorical_cols
