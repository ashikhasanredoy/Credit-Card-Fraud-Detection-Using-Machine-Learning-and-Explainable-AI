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


def upsample_minority_class(
    X: pd.DataFrame,
    y: pd.Series,
    target_minority_count: int = 1500,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Upsample minority class (Class=1) to target count using SMOTE.

    Args:
        X: Feature matrix DataFrame.
        y: Target series.
        target_minority_count: Target number of minority class samples (default 1500).
        random_state: Random state seed.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Resampled feature matrix X and target vector y.
    """
    current_count = int((y == 1).sum())
    if current_count >= target_minority_count:
        return X.copy(), y.copy()

    from imblearn.over_sampling import SMOTE

    smote = SMOTE(
        sampling_strategy={1: target_minority_count},
        random_state=random_state,
        k_neighbors=min(5, current_count - 1),
    )
    X_res, y_res = smote.fit_resample(X, y)
    X_res_df = pd.DataFrame(X_res, columns=X.columns)
    y_res_s = pd.Series(y_res, name=y.name, dtype=int)
    return X_res_df, y_res_s



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


def stratified_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Split dataset into train (70%), validation (15%), and test (15%) sets using stratified sampling.

    Args:
        X: Feature matrix DataFrame.
        y: Target series.
        train_size: Proportion of dataset for training (default 0.70).
        val_size: Proportion of dataset for validation (default 0.15).
        test_size: Proportion of dataset for testing (default 0.15).
        random_state: Random state seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
            (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        from src.exceptions import DataSplitError
        raise DataSplitError(
            f"train_size ({train_size}) + val_size ({val_size}) + test_size ({test_size}) "
            f"must sum to 1.0, got {total}"
        )


    n_total = len(X)
    n_test = int(round(n_total * test_size))
    n_val = int(round(n_total * val_size))

    # First split off the test set with exact sample count
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=n_test, random_state=random_state, stratify=y
    )

    # Second split: separate validation set with exact sample count
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=n_val,
        random_state=random_state,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test



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
