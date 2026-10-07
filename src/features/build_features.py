"""Feature engineering and transformer pipeline constructions."""

from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def build_numeric_pipeline() -> Pipeline:
    """Build preprocessing pipeline for numerical features.

    Imputes missing values using median strategy and scales using RobustScaler.

    Returns:
        Pipeline: Scikit-learn Pipeline with SimpleImputer and RobustScaler.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )


def build_preprocessor(numeric_cols: List[str]) -> ColumnTransformer:
    """Build ColumnTransformer applying numeric pipeline on specified numeric columns.

    Args:
        numeric_cols: List of numerical column names.

    Returns:
        ColumnTransformer: Configured preprocessor.
    """
    numeric_pipe = build_numeric_pipeline()
    return ColumnTransformer([("num", numeric_pipe, numeric_cols)])
