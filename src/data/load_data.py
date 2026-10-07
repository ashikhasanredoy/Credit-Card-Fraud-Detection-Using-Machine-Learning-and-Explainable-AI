"""Data loading utilities for ARFF and CSV formats."""

from pathlib import Path
from typing import Union
import pandas as pd
from scipy.io import arff


def load_arff(path: Union[str, Path]) -> pd.DataFrame:
    """Load an ARFF data file into a pandas DataFrame.

    Args:
        path: File path to the .arff dataset.

    Returns:
        pd.DataFrame: Loaded dataset as a DataFrame.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"ARFF file not found at: {file_path}")
    data, _ = arff.loadarff(str(file_path))
    return pd.DataFrame(data)


def load_processed_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load a processed CSV dataset.

    Args:
        path: File path to the processed .csv dataset.

    Returns:
        pd.DataFrame: Loaded dataset as a DataFrame.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Processed CSV file not found at: {file_path}")
    return pd.read_csv(file_path)
