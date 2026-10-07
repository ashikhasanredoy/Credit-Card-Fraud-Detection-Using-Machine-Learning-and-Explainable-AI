"""Data visualization routines for exploratory data analysis."""

from pathlib import Path
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_boxplots(
    df: pd.DataFrame,
    numeric_cols: List[str],
    output_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Generate and save boxplots for each numerical feature.

    Args:
        df: Input DataFrame.
        numeric_cols: List of numeric column names.
        output_dir: Directory path to save generated boxplot figures.
    """
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        plt.boxplot(df[col].dropna())
        plt.ylabel(col)
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()

        if output_dir:
            file_path = Path(output_dir) / f"boxplot_{col}.png"
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_class_distribution(
    df: pd.DataFrame,
    target_col: str = "Class",
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Generate and save class distribution bar chart.

    Args:
        df: Input DataFrame.
        target_col: Target column name. Defaults to "Class".
        output_path: File path to save the generated chart.
    """
    class_counts = df[target_col].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    class_counts.plot(kind="bar")
    plt.xlabel("Class")
    plt.ylabel("Number of Transactions")
    plt.title("Fraud vs Legitimate Transactions")
    plt.xticks(rotation=0)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlation_matrix(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Generate and save heatmap visualization of feature correlation matrix.

    Args:
        df: Input DataFrame.
        output_path: File path to save the generated correlation heatmap.
    """
    numeric_df = df.select_dtypes(include=np.number)
    correlation = numeric_df.corr()

    plt.figure(figsize=(12, 8))
    plt.imshow(correlation, aspect="auto")
    plt.colorbar()
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
