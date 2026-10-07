"""Exploratory Data Analysis (EDA) pipeline execution script."""

from pathlib import Path
import pandas as pd

from src.config import (
    DATA_PROCESSED_PATH,
    DATA_RAW_PATH,
    FIGURES_DIR,
    TARGET_COLUMN,
)
from src.data.load_data import load_arff
from src.data.preprocess import (
    decode_bytes_columns,
    get_numeric_categorical_columns,
    remove_duplicates,
)
from src.visualization.plots import (
    plot_boxplots,
    plot_class_distribution,
    plot_correlation_matrix,
)


def run_eda(
    raw_path: Path = DATA_RAW_PATH,
    processed_path: Path = DATA_PROCESSED_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """Run full Exploratory Data Analysis workflow.

    Loads ARFF raw data, decodes byte strings, removes duplicates,
    generates summary plots, and saves processed CSV.

    Args:
        raw_path: Path to raw ARFF dataset.
        processed_path: Path to output processed CSV.
        figures_dir: Path to directory for saving generated figures.

    Returns:
        pd.DataFrame: Cleaned and processed DataFrame.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    boxplots_dir = figures_dir / "boxplots"
    boxplots_dir.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw dataset from {raw_path}...")
    df = load_arff(raw_path)
    print(f"Raw dataset shape: {df.shape}")

    print("Decoding byte string columns...")
    df = decode_bytes_columns(df)

    print("Removing duplicate records...")
    df = remove_duplicates(df)
    print(f"Shape after removing duplicates: {df.shape}")

    print(f"Target '{TARGET_COLUMN}' class counts:")
    print(df[TARGET_COLUMN].value_counts())
    print(f"\nTarget '{TARGET_COLUMN}' class proportions:")
    print(df[TARGET_COLUMN].value_counts(normalize=True))

    numeric_cols, categorical_cols = get_numeric_categorical_columns(
        df.drop(columns=[TARGET_COLUMN])
    )
    print(f"Identified {len(numeric_cols)} numerical features and {len(categorical_cols)} categorical features.")

    print("Generating boxplots...")
    plot_boxplots(df, numeric_cols, output_dir=boxplots_dir)

    print("Generating class distribution plot...")
    plot_class_distribution(
        df,
        target_col=TARGET_COLUMN,
        output_path=figures_dir / "class_distribution.png",
    )

    print("Generating correlation matrix heatmap...")
    plot_correlation_matrix(
        df,
        output_path=figures_dir / "correlation_matrix.png",
    )

    print(f"Saving cleaned dataset to {processed_path}...")
    df.to_csv(processed_path, index=False)
    print("EDA Pipeline completed successfully!")

    return df


if __name__ == "__main__":
    run_eda()
