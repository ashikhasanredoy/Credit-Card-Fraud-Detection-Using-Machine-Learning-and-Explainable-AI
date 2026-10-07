"""SHAP (SHapley Additive exPlanations) explainability workflows and visualizations."""

from pathlib import Path
from typing import Any, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def build_explainer(model: Any) -> shap.TreeExplainer:
    """Build a TreeExplainer for a tree-based model.

    Args:
        model: Fitted tree-based model (e.g. LightGBM, XGBoost, RandomForest).

    Returns:
        shap.TreeExplainer: Initialized SHAP tree explainer.
    """
    return shap.TreeExplainer(model)


def compute_shap_values(
    explainer: shap.TreeExplainer, X_transformed: np.ndarray
) -> np.ndarray:
    """Compute SHAP values for transformed features matrix.

    Args:
        explainer: Fitted SHAP TreeExplainer.
        X_transformed: Transformed feature array.

    Returns:
        np.ndarray: Computed SHAP values.
    """
    return explainer.shap_values(X_transformed)


def plot_shap_summary(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Generate and save standard SHAP summary plot.

    Args:
        shap_values: Computed SHAP values.
        X_transformed: Transformed feature array.
        feature_names: List of feature names.
        output_path: File path to save the generated plot.
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_bar(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Generate and save SHAP global feature importance bar plot.

    Args:
        shap_values: Computed SHAP values.
        X_transformed: Transformed feature array.
        feature_names: List of feature names.
        output_path: File path to save the generated plot.
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_waterfall(
    explainer: shap.TreeExplainer,
    X_transformed: np.ndarray,
    index: int = 0,
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Generate and save SHAP waterfall plot for an individual transaction sample.

    Args:
        explainer: Fitted SHAP TreeExplainer.
        X_transformed: Transformed feature array.
        index: Sample row index to explain. Defaults to 0.
        output_path: File path to save the generated plot.
    """
    plt.figure(figsize=(10, 6))
    shap_exp = explainer(X_transformed)
    shap.plots.waterfall(shap_exp[index], show=False)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_dependence(
    feature_name: str,
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Generate and save SHAP dependence plot for a selected feature.

    Args:
        feature_name: Target feature name for dependence visualization.
        shap_values: Computed SHAP values.
        X_transformed: Transformed feature array.
        feature_names: List of feature names.
        output_path: File path to save the generated plot.
    """
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        feature_name,
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_top_shap_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    n: int = 15,
) -> pd.DataFrame:
    """Compute and rank top N global mean absolute SHAP importances.

    Args:
        shap_values: Computed SHAP values.
        feature_names: List of feature names.
        n: Number of top features to return. Defaults to 15.

    Returns:
        pd.DataFrame: Ranked dataframe of top feature importances.
    """
    # In case multi-output shap_values is returned for binary classification
    if isinstance(shap_values, list):
        vals = shap_values[1]
    elif len(shap_values.shape) == 3:
        vals = shap_values[:, :, 1]
    else:
        vals = shap_values

    importance = np.abs(vals).mean(axis=0)
    shap_importance = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance}
    )
    shap_importance = shap_importance.sort_values("Importance", ascending=False)
    return shap_importance.head(n).reset_index(drop=True)
