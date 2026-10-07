"""End-to-end model training, tuning, ensembling, evaluation, and explainability pipeline."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    BEST_MODEL_PATH,
    DATA_PROCESSED_PATH,
    FIGURES_DIR,
    MODEL_DIR,
    PARAM_GRID,
    PREDICTIONS_DIR,
    RANDOM_STATE,
    TABLES_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    set_seed,
)
from src.data.load_data import load_processed_csv
from src.data.preprocess import (
    get_numeric_categorical_columns,
    split_features_target,
    stratified_split,
)
from src.explainability.shap_analysis import (
    build_explainer,
    compute_shap_values,
    get_top_shap_importance,
    plot_shap_bar,
    plot_shap_dependence,
    plot_shap_summary,
    plot_shap_waterfall,
)
from src.features.build_features import build_preprocessor
from src.models.ensemble import fit_all_ensembles
from src.models.evaluate import print_confusion_matrix
from src.models.train import build_models, train_all_models
from src.models.tune import random_search_tune


def run_modeling_pipeline(
    processed_data_path: Path = DATA_PROCESSED_PATH,
    model_save_path: Path = BEST_MODEL_PATH,
    figures_dir: Path = FIGURES_DIR,
    tables_dir: Path = TABLES_DIR,
    predictions_dir: Path = PREDICTIONS_DIR,
) -> None:
    """Execute complete modeling, hyperparameter optimization, and explainability pipeline.

    Args:
        processed_data_path: Path to cleaned CSV dataset.
        model_save_path: Path to save final serialized production model.
        figures_dir: Directory to save evaluation and explainability plots.
        tables_dir: Directory to save metric comparison tables.
        predictions_dir: Directory to save test set predictions.
    """
    set_seed(RANDOM_STATE)

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading processed dataset from {processed_data_path}...")
    df = load_processed_csv(processed_data_path)


    X, y = split_features_target(df, target_col=TARGET_COLUMN)
    numeric_cols, _ = get_numeric_categorical_columns(X)

    # Preprocessor
    preprocessor = build_preprocessor(numeric_cols)

    # Stratified Train/Test split
    print(f"Splitting dataset (test_size={TEST_SIZE}, random_state={RANDOM_STATE})...")
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Build and train baseline models
    print("\n--- Training Base Models ---")
    base_models = build_models()
    trained_models, results = train_all_models(
        base_models, preprocessor, X_train, y_train, X_test, y_test
    )

    # Export model comparison table
    results_df = pd.DataFrame(results).T
    model_comparison_path = tables_dir / "model_comparison.csv"
    results_df.to_csv(model_comparison_path)
    print(f"\nSaved baseline model comparison table to: {model_comparison_path}")
    print(results_df)

    # Select best baseline model based on F1-Score
    best_model_name = str(results_df["f1"].idxmax())
    print(f"\nTop Base Model selected by F1 score: {best_model_name}")

    # Hyperparameter tuning on best model
    print(f"\nTuning hyperparameters for {best_model_name}...")
    tune_pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", base_models[best_model_name])]
    )
    best_tuned_model, best_params = random_search_tune(
        pipeline=tune_pipeline,
        param_grid=PARAM_GRID[best_model_name],
        X_train=X_train,
        y_train=y_train,
        n_iter=10,
        cv=3,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
    )

    # Build and evaluate ensemble architectures
    print("\n--- Training Ensemble Models ---")
    ensemble_models, final_results = fit_all_ensembles(
        trained_models=trained_models,
        preprocessor=preprocessor,
        best_model_name=best_model_name,
        base_models=base_models,
        best_tuned_model=best_tuned_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=RANDOM_STATE,
    )

    final_results_df = pd.DataFrame(final_results).T.sort_values(
        "roc_auc", ascending=False
    )
    final_results_path = tables_dir / "final_results.csv"
    final_results_df.to_csv(final_results_path)
    print(f"\nFinal Models Summary:\n{final_results_df}")

    # Production Model: VotingClassifier Pipeline
    best_final_model = ensemble_models["Voting"]
    y_pred = best_final_model.predict(X_test)
    y_prob = best_final_model.predict_proba(X_test)[:, 1]

    print("\nConfusion Matrix for Best Model (Voting Classifier):")
    print_confusion_matrix(y_test, y_pred)

    # Save test predictions
    predictions_df = pd.DataFrame(
        {"y_true": y_test, "y_pred": y_pred, "y_prob": y_prob}
    )
    predictions_path = predictions_dir / "y_test_pred.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved test predictions to: {predictions_path}")

    # Serialize Best Production Model
    joblib.dump(best_final_model, model_save_path)
    print(f"\nProduction model saved to: {model_save_path}")

    # Explainability with SHAP (LightGBM)
    print("\n--- Running SHAP Explainability Analysis ---")
    shap_model_pipeline = trained_models["LightGBM"]
    X_transformed = shap_model_pipeline.named_steps["preprocessor"].transform(
        X_test
    )
    feature_names = (
        shap_model_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    )
    lgb_model = shap_model_pipeline.named_steps["model"]

    explainer = build_explainer(lgb_model)
    shap_values = compute_shap_values(explainer, X_transformed)

    # SHAP Summary Plot
    summary_plot_path = figures_dir / "shap_summary.png"
    plot_shap_summary(
        shap_values, X_transformed, feature_names, output_path=summary_plot_path
    )

    # SHAP Bar Plot
    bar_plot_path = figures_dir / "shap_bar.png"
    plot_shap_bar(
        shap_values, X_transformed, feature_names, output_path=bar_plot_path
    )

    # SHAP Waterfall Plot
    waterfall_plot_path = figures_dir / "shap_waterfall.png"
    plot_shap_waterfall(
        explainer, X_transformed, index=0, output_path=waterfall_plot_path
    )

    # SHAP Dependence Plot for Amount feature
    dependence_plot_path = figures_dir / "shap_dependence_amount.png"
    plot_shap_dependence(
        "num__Amount",
        shap_values,
        X_transformed,
        feature_names,
        output_path=dependence_plot_path,
    )

    # SHAP Importance Table
    shap_importance_df = get_top_shap_importance(
        shap_values, feature_names, n=15
    )
    shap_importance_path = tables_dir / "shap_importance.csv"
    shap_importance_df.to_csv(shap_importance_path, index=False)
    print(f"Top 15 SHAP Feature Importances:\n{shap_importance_df}")

    print("\nModeling & SHAP pipeline completed successfully!")


if __name__ == "__main__":
    run_modeling_pipeline()
