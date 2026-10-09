"""End-to-end model training, tuning, ensembling, evaluation, and explainability pipeline."""

import json
from pathlib import Path
from typing import Optional
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from src.config import (
    DATA_PROCESSED_PATH,
    FIGURES_DIR,
    LOGS_DIR,
    MODEL2_PATH,
    MODEL_DIR,
    PARAM_GRID,
    PREDICTIONS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    RESULTS_LOGS_DIR,
    RESULTS_SHAP_DIR,
    RESULTS_TEST_DIR,
    RESULTS_TRAIN_DIR,
    RESULTS_VAL_DIR,
    TABLES_DIR,
    TARGET_COLUMN,
    TARGET_FRAUD_COUNT,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
    set_seed,
)
from src.data.load_data import load_processed_csv
from src.data.preprocess import (
    get_numeric_categorical_columns,
    split_features_target,
    stratified_train_val_test_split,
    upsample_minority_class,
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
from src.models.ensemble import fit_and_evaluate_all_ensembles_splits
from src.models.evaluate import evaluate_pipeline, print_confusion_matrix
from src.models.train import build_models, train_and_evaluate_all_splits
from src.models.tune import random_search_tune
from src.utils.logger import setup_logger


def run_modeling_pipeline(
    processed_data_path: Path = DATA_PROCESSED_PATH,
    model_save_path: Path = MODEL2_PATH,
    figures_dir: Path = FIGURES_DIR,
    tables_dir: Path = TABLES_DIR,
    predictions_dir: Path = PREDICTIONS_DIR,
    results_dir: Path = RESULTS_DIR,
    train_results_dir: Path = RESULTS_TRAIN_DIR,
    val_results_dir: Path = RESULTS_VAL_DIR,
    test_results_dir: Path = RESULTS_TEST_DIR,
    shap_results_dir: Path = RESULTS_SHAP_DIR,
    logs_dir: Path = LOGS_DIR,
    results_logs_dir: Path = RESULTS_LOGS_DIR,
    target_fraud_count: int = TARGET_FRAUD_COUNT,
    train_size: float = TRAIN_SIZE,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> None:
    """Execute complete modeling, tuning, evaluation, logging, and serialization pipeline.

    Args:
        processed_data_path: Path to cleaned CSV dataset.
        model_save_path: Path to save primary serialized model (e.g., model2.pkl).
        figures_dir: Directory to save evaluation and explainability plots.
        tables_dir: Directory to save metric comparison tables.
        predictions_dir: Directory to save test/val set predictions.
        results_dir: Root results directory.
        train_results_dir: Dedicated directory for train split results.
        val_results_dir: Dedicated directory for validation split results.
        test_results_dir: Dedicated directory for test split results.
        shap_results_dir: Dedicated directory for SHAP explainability results.
        logs_dir: Dedicated project logs directory.
        results_logs_dir: Dedicated results logs directory.
        target_fraud_count: Target number of fraud instances after SMOTE upsampling.
        train_size: Proportion of dataset for training (0.70).
        val_size: Proportion of dataset for validation (0.15).
        test_size: Proportion of dataset for testing (0.15).
        random_state: Random seed for reproducibility.
    """
    # Environment setup and directory structure
    set_seed(random_state)

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    train_results_dir.mkdir(parents=True, exist_ok=True)
    val_results_dir.mkdir(parents=True, exist_ok=True)
    test_results_dir.mkdir(parents=True, exist_ok=True)
    shap_results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_logs_dir.mkdir(parents=True, exist_ok=True)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="fraud_detection",
        log_dir=logs_dir,
        log_file="fraud_detection.log",
        additional_log_dirs=[results_logs_dir],
    )
    logger.info("=" * 70)
    logger.info("Starting Fraud Detection Modeling & Evaluation Pipeline")
    logger.info("=" * 70)
    logger.info(f"Target Fraud Count (Upsampling): {target_fraud_count:,}")
    logger.info(f"Configuration -> Train: {train_size*100:.0f}%, Val: {val_size*100:.0f}%, Test: {test_size*100:.0f}%")
    logger.info(f"Random State: {random_state}")
    logger.info(f"Results Directory: {results_dir.resolve()}")
    logger.info(f"Train Results Directory: {train_results_dir.resolve()}")
    logger.info(f"Validation Results Directory: {val_results_dir.resolve()}")
    logger.info(f"Test Results Directory: {test_results_dir.resolve()}")
    logger.info(f"SHAP Results Directory: {shap_results_dir.resolve()}")
    logger.info(f"Logs Directory: {logs_dir.resolve()} & {results_logs_dir.resolve()}")
    logger.info(f"Target Model Save Path: {model_save_path.resolve()}")

    # Ingest preprocessed dataset
    logger.info(f"Loading processed dataset from {processed_data_path}...")
    print(f"Loading processed dataset from {processed_data_path}...")
    df = load_processed_csv(processed_data_path)
    logger.info(f"Dataset successfully loaded. Total rows: {len(df):,}, Total columns: {len(df.columns)}")

    X, y = split_features_target(df, target_col=TARGET_COLUMN)
    numeric_cols, _ = get_numeric_categorical_columns(X)
    original_fraud_count = int((y == 1).sum())
    logger.info(f"Original dataset: Legitimate={int((y==0).sum()):,}, Fraud={original_fraud_count:,}")

    # Resample minority class to target count
    if target_fraud_count > original_fraud_count:
        logger.info(f"Upsampling fraud data from {original_fraud_count:,} to {target_fraud_count:,} samples using SMOTE...")
        print(f"\nUpsampling fraud samples from {original_fraud_count:,} to {target_fraud_count:,}...")
        X, y = upsample_minority_class(
            X, y, target_minority_count=target_fraud_count, random_state=random_state
        )
        logger.info(f"Upsampling complete. New distribution: Legitimate={int((y==0).sum()):,}, Fraud={int((y==1).sum()):,}")

    # Feature transformer
    preprocessor = build_preprocessor(numeric_cols)
    logger.info("Constructed ColumnTransformer preprocessor (RobustScaler).")

    # Stratified multi-split partitioning
    logger.info(
        f"Executing Stratified Split (Train={train_size*100:.0f}%, Val={val_size*100:.0f}%, "
        f"Test={test_size*100:.0f}%, random_state={random_state})..."
    )
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
        X,
        y,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    n_fraud_train = int(y_train.sum())
    n_fraud_val = int(y_val.sum())
    n_fraud_test = int(y_test.sum())

    logger.info(
        f"Train set: {len(X_train):,} samples (Fraud: {n_fraud_train:,}, {n_fraud_train/len(X_train)*100:.3f}% | {n_fraud_train/target_fraud_count*100:.1f}% of total fraud) | "
        f"Val set: {len(X_val):,} samples (Fraud: {n_fraud_val:,}, {n_fraud_val/len(X_val)*100:.3f}% | {n_fraud_val/target_fraud_count*100:.1f}% of total fraud) | "
        f"Test set: {len(X_test):,} samples (Fraud: {n_fraud_test:,}, {n_fraud_test/len(X_test)*100:.3f}% | {n_fraud_test/target_fraud_count*100:.1f}% of total fraud)"
    )
    print(
        f"\nDataset Splits (Exact Fraud Stratification across {target_fraud_count:,} total frauds):\n"
        f"  - Train : {X_train.shape[0]:,} rows (Fraud: {n_fraud_train:,} -> {n_fraud_train/target_fraud_count*100:.0f}% of fraud)\n"
        f"  - Val   : {X_val.shape[0]:,} rows (Fraud: {n_fraud_val:,} -> {n_fraud_val/target_fraud_count*100:.0f}% of fraud)\n"
        f"  - Test  : {X_test.shape[0]:,} rows (Fraud: {n_fraud_test:,} -> {n_fraud_test/target_fraud_count*100:.0f}% of fraud)"
    )

    # Train baseline classifiers
    logger.info("\n--- Training Base Models ---")
    print("\n--- Training Base Models ---")
    base_models = build_models()
    trained_base_models, base_train_scores, base_val_scores, base_test_scores = (
        train_and_evaluate_all_splits(
            models=base_models,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )
    )

    # Select baseline candidate via validation F1 score
    base_val_df = pd.DataFrame(base_val_scores).T
    best_base_model_name = str(base_val_df["f1"].idxmax())
    logger.info(f"Top Base Model by Validation F1: {best_base_model_name} (F1={base_val_df.loc[best_base_model_name, 'f1']:.4f})")
    print(f"\nTop Base Model selected by Validation F1 score: {best_base_model_name}")

    # Hyperparameter optimization via cross-validation
    logger.info(f"\n--- Hyperparameter Optimization for {best_base_model_name} ---")
    print(f"\nTuning hyperparameters for {best_base_model_name}...")
    tune_pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", base_models[best_base_model_name])]
    )
    best_tuned_model, best_params = random_search_tune(
        pipeline=tune_pipeline,
        param_grid=PARAM_GRID[best_base_model_name],
        X_train=X_train,
        y_train=y_train,
        n_iter=10,
        cv=3,
        scoring="roc_auc",
        random_state=random_state,
    )

    # Save tuning params into results
    tuning_record = {
        "best_base_model": best_base_model_name,
        "best_hyperparameters": {k: (str(v) if isinstance(v, (type, object)) else v) for k, v in best_params.items()},
    }
    with open(results_dir / "hyperparameter_tuning.json", "w", encoding="utf-8") as f:
        json.dump(tuning_record, f, indent=4)

    # Train and evaluate ensemble classifiers
    logger.info("\n--- Training & Evaluating Ensemble Models ---")
    print("\n--- Training Ensemble Models ---")
    ensemble_models, ens_train_scores, ens_val_scores, ens_test_scores = (
        fit_and_evaluate_all_ensembles_splits(
            trained_models=trained_base_models,
            preprocessor=preprocessor,
            best_model_name=best_base_model_name,
            base_models=base_models,
            best_tuned_model=best_tuned_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            random_state=random_state,
        )
    )

    # Aggregate evaluation metrics
    all_train_scores = {**base_train_scores, **ens_train_scores}
    all_val_scores = {**base_val_scores, **ens_val_scores}
    all_test_scores = {**base_test_scores, **ens_test_scores}

    train_scores_df = pd.DataFrame(all_train_scores).T
    val_scores_df = pd.DataFrame(all_val_scores).T
    test_scores_df = pd.DataFrame(all_test_scores).T

    # Save dedicated TRAIN results
    train_scores_df.to_csv(train_results_dir / "train_scores.csv")
    with open(train_results_dir / "train_scores.json", "w", encoding="utf-8") as f:
        json.dump(all_train_scores, f, indent=4)

    # Save dedicated VALIDATION results
    val_scores_df.to_csv(val_results_dir / "validation_scores.csv")
    val_scores_df.to_csv(val_results_dir / "val_scores.csv")
    with open(val_results_dir / "validation_scores.json", "w", encoding="utf-8") as f:
        json.dump(all_val_scores, f, indent=4)
    with open(val_results_dir / "val_scores.json", "w", encoding="utf-8") as f:
        json.dump(all_val_scores, f, indent=4)

    # Save dedicated TEST results
    test_scores_df.to_csv(test_results_dir / "test_scores.csv")
    with open(test_results_dir / "test_scores.json", "w", encoding="utf-8") as f:
        json.dump(all_test_scores, f, indent=4)

    # Save in root results for compatibility
    train_scores_df.to_csv(results_dir / "train_scores.csv")
    val_scores_df.to_csv(results_dir / "val_scores.csv")
    test_scores_df.to_csv(results_dir / "test_scores.csv")
    with open(results_dir / "train_scores.json", "w", encoding="utf-8") as f:
        json.dump(all_train_scores, f, indent=4)
    with open(results_dir / "val_scores.json", "w", encoding="utf-8") as f:
        json.dump(all_val_scores, f, indent=4)
    with open(results_dir / "test_scores.json", "w", encoding="utf-8") as f:
        json.dump(all_test_scores, f, indent=4)

    # Save backward-compatible tables
    val_scores_df.to_csv(tables_dir / "model_comparison.csv")
    test_scores_df.to_csv(tables_dir / "final_results.csv")

    # Build Comprehensive Multi-Split Comparison Summary
    summary_rows = []
    for model_name in all_val_scores.keys():
        row = {"Model": model_name}
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            row[f"Train_{metric.upper()}"] = round(all_train_scores[model_name].get(metric, 0.0), 4)
            row[f"Val_{metric.upper()}"] = round(all_val_scores[model_name].get(metric, 0.0), 4)
            row[f"Test_{metric.upper()}"] = round(all_test_scores[model_name].get(metric, 0.0), 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("Val_ROC_AUC", ascending=False)
    summary_df.to_csv(results_dir / "all_scores_summary.csv", index=False)
    with open(results_dir / "all_scores_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=4)

    logger.info("Saved all evaluation scores across train, val, and test to respective folders.")
    print(f"\nAll Scores Summary (Saved to {results_dir / 'all_scores_summary.csv'}):\n{summary_df[['Model', 'Val_F1', 'Val_ROC_AUC', 'Test_F1', 'Test_ROC_AUC']]}")

    # Production model selection
    all_models_dict = {**trained_base_models, **ensemble_models}
    
    # Priority: Voting ensemble or model with highest validation ROC-AUC
    if "Voting" in ensemble_models and val_scores_df.loc["Voting", "roc_auc"] >= 0.90:
        best_model_key = "Voting"
    else:
        best_model_key = str(val_scores_df["roc_auc"].idxmax())

    best_final_model = all_models_dict[best_model_key]
    logger.info(f"Selected Best Production Model: {best_model_key}")
    print(f"\nBest Production Model: {best_model_key}")

    # Model evaluation diagnostics and predictions
    train_pred = best_final_model.predict(X_train)
    val_pred = best_final_model.predict(X_val)
    test_pred = best_final_model.predict(X_test)
    train_prob = best_final_model.predict_proba(X_train)[:, 1] if hasattr(best_final_model, "predict_proba") else train_pred
    val_prob = best_final_model.predict_proba(X_val)[:, 1] if hasattr(best_final_model, "predict_proba") else val_pred
    test_prob = best_final_model.predict_proba(X_test)[:, 1] if hasattr(best_final_model, "predict_proba") else test_pred

    # Save Predictions per folder
    train_pred_df = pd.DataFrame({"y_true": y_train, "y_pred": train_pred, "y_prob": train_prob})
    train_pred_df.to_csv(train_results_dir / "train_predictions.csv", index=False)

    val_pred_df = pd.DataFrame({"y_true": y_val, "y_pred": val_pred, "y_prob": val_prob})
    val_pred_df.to_csv(val_results_dir / "validation_predictions.csv", index=False)
    val_pred_df.to_csv(val_results_dir / "val_predictions.csv", index=False)

    test_pred_df = pd.DataFrame({"y_true": y_test, "y_pred": test_pred, "y_prob": test_prob})
    test_pred_df.to_csv(test_results_dir / "test_predictions.csv", index=False)
    test_pred_df.to_csv(test_results_dir / "y_test_pred.csv", index=False)
    test_pred_df.to_csv(predictions_dir / "y_test_pred.csv", index=False)
    test_pred_df.to_csv(results_dir / "y_test_pred.csv", index=False)

    # Confusion Matrices
    cm_train = confusion_matrix(y_train, train_pred).tolist()
    cm_val = confusion_matrix(y_val, val_pred).tolist()
    cm_test = confusion_matrix(y_test, test_pred).tolist()

    with open(train_results_dir / "train_confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump({"model": best_model_key, "confusion_matrix": cm_train}, f, indent=4)
    with open(val_results_dir / "val_confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump({"model": best_model_key, "confusion_matrix": cm_val}, f, indent=4)
    with open(test_results_dir / "test_confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump({"model": best_model_key, "confusion_matrix": cm_test}, f, indent=4)

    confusion_matrices = {
        "best_model": best_model_key,
        "train_confusion_matrix": cm_train,
        "val_confusion_matrix": cm_val,
        "test_confusion_matrix": cm_test,
    }
    with open(results_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(confusion_matrices, f, indent=4)

    # Classification Reports
    rep_train = classification_report(y_train, train_pred, digits=4)
    rep_val = classification_report(y_val, val_pred, digits=4)
    rep_test = classification_report(y_test, test_pred, digits=4)

    with open(train_results_dir / "train_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"=== MODEL: {best_model_key} (TRAIN SET) ===\n\n{rep_train}\n")
    with open(val_results_dir / "validation_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"=== MODEL: {best_model_key} (VALIDATION SET) ===\n\n{rep_val}\n")
    with open(test_results_dir / "test_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"=== MODEL: {best_model_key} (TEST SET) ===\n\n{rep_test}\n")

    reports_content = (
        f"=== BEST MODEL: {best_model_key} ===\n\n"
        f"--- TRAINING SET CLASSIFICATION REPORT ---\n{rep_train}\n\n"
        f"--- VALIDATION SET CLASSIFICATION REPORT ---\n{rep_val}\n\n"
        f"--- TEST SET CLASSIFICATION REPORT ---\n{rep_test}\n"
    )
    with open(results_dir / "classification_reports.txt", "w", encoding="utf-8") as f:
        f.write(reports_content)

    best_metrics_summary = {
        "best_model_name": best_model_key,
        "train_metrics": all_train_scores[best_model_key],
        "validation_metrics": all_val_scores[best_model_key],
        "test_metrics": all_test_scores[best_model_key],
    }
    with open(results_dir / "best_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_metrics_summary, f, indent=4)

    logger.info(f"Validation Confusion Matrix for {best_model_key}:\n{cm_val}")
    logger.info(f"Test Confusion Matrix for {best_model_key}:\n{cm_test}")
    print("\nConfusion Matrix for Best Model (Test Set):")
    print_confusion_matrix(y_test, test_pred)

    # Model serialization
    joblib.dump(best_final_model, model_save_path)
    logger.info(f"Successfully serialized model to: {model_save_path}")
    print(f"\nProduction model saved to: {model_save_path}")

    # Feature attribution analysis with SHAP
    logger.info("\n--- Running SHAP Explainability Analysis ---")
    print("\n--- Running SHAP Explainability Analysis ---")
    shap_model_pipeline = trained_base_models["LightGBM"]
    X_transformed = shap_model_pipeline.named_steps["preprocessor"].transform(
        X_test
    )
    feature_names = (
        shap_model_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    )
    lgb_model = shap_model_pipeline.named_steps["model"]

    explainer = build_explainer(lgb_model)
    shap_values = compute_shap_values(explainer, X_transformed)

    # SHAP Summary Plot (save in figures_dir and shap_results_dir)
    plot_shap_summary(
        shap_values, X_transformed, feature_names, output_path=figures_dir / "shap_summary.png"
    )
    plot_shap_summary(
        shap_values, X_transformed, feature_names, output_path=shap_results_dir / "shap_summary.png"
    )

    # SHAP Bar Plot
    plot_shap_bar(
        shap_values, X_transformed, feature_names, output_path=figures_dir / "shap_bar.png"
    )
    plot_shap_bar(
        shap_values, X_transformed, feature_names, output_path=shap_results_dir / "shap_bar.png"
    )

    # SHAP Waterfall Plot
    plot_shap_waterfall(
        explainer, X_transformed, index=0, output_path=figures_dir / "shap_waterfall.png"
    )
    plot_shap_waterfall(
        explainer, X_transformed, index=0, output_path=shap_results_dir / "shap_waterfall.png"
    )

    # SHAP Dependence Plot for Amount feature
    plot_shap_dependence(
        "num__Amount",
        shap_values,
        X_transformed,
        feature_names,
        output_path=figures_dir / "shap_dependence_amount.png",
    )
    plot_shap_dependence(
        "num__Amount",
        shap_values,
        X_transformed,
        feature_names,
        output_path=shap_results_dir / "shap_dependence_amount.png",
    )

    # SHAP Importance Table
    shap_importance_df = get_top_shap_importance(
        shap_values, feature_names, n=15
    )
    shap_importance_df.to_csv(tables_dir / "shap_importance.csv", index=False)
    shap_importance_df.to_csv(results_dir / "shap_importance.csv", index=False)
    shap_importance_df.to_csv(shap_results_dir / "shap_importance.csv", index=False)
    with open(shap_results_dir / "shap_importance.json", "w", encoding="utf-8") as f:
        json.dump(shap_importance_df.to_dict(orient="records"), f, indent=4)

    logger.info(f"Top 15 SHAP Feature Importances:\n{shap_importance_df}")
    print(f"Top 15 SHAP Feature Importances:\n{shap_importance_df}")

    logger.info("=" * 70)
    logger.info("Modeling, Evaluation & SHAP pipeline completed successfully!")
    logger.info("=" * 70)
    print("\nModeling & SHAP pipeline completed successfully!")


if __name__ == "__main__":
    run_modeling_pipeline()
