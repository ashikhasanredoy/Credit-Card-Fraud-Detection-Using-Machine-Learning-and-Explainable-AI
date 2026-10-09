from typing import Any, Dict, Optional, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

from src.models.evaluate import evaluate_pipeline
from src.utils.logger import get_logger

logger = get_logger("fraud_detection.ensemble")


def build_voting_classifier(
    trained_models: Dict[str, Pipeline],
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """Build soft voting ensemble pipeline using XGBoost, LightGBM, and Random Forest.

    Args:
        trained_models: Dictionary of trained base model pipelines.
        preprocessor: ColumnTransformer preprocessor.

    Returns:
        Pipeline: VotingClassifier pipeline.
    """
    voting = VotingClassifier(
        estimators=[
            ("xgb", trained_models["XGBoost"].named_steps["model"]),
            ("lgb", trained_models["LightGBM"].named_steps["model"]),
            ("rf", trained_models["RandomForest"].named_steps["model"]),
        ],
        voting="soft",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", voting)])


def build_stacking_classifier(
    trained_models: Dict[str, Pipeline],
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """Build StackingClassifier pipeline with LogisticRegression meta-estimator.

    Args:
        trained_models: Dictionary of trained base model pipelines.
        preprocessor: ColumnTransformer preprocessor.

    Returns:
        Pipeline: StackingClassifier pipeline.
    """
    stacking = StackingClassifier(
        estimators=[
            ("xgb", trained_models["XGBoost"].named_steps["model"]),
            ("rf", trained_models["RandomForest"].named_steps["model"]),
            ("lgb", trained_models["LightGBM"].named_steps["model"]),
        ],
        final_estimator=LogisticRegression(),
        stack_method="predict_proba",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", stacking)])


def build_smote_pipeline(
    preprocessor: ColumnTransformer,
    model: Any,
    random_state: int = 42,
) -> ImbPipeline:
    """Build imbalanced-learn pipeline integrating SMOTETomek oversampling and cleaning.

    Args:
        preprocessor: ColumnTransformer preprocessor.
        model: Base classifier instance.
        random_state: Random state seed.

    Returns:
        ImbPipeline: Pipeline with preprocessor, SMOTETomek, and classifier.
    """
    return ImbPipeline(
        [
            ("preprocessor", preprocessor),
            ("smote", SMOTETomek(random_state=random_state)),
            ("model", model),
        ]
    )


def fit_all_ensembles(
    trained_models: Dict[str, Pipeline],
    preprocessor: ColumnTransformer,
    best_model_name: str,
    base_models: Dict[str, Any],
    best_tuned_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Train and evaluate Voting, Stacking, SMOTETomek, and Tuned model pipelines.

    Args:
        trained_models: Dictionary of trained base pipelines.
        preprocessor: Fitted or un-fitted ColumnTransformer.
        best_model_name: Name of top performing base model.
        base_models: Dictionary of base model instances.
        best_tuned_model: Fitted best tuned pipeline.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        random_state: Random state seed.
        X_val: Optional validation features.
        y_val: Optional validation labels.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
            Dictionary of ensemble models and dictionary of their evaluation metrics.
    """
    logger.info("Fitting Voting Classifier...")
    print("Fitting Voting Classifier...")
    voting_pipe = build_voting_classifier(trained_models, preprocessor)
    voting_pipe.fit(X_train, y_train)

    logger.info("Fitting Stacking Classifier...")
    print("Fitting Stacking Classifier...")
    stack_pipe = build_stacking_classifier(trained_models, preprocessor)
    stack_pipe.fit(X_train, y_train)

    logger.info("Fitting SMOTE-Tomek Pipeline...")
    print("Fitting SMOTE-Tomek Pipeline...")
    smote_pipe = build_smote_pipeline(
        preprocessor, base_models[best_model_name], random_state=random_state
    )
    smote_pipe.fit(X_train, y_train)

    ensemble_models = {
        "Best_Tuned": best_tuned_model,
        "Voting": voting_pipe,
        "Stacking": stack_pipe,
        "SMOTE": smote_pipe,
    }

    final_results = {}
    for name, mdl in ensemble_models.items():
        test_m = evaluate_pipeline(mdl, X_test, y_test)
        final_results[name] = test_m
        if X_val is not None and y_val is not None:
            val_m = evaluate_pipeline(mdl, X_val, y_val)
            train_m = evaluate_pipeline(mdl, X_train, y_train)
            logger.info(
                f"[Ensemble: {name}] Train F1: {train_m['f1']:.4f} | "
                f"Val F1: {val_m['f1']:.4f}, Val ROC-AUC: {val_m['roc_auc']:.4f} | "
                f"Test F1: {test_m['f1']:.4f}, Test ROC-AUC: {test_m['roc_auc']:.4f}"
            )
        else:
            logger.info(f"[Ensemble: {name}] Test metrics: {test_m}")

    return ensemble_models, final_results


def fit_and_evaluate_all_ensembles_splits(
    trained_models: Dict[str, Pipeline],
    preprocessor: ColumnTransformer,
    best_model_name: str,
    base_models: Dict[str, Any],
    best_tuned_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
]:
    """Train ensemble models and return score dictionaries for train, val, and test splits.

    Args:
        trained_models: Dictionary of trained base pipelines.
        preprocessor: ColumnTransformer preprocessor.
        best_model_name: Name of top performing base model.
        base_models: Dictionary of base model instances.
        best_tuned_model: Fitted best tuned pipeline.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Test features.
        y_test: Test labels.
        random_state: Random state seed.

    Returns:
        Tuple of (ensemble_models, train_scores, val_scores, test_scores).
    """
    ensemble_models, _ = fit_all_ensembles(
        trained_models=trained_models,
        preprocessor=preprocessor,
        best_model_name=best_model_name,
        base_models=base_models,
        best_tuned_model=best_tuned_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=random_state,
        X_val=X_val,
        y_val=y_val,
    )

    train_scores: Dict[str, Dict[str, float]] = {}
    val_scores: Dict[str, Dict[str, float]] = {}
    test_scores: Dict[str, Dict[str, float]] = {}

    for name, model in ensemble_models.items():
        train_scores[name] = evaluate_pipeline(model, X_train, y_train)
        val_scores[name] = evaluate_pipeline(model, X_val, y_val)
        test_scores[name] = evaluate_pipeline(model, X_test, y_test)

    return ensemble_models, train_scores, val_scores, test_scores
