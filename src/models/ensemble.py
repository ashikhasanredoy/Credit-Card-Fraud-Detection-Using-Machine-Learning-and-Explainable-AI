"""Ensemble models construction and imbalanced learning pipelines."""

from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

from src.models.evaluate import evaluate_pipeline


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

    Returns:
        Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
            Dictionary of ensemble models and dictionary of their evaluation metrics.
    """
    # 1. Voting Pipeline
    print("Fitting Voting Classifier...")
    voting_pipe = build_voting_classifier(trained_models, preprocessor)
    voting_pipe.fit(X_train, y_train)

    # 2. Stacking Pipeline
    print("Fitting Stacking Classifier...")
    stack_pipe = build_stacking_classifier(trained_models, preprocessor)
    stack_pipe.fit(X_train, y_train)

    # 3. SMOTE Pipeline
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

    final_results = {
        "Best_Tuned": evaluate_pipeline(best_tuned_model, X_test, y_test),
        "Voting": evaluate_pipeline(voting_pipe, X_test, y_test),
        "Stacking": evaluate_pipeline(stack_pipe, X_test, y_test),
        "SMOTE": evaluate_pipeline(smote_pipe, X_test, y_test),
    }

    return ensemble_models, final_results
