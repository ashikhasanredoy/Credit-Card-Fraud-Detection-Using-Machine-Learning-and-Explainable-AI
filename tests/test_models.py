"""Unit tests for model pipeline initialization, training, and inference."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.build_features import build_preprocessor
from src.models.train import build_models, fit_single_model
from src.models.ensemble import (
    build_voting_classifier,
    build_stacking_classifier,
    build_smote_pipeline,
)


def test_base_models_build_and_fit() -> None:
    """Test building base classifiers and fitting single pipeline on synthetic data."""
    np.random.seed(42)
    n_samples = 60
    n_features = 5

    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"V{i}" for i in range(n_features)],
    )
    y_train = pd.Series([0] * 50 + [1] * 10)

    numeric_cols = X_train.columns.tolist()
    preprocessor = build_preprocessor(numeric_cols)

    models = build_models()
    assert "LogisticRegression" in models
    assert "RandomForest" in models
    assert "XGBoost" in models
    assert "LightGBM" in models
    assert "CatBoost" in models

    # Test single pipeline fit and predict
    pipe = fit_single_model(
        "LogisticRegression",
        models["LogisticRegression"],
        preprocessor,
        X_train,
        y_train,
    )
    preds = pipe.predict(X_train)
    probs = pipe.predict_proba(X_train)

    assert len(preds) == n_samples
    assert probs.shape == (n_samples, 2)


def test_ensemble_pipelines() -> None:
    """Test constructing voting, stacking, and SMOTE pipelines."""
    np.random.seed(42)
    n_samples = 40
    n_features = 4

    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"V{i}" for i in range(n_features)],
    )
    y_train = pd.Series([0] * 30 + [1] * 10)

    numeric_cols = X_train.columns.tolist()
    preprocessor = build_preprocessor(numeric_cols)

    base_models = build_models()
    trained_base = {}
    for name in ["XGBoost", "LightGBM", "RandomForest"]:
        pipe = fit_single_model(
            name, base_models[name], preprocessor, X_train, y_train
        )
        trained_base[name] = pipe

    # Voting Classifier Pipeline
    voting_pipe = build_voting_classifier(trained_base, preprocessor)
    voting_pipe.fit(X_train, y_train)
    v_preds = voting_pipe.predict(X_train)
    assert len(v_preds) == n_samples

    # Stacking Classifier Pipeline
    stack_pipe = build_stacking_classifier(trained_base, preprocessor)
    stack_pipe.fit(X_train, y_train)
    s_preds = stack_pipe.predict(X_train)
    assert len(s_preds) == n_samples

    # SMOTE Pipeline
    smote_pipe = build_smote_pipeline(preprocessor, base_models["LogisticRegression"])
    smote_pipe.fit(X_train, y_train)
    smote_preds = smote_pipe.predict(X_train)
    assert len(smote_preds) == n_samples


def test_train_and_evaluate_all_splits() -> None:
    """Test train_and_evaluate_all_splits returns metrics for train, val, and test splits."""
    from src.models.train import train_and_evaluate_all_splits

    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(30, 3), columns=["V1", "V2", "V3"])
    y_train = pd.Series([0] * 25 + [1] * 5)
    X_val = pd.DataFrame(np.random.randn(15, 3), columns=["V1", "V2", "V3"])
    y_val = pd.Series([0] * 12 + [1] * 3)
    X_test = pd.DataFrame(np.random.randn(15, 3), columns=["V1", "V2", "V3"])
    y_test = pd.Series([0] * 12 + [1] * 3)

    preprocessor = build_preprocessor(["V1", "V2", "V3"])
    models = {"LogisticRegression": build_models()["LogisticRegression"]}

    trained, train_s, val_s, test_s = train_and_evaluate_all_splits(
        models, preprocessor, X_train, y_train, X_val, y_val, X_test, y_test
    )

    assert "LogisticRegression" in trained
    assert "LogisticRegression" in train_s
    assert "LogisticRegression" in val_s
    assert "LogisticRegression" in test_s

    for metric_dict in [train_s["LogisticRegression"], val_s["LogisticRegression"], test_s["LogisticRegression"]]:
        assert "f1" in metric_dict
        assert "roc_auc" in metric_dict
        assert "precision" in metric_dict
        assert "recall" in metric_dict
        assert "accuracy" in metric_dict


def test_logger_setup(tmp_path) -> None:
    """Test logger writes output to specified log file."""
    from src.utils.logger import setup_logger

    log_dir = tmp_path / "logs"
    logger = setup_logger(name="test_logger", log_dir=log_dir, log_file="test.log")
    test_msg = "Fraud detection test message"
    logger.info(test_msg)

    log_file = log_dir / "test.log"
    assert log_file.exists()
    assert test_msg in log_file.read_text(encoding="utf-8")

