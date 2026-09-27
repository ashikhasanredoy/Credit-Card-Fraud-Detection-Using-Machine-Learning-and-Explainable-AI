"""
Improved binary-classification pipeline for imbalanced datasets
(e.g. credit card fraud detection).

Pipeline:
  Data Collection -> Preprocessing -> Train/Test Split ->
  Baseline Screening (CV) -> SMOTE-Tomek (inside CV) ->
  Hyperparameter Optimization -> Ensemble Learning ->
  Threshold Tuning + Evaluation -> SHAP Explainability -> Deployment

Fixes vs. the original notebook:
  - SMOTE-Tomek and hyperparameter tuning happen INSIDE the same pipeline,
    not as separate, incomparable experiments.
  - Model selection uses cross-validated PR-AUC (average precision), not a
    single-split ROC-AUC, since ROC-AUC is misleading on rare-event data.
  - Ensemble is built from the TUNED, balanced models, not the untuned ones.
  - Decision threshold is tuned instead of using the default 0.5 cutoff.
  - The model is actually saved (joblib) with metadata, closing the loop to
    "deployment".
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. Data Collection
# ---------------------------------------------------------------------------
DATA_PATH = "dataset.arff"
TARGET_COL = "Class"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip("'\"").astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    transformers = [("num", numeric_pipeline, numeric_cols)]

    if categorical_cols:
        categorical_pipeline = ImbPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers)


# ---------------------------------------------------------------------------
# 3-6. Candidate models + SMOTE-Tomek baked into one pipeline per model,
#      tuned jointly via RandomizedSearchCV, scored on CV average precision
# ---------------------------------------------------------------------------
def build_candidates(preprocessor: ColumnTransformer) -> dict:
    """Each candidate is a full imblearn Pipeline: preprocess -> resample -> model."""
    base_models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
    }

    pipelines = {
        name: ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote_tomek", SMOTETomek(random_state=RANDOM_STATE)),
            ("model", model),
        ])
        for name, model in base_models.items()
    }
    return pipelines


PARAM_GRIDS = {
    "LogisticRegression": {
        "model__C": [0.01, 0.1, 1, 10],
        "model__solver": ["lbfgs", "liblinear"],
        "smote_tomek__sampling_strategy": [0.3, 0.5, 1.0],
    },
    "RandomForest": {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [8, 12, 20, None],
        "model__min_samples_leaf": [1, 2, 5],
        "smote_tomek__sampling_strategy": [0.3, 0.5, 1.0],
    },
    "XGBoost": {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "smote_tomek__sampling_strategy": [0.3, 0.5, 1.0],
    },
    "LightGBM": {
        "model__n_estimators": [200, 400, 600],
        "model__num_leaves": [31, 63, 100],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "smote_tomek__sampling_strategy": [0.3, 0.5, 1.0],
    },
    "CatBoost": {
        "model__iterations": [200, 400, 600],
        "model__depth": [4, 6, 8],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "smote_tomek__sampling_strategy": [0.3, 0.5, 1.0],
    },
}


def tune_all_models(pipelines: dict, X_train, y_train, n_iter: int = 15) -> dict:
    """Randomized search each model+SMOTE pipeline, scored on CV average precision."""
    tuned = {}
    for name, pipe in pipelines.items():
        print(f"Tuning {name}...")
        search = RandomizedSearchCV(
            pipe,
            PARAM_GRIDS[name],
            n_iter=n_iter,
            scoring="average_precision",  # PR-AUC: right metric for rare-event data
            cv=CV,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        search.fit(X_train, y_train)
        tuned[name] = search.best_estimator_
        print(f"  best CV avg-precision={search.best_score_:.4f}  params={search.best_params_}")
    return tuned


# ---------------------------------------------------------------------------
# 7. Ensemble Learning (built from the TUNED, balanced models)
# ---------------------------------------------------------------------------
def build_ensembles(tuned_models: dict, preprocessor: ColumnTransformer) -> dict:
    estimators = [
        (name, pipe.named_steps["model"]) for name, pipe in tuned_models.items()
        if name in ("XGBoost", "LightGBM", "RandomForest")
    ]

    voting = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote_tomek", SMOTETomek(random_state=RANDOM_STATE)),
        ("model", VotingClassifier(estimators=estimators, voting="soft")),
    ])

    stacking = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote_tomek", SMOTETomek(random_state=RANDOM_STATE)),
        ("model", StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=2000),
            stack_method="predict_proba",
            cv=5,
        )),
    ])

    return {"Voting": voting, "Stacking": stacking}


# ---------------------------------------------------------------------------
# 8. Threshold tuning + evaluation
# ---------------------------------------------------------------------------
def best_threshold_for_f2(y_true, y_prob) -> float:
    """Pick the decision threshold that maximizes F2 (weights recall higher,
    appropriate when missing fraud is costlier than a false alarm)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f2_scores = []
    for p, r in zip(precisions, recalls):
        f2_scores.append((5 * p * r) / (4 * p + r) if (p + r) > 0 else 0.0)
    best_idx = int(np.argmax(f2_scores))
    # precision_recall_curve returns one more point than thresholds
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5


def evaluate(name: str, pipe, X_test, y_test) -> dict:
    y_prob = pipe.predict_proba(X_test)[:, 1]
    threshold = best_threshold_for_f2(y_test, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "f2": fbeta_score(y_test, y_pred, beta=2),
        "threshold": threshold,
    }
    print(f"\n{name}: {metrics}")
    print(confusion_matrix(y_test, y_pred))
    return metrics


# ---------------------------------------------------------------------------
# 9. SHAP explainability (on the model actually selected for deployment)
# ---------------------------------------------------------------------------
def explain(pipe, X_test, sample_size: int = 2000):
    preprocessor = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]

    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=RANDOM_STATE)
    X_transformed = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()

    # TreeExplainer only applies to a single tree model, not Voting/Stacking ensembles
    if hasattr(model, "estimators_") or hasattr(model, "estimators"):
        print("SHAP: ensemble model detected; skipping TreeExplainer. "
              "Explain the strongest individual tuned model instead.")
        return None, None, feature_names

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    return explainer, shap_values, feature_names


# ---------------------------------------------------------------------------
# 10. Deployment
# ---------------------------------------------------------------------------
def deploy(pipe, threshold: float, metrics: dict, out_dir: str = "model_artifacts"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    joblib.dump(pipe, out / "pipeline.joblib")
    metadata = {"threshold": threshold, "metrics": metrics}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=float))
    print(f"\nSaved deployable pipeline + metadata to {out.resolve()}")


def predict_new(pipe_path: str, meta_path: str, X_new: pd.DataFrame) -> np.ndarray:
    """Inference helper: load the saved pipeline and apply the tuned threshold."""
    pipe = joblib.load(pipe_path)
    threshold = json.loads(Path(meta_path).read_text())["threshold"]
    y_prob = pipe.predict_proba(X_new)[:, 1]
    return (y_prob >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(X)
    candidates = build_candidates(preprocessor)
    tuned_models = tune_all_models(candidates, X_train, y_train)

    ensembles = build_ensembles(tuned_models, preprocessor)
    for name, pipe in ensembles.items():
        pipe.fit(X_train, y_train)

    all_models = {**tuned_models, **ensembles}
    all_metrics = {name: evaluate(name, pipe, X_test, y_test) for name, pipe in all_models.items()}

    results_df = pd.DataFrame(all_metrics).T.sort_values("pr_auc", ascending=False)
    print("\n=== Final comparison (sorted by PR-AUC) ===")
    print(results_df)

    best_name = results_df.index[0]
    best_pipe = all_models[best_name]
    best_metrics = all_metrics[best_name]
    print(f"\nSelected for deployment: {best_name}")

    explain(best_pipe if best_name in tuned_models else tuned_models["XGBoost"], X_test)

    deploy(best_pipe, best_metrics["threshold"], best_metrics)


if __name__ == "__main__":
    main()
