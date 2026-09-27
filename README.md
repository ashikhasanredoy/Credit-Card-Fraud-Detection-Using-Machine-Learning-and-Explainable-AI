# 🛡️ End-to-End Credit Card Fraud Detection Pipeline

An end-to-end, production-ready Machine Learning system engineered for **highly imbalanced binary classification** (e.g., Credit Card Fraud Detection).

This project includes both an exploratory research workflow (`models.ipynb`) and an automated, modular production pipeline (`fraud_detection_pipeline.py`) featuring leak-free resampling, hyperparameter optimization, ensemble architectures, cost-sensitive threshold tuning, SHAP model interpretability, and automated deployment artifact packaging.

---

## 📌 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Key Improvements & Design Decisions](#-key-improvements--design-decisions)
- [Project Structure](#-project-structure)
- [Machine Learning Workflow](#-machine-learning-workflow)
  - [1. Data Ingestion & Preprocessing](#1-data-ingestion--preprocessing)
  - [2. Leak-Free Resampling (SMOTE-Tomek)](#2-leak-free-resampling-smote-tomek)
  - [3. Candidate Models & Tuning](#3-candidate-models--tuning)
  - [4. Ensemble Methods](#4-ensemble-methods)
  - [5. Cost-Sensitive Threshold Tuning (F2-Score Optimization)](#5-cost-sensitive-threshold-tuning-f_2-score-optimization)
  - [6. SHAP Explainability & Feature Importance](#6-shap-explainability--feature-importance)
  - [7. Artifact Export & Inference](#7-artifact-export--inference)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
  - [Running the Production Pipeline](#running-the-production-pipeline)
  - [Running Real-Time Inference](#running-real-time-inference)
  - [Running the Jupyter Notebook](#running-the-jupyter-notebook)
- [Evaluation Metrics & Results](#-evaluation-metrics--results)
- [Dependencies](#-dependencies)
- [License](#-license)

---

## 🚀 Overview & Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────────┐
│  dataset.arff   │ ──► │  Preprocessing   │ ──► │ Stratified Train/Test  │
│ (Raw Transactions)    │ (RobustScaler +  │     │       80 / 20 Split    │
└─────────────────┘     │  One-Hot Encode) │     └───────────┬────────────┘
                        └──────────────────┘                 │
                                                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               5-Fold Stratified Cross-Validation Loop                   │
│  ┌─────────────────────────┐         ┌───────────────────────────────┐  │
│  │   SMOTE-Tomek Resample  │  ─────► │    RandomizedSearchCV         │  │
│  │   (Train folds only)    │         │ (Optimize PR-AUC / Avg. Prec) │  │
│  └─────────────────────────┘         └───────────────────────────────┘  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │            Tuned Base Models & Ensembles                │
        │   • Logistic Regression   • Random Forest               │
        │   • XGBoost               • LightGBM                    │
        │   • CatBoost              • Soft Voting & Stacking      │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│  Threshold Tuning     │ ──► │  SHAP TreeExplainer   │ ──► │   Deployment Export   │
│ (Maximize F2 for recall)    │ (Feature attribution) │     │ (pipeline.joblib +    │
└───────────────────────┘     └───────────────────────┘     │  metadata.json)       │
                                                            └───────────────────────┘
```

---

## 💡 Key Improvements & Design Decisions

| Challenge in Imbalanced Fraud Detection | Standard / Naive Approach | Our Solution in `fraud_detection_pipeline.py` |
| :--- | :--- | :--- |
| **Data Leakage in Resampling** | Applying SMOTE to the entire dataset before train/test split or CV. | Encapsulated `SMOTETomek` directly inside `imblearn.pipeline.Pipeline` ensuring synthetic samples are generated **strictly within training folds**. |
| **Misleading Evaluation Metrics** | Relying on Accuracy or single-split ROC-AUC (which masks high false-positive rates on rare events). | Optimized hyperparameters using **Cross-Validated PR-AUC (Average Precision)** and evaluated test sets on **569XScore**, Recall, and PR-AUC. |
| **Arbitrary Decision Boundaries** | Using default probability threshold of zsh.5$. | Dynamic **569XScore threshold optimization** (`beta=2`), prioritizing fraud capture (minimizing costly False Negatives) over false alarms. |
| **Suboptimal Ensembling** | Ensembling baseline untuned models. | Constructing **Soft Voting** and **Stacking Classifiers** leveraging top-performing hyperparameter-tuned estimators. |
| **Model Explainability & Production Readiness** | Black-box predictions with no deployment hooks. | Integrated **SHAP TreeExplainer** diagnostics and automated export of production-ready serialized models (`pipeline.joblib`) with config metadata. |

---

## 📂 Project Structure

```text
.
├── dataset.arff                 # Credit card transaction dataset (features + target class)
├── fraud_detection_pipeline.py  # Production Python pipeline (train, tune, evaluate, deploy)
├── models.ipynb                 # Interactive Jupyter notebook for EDA & experimentation
├── model_artifacts/             # Exported deployment directory (auto-generated)
│   ├── pipeline.joblib          # Serialized scikit-learn/imblearn full pipeline
│   └── metadata.json            # Optimal classification threshold & evaluation metrics
└── README.md                    # Project documentation
```

---

## ⚙️ Machine Learning Workflow

### 1. Data Ingestion & Preprocessing
- **Source:** Loaded from `dataset.arff` via `scipy.io.arff`.
- **Target Variable:** `Class` (`0` = Legitimate transaction, `1` = Fraudulent transaction).
- **Transformation Pipeline:**
  - **Numeric Features:** Handled with `SimpleImputer(strategy="median")` and scaled using `RobustScaler` (handles extreme transaction outliers).
  - **Categorical Features:** Imputed using `most_frequent` and transformed using `OneHotEncoder(handle_unknown="ignore", drop="first")`.

### 2. Leak-Free Resampling (SMOTE-Tomek)
- Uses **SMOTE-Tomek** (`imblearn.combine.SMOTETomek`) to combine:
  - **SMOTE (Synthetic Minority Over-sampling Technique):** Synthesizes minority class points.
  - **Tomek Links:** Cleans boundary ambiguities by removing overlapping pairs of instances.
- **Sampling Strategy Tuning:** `sampling_strategy` ratio (zsh.3, 0.5, 1.0$) is tuned directly inside CV.

### 3. Candidate Models & Tuning
The pipeline tunes hyperparameters for 5 base algorithms using `RandomizedSearchCV` with 5-fold `StratifiedKFold`:
1. **Logistic Regression** (L2-penalized baseline)
2. **Random Forest Classifier** (Bagging tree ensemble)
3. **XGBoost Classifier** (Gradient boosting)
4. **LightGBM Classifier** (Histogram-based gradient boosting)
5. **CatBoost Classifier** (Symmetric tree boosting)

### 4. Ensemble Methods
Constructs meta-models from the tuned tree-based models (`XGBoost`, `LightGBM`, `RandomForest`):
- **Soft Voting Classifier:** Weighted average of predicted class probabilities.
- **Stacking Classifier:** Meta-learner (`LogisticRegression`) trained on cross-validated probability predictions of the base estimators.

### 5. Cost-Sensitive Threshold Tuning (569XScore Optimization)
In fraud detection, **False Negatives (missed fraud) are significantly more costly than False Positives (manual verification)**.
- Precision-Recall curve is computed for each model.
- Threshold $\tau \in [0, 1]$ is selected to maximize the **569XScore**:
  7015\text{F}_2 = \frac{5 \cdot \text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}7015

### 6. SHAP Explainability & Feature Importance
- Employs `shap.TreeExplainer` on the top tree-based model to calculate SHAP values.
- Generates **Summary Plots**, **Bar Feature Importance**, **Waterfall Plots**, and **Dependence Plots** to ensure compliance, transparency, and auditability.

### 7. Artifact Export & Inference
- Automatically serializes the best performing pipeline into `model_artifacts/pipeline.joblib`.
- Records performance metrics and the optimal decision threshold in `model_artifacts/metadata.json`.

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.9+ (Recommended: Python 3.10 or 3.11)
- Virtual environment tool (`venv` or `conda`)

### Environment Setup
```bash
# Clone the repository / navigate to project directory
git clone <repository-url>
cd <project-directory>

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install numpy pandas scipy scikit-learn imbalanced-learn xgboost lightgbm catboost shap joblib matplotlib
```

---

## 💻 Usage Guide

### Running the Production Pipeline
To execute the complete end-to-end training, hyperparameter search, threshold calibration, and artifact export:

```bash
python fraud_detection_pipeline.py
```

**Expected Output:**
- Progress logs for candidate model tuning via randomized search.
- Summary comparison table ranked by **PR-AUC**.
- Confusion matrix and metrics calculated at the optimal $ threshold.
- Serialized artifacts saved in `./model_artifacts/`.

### Running Real-Time Inference
To make predictions on new incoming transaction data using the saved artifact:

```python
import pandas as pd
from fraud_detection_pipeline import predict_new

# Load new incoming batch of transaction records
df_new = pd.read_csv("new_transactions.csv")

# Predict fraud flags (0 or 1) using tuned pipeline and optimal threshold
predictions = predict_new(
    pipe_path="model_artifacts/pipeline.joblib",
    meta_path="model_artifacts/metadata.json",
    X_new=df_new
)

df_new["is_fraud_predicted"] = predictions
print(df_new[["is_fraud_predicted"]].value_counts())
```

### Running the Jupyter Notebook
For exploratory analysis, visualization, and step-by-step SHAP explainability charts:

```bash
jupyter notebook models.ipynb
```

---

## 📊 Evaluation Metrics & Results

All models are evaluated on test data using comprehensive metrics suited for extreme class imbalance:

| Metric | Purpose / Significance |
| :--- | :--- |
| **PR-AUC (Avg. Precision)** | Primary ranking metric; robust against class skew. |
| **569XScore** | Weighted F-measure placing \times$ emphasis on Recall over Precision. |
| **Recall (Sensitivity)** | Proportion of actual fraudulent transactions detected. |
| **Precision** | Proportion of flagged transactions that were truly fraudulent. |
| **ROC-AUC** | Global discriminative capability across all thresholds. |

---

## 🛠️ Dependencies

- **[scikit-learn](https://scikit-learn.org/)**: Preprocessing, classification algorithms, metric evaluation, model stacking/voting.
- **[imbalanced-learn](https://imbalanced-learn.org/)**: `SMOTETomek` resampling and imbalanced pipelines.
- **[XGBoost](https://xgboost.readthedocs.io/)**, **[LightGBM](https://lightgbm.readthedocs.io/)**, **[CatBoost](https://catboost.ai/)**: High-performance gradient boosting frameworks.
- **[SHAP](https://shap.readthedocs.io/)**: Model interpretability via Shapley additive explanations.
- **[Joblib](https://joblib.readthedocs.io/)**: Model serialization and persistence.
- **[SciPy](https://scipy.org/)**: ARFF file parsing (`scipy.io.arff`).
- **[Pandas](https://pandas.pydata.org/)** & **[NumPy](https://numpy.org/)**: High-performance data manipulation and numerical computation.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
