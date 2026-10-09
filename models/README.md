# Model Card: Fraud Detection Models

## 📌 Model Overview
- **Production Model**: Soft Voting Ensemble Fraud Classifier
- **Model File**: `model2.pkl`
- **Architecture**: `VotingClassifier(voting='soft')` combining:
  - **XGBoost Classifier** (`XGBClassifier`)
  - **LightGBM Classifier** (`LGBMClassifier`)
  - **Random Forest Classifier** (`RandomForestClassifier`)
- **Preprocessing Pipeline**: `ColumnTransformer` with `SimpleImputer(strategy='median')` and `RobustScaler()`
- **Tuned Model Engine**: LightGBM Classifier (`LGBMClassifier`) with RandomizedSearchCV 3-Fold Stratified Cross-Validation
- **Date**: October 2026

---

## 🏆 Performance Comparison: Production vs. Best Tuned Model

### 1. Production Model (`model2.pkl` - Soft Voting Ensemble)

| Metric | Training Set | Validation Set (15%) | Holdout Test Set (15%) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 1.0000 (100.0%) | 0.9995 (99.95%) | **0.9993 (99.93%)** |
| **Precision** | 1.0000 (100.0%) | 0.9636 (96.36%) | **0.9712 (97.12%)** |
| **Recall (Sensitivity)** | 1.0000 (100.0%) | 0.9422 (94.22%) | **0.8978 (89.78%)** |
| **F1-Score** | 1.0000 | 0.9528 | **0.9330** |
| **ROC-AUC** | 1.0000 | 0.9965 | **0.9942** |

#### Confusion Matrices for Production Model (`Voting`):

##### A. Holdout Test Set (42,713 samples, 225 Frauds):
| | Predicted Legitimate (0) | Predicted Fraud (1) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Legitimate (0)** | **42,482** (TN) | **6** (FP) | 42,488 |
| **Actual Fraud (1)** | **23** (FN) | **202** (TP) | 225 |
| **Total** | 42,505 | 208 | **42,713** |

* **Fraud Catch Rate (Recall)**: `89.78%` (202 / 225)
* **Precision**: `97.12%` (202 / 208)
* **False Alarm Rate**: `0.014%` (6 / 42,488)

##### B. Validation Set (42,713 samples, 225 Frauds):
| | Predicted Legitimate (0) | Predicted Fraud (1) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Legitimate (0)** | **42,480** (TN) | **8** (FP) | 42,488 |
| **Actual Fraud (1)** | **13** (FN) | **212** (TP) | 225 |
| **Total** | 42,493 | 220 | **42,713** |

##### C. Training Set (199,327 samples, 1,050 Frauds):
| | Predicted Legitimate (0) | Predicted Fraud (1) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Legitimate (0)** | **198,277** (TN) | **0** (FP) | 198,277 |
| **Actual Fraud (1)** | **0** (FN) | **1,050** (TP) | 1,050 |
| **Total** | 198,277 | 1,050 | **199,327** |

---

### 2. Best Tuned Model (`Best_Tuned` - LightGBM)
* **Optimization Method**: `RandomizedSearchCV` (10 iterations, 3-fold Stratified Cross-Validation, ROC-AUC scoring)
* **Optimal Hyperparameters**:
  ```json
  {
      "model__num_leaves": 31,
      "model__n_estimators": 300,
      "model__learning_rate": 0.1
  }
  ```

| Metric | Training Set | Validation Set (15%) | Holdout Test Set (15%) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 1.0000 (100.0%) | 0.9996 (99.96%) | **0.9994 (99.94%)** |
| **Precision** | 1.0000 (100.0%) | 0.9643 (96.43%) | **0.9674 (96.74%)** |
| **Recall (Sensitivity)** | 1.0000 (100.0%) | 0.9600 (96.00%) | **0.9244 (92.44%)** |
| **F1-Score** | 1.0000 | 0.9621 | **0.9455** |
| **ROC-AUC** | 1.0000 | 0.9964 | **0.9937** |

---

## 📊 Complete Multi-Model Benchmark Matrices

### 1. Holdout Test Set Benchmark (42,713 Transactions, 225 Frauds)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Voting (Production)** | 0.999321 | 0.971154 | 0.897778 | 0.933025 | **0.994191** |
| **Stacking** | **0.999368** | **0.985294** | 0.893333 | 0.937063 | 0.994127 |
| **XGBoost** | 0.999227 | 0.970588 | 0.880000 | 0.923077 | 0.994026 |
| **CatBoost** | 0.999204 | 0.896266 | **0.960000** | 0.927039 | 0.993786 |
| **Best_Tuned** | **0.999438** | 0.967442 | 0.924444 | **0.945455** | 0.993724 |
| **SMOTE** | 0.999134 | 0.888430 | 0.955556 | 0.920771 | 0.993213 |
| **RandomForest** | 0.999017 | 0.974093 | 0.835556 | 0.899522 | 0.992552 |
| **LightGBM** | 0.999344 | 0.958140 | 0.915556 | 0.936364 | 0.990972 |
| **LogisticRegression** | 0.998548 | 0.940541 | 0.773333 | 0.848780 | 0.980259 |
| **DecisionTree** | 0.998502 | 0.860987 | 0.853333 | 0.857143 | 0.926302 |

---

### 2. Validation Set Benchmark (42,713 Transactions, 225 Frauds)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CatBoost** | 0.999134 | 0.882114 | 0.964444 | 0.921444 | **0.998702** |
| **SMOTE** | 0.999462 | 0.931624 | **0.968889** | 0.949891 | 0.997508 |
| **XGBoost** | 0.999321 | 0.957944 | 0.911111 | 0.933941 | 0.996870 |
| **Voting** | 0.999508 | 0.963636 | 0.942222 | 0.952809 | 0.996529 |
| **Stacking** | 0.999485 | 0.963470 | 0.937778 | 0.950450 | 0.996488 |
| **Best_Tuned** | **0.999602** | **0.964286** | 0.960000 | **0.962138** | 0.996380 |
| **LightGBM** | **0.999602** | **0.964286** | 0.960000 | **0.962138** | 0.996370 |
| **RandomForest** | 0.999227 | 0.957143 | 0.893333 | 0.924138 | 0.994691 |
| **LogisticRegression** | 0.998548 | 0.897561 | 0.817778 | 0.855814 | 0.993604 |
| **DecisionTree** | 0.998431 | 0.826446 | 0.888889 | 0.856531 | 0.943950 |

---

### 3. Training Set Benchmark (199,327 Transactions, 1,050 Frauds)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Voting** | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| **Stacking** | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| **Best_Tuned** | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| **LightGBM** | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| **RandomForest** | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| **DecisionTree** | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| **SMOTE** | 0.999995 | 0.999049 | 1.000000 | 0.999524 | 1.000000 |
| **XGBoost** | 0.999885 | 0.999028 | 0.979048 | 0.988937 | 1.000000 |
| **CatBoost** | 0.999543 | 0.920245 | 1.000000 | 0.958466 | 1.000000 |
| **LogisticRegression** | 0.998595 | 0.935520 | 0.787619 | 0.855222 | 0.991510 |

---

## 📈 Dataset & Stratification Summary
- **Source**: Credit Card Fraud Detection Dataset (283,726 unique transactions after deduplication)
- **Features**: 30 numerical features (`Time`, `V1`–`V28`, `Amount`)
- **Target**: `Class` (Binary: `0` = Legitimate, `1` = Fraud)
- **Resampling**: Minority class upsampled via SMOTE from 473 to 1,500 fraud transactions
- **Dataset Partitioning**:
  - **Train Set (70%)**: `199,327` transactions (1,050 Frauds \| 0.527%)
  - **Validation Set (15%)**: `42,713` transactions (225 Frauds \| 0.527%)
  - **Test Set (15%)**: `42,713` transactions (225 Frauds \| 0.527%)
- **Reproducibility**: Random seed fixed to `42`

---

## 🎯 Intended Use & Deployment
- **Intended Use**: Real-time and batch fraud inference with user-tunable sensitivity thresholds in the Streamlit Dashboard (`app.py`).
- **Explainability**: Interpretability generated via SHAP (`shap.TreeExplainer`) capturing feature attribution rankings (`V14`, `V4`, `V12`, `V10`, etc.).
