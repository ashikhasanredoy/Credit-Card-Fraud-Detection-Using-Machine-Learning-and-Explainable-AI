# Credit Card Fraud Detection Using Machine Learning and Explainable AI

## Project Overview

This project develops a machine learning system for detecting fraudulent credit card transactions. Since fraud detection is a highly imbalanced classification problem, the project focuses not only on overall accuracy but also on **precision, recall, F1-score, ROC-AUC, and confusion matrix analysis**.

Multiple machine learning algorithms are trained and compared, followed by hyperparameter optimization, ensemble learning, class-imbalance handling using **SMOTE-Tomek**, and model explainability using **SHAP**.

---

# Machine Learning Pipeline

```text
Dataset
   ↓
Load ARFF Dataset
   ↓
Convert to Pandas DataFrame
   ↓
Feature Identification
   ↓
Data Preprocessing (Median Imputation + Robust Scaling)
   ↓
Train-Test Split
   ↓
Baseline Model Training
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Hyperparameter Optimization
   ↓
Tuned Model
   ↓
Ensemble Learning
   ├── Soft Voting
   └── Stacking
   ↓
SMOTE-Tomek
   ↓
Final Model Comparison
   ↓
Final Model Selection
   ↓
Confusion Matrix
   ↓
SHAP Explainability
```

---

# 1. Dataset Loading

The dataset is loaded from an ARFF file using SciPy.

```python
from scipy.io import arff

data, meta = arff.loadarff("dataset.arff")
df = pd.DataFrame(data)
```

### What is ARFF?

**ARFF (Attribute-Relation File Format)** is a dataset format commonly used with the WEKA machine learning platform.

The `loadarff()` function loads both:

* Dataset values
* Dataset metadata

The loaded data is then converted into a Pandas DataFrame for easier preprocessing and analysis.

---

# 2. Data Preprocessing

The dataset consists entirely of numerical features (`Time`, `V1`–`V28`, and `Amount`). The numerical preprocessing pipeline handles missing values and scaling:

```text
Numerical Features (Time, V1–V28, Amount)
                  ↓
          Median Imputation
                  ↓
            Robust Scaling
```

```python
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])
```

---

## 2.1 Median Imputation

```python
SimpleImputer(strategy="median")
```

Missing numerical values are replaced with the **median** of the corresponding feature.

Median imputation is useful because it is less affected by extreme values than mean imputation.

For example:

```text
Values:
10, 12, 15, 20, 500

Median = 15
```

If a value is missing, it can be replaced with `15`.

---

## 2.2 Robust Scaling

```python
RobustScaler()
```

RobustScaler scales numerical features using statistics based on the **median and interquartile range (IQR)**.

It is particularly useful when the dataset contains outliers (such as extreme transaction amounts).

Unlike standard scaling, RobustScaler is less sensitive to extreme observations.

---

# 3. ColumnTransformer

The preprocessing pipeline is wrapped using `ColumnTransformer`:

```python
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols)
])
```

This allows the model to process all numerical features consistently within a single pipeline.

---

# 4. Train-Test Split

The dataset is divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

### `random_state=42`

Ensures that the same split can be reproduced.

### `stratify=y`

This is especially important for fraud detection.

Because fraudulent transactions are much less frequent than legitimate transactions, stratification attempts to preserve the class distribution in both training and testing sets.

---

# 5. Baseline Machine Learning Models

Six different classification algorithms are evaluated:

```text
Logistic Regression
Decision Tree
Random Forest
XGBoost
LightGBM
CatBoost
```

The purpose is to compare different machine learning approaches rather than relying on a single algorithm.

---

# 6. Logistic Regression

```python
LogisticRegression(max_iter=1000)
```

Logistic Regression is a linear classification algorithm.

It estimates the probability that a transaction belongs to a particular class.

In this project:

```text
0 → Legitimate Transaction
1 → Fraudulent Transaction
```

It provides a useful baseline because it is relatively simple and interpretable.

---

# 7. Decision Tree

```python
DecisionTreeClassifier()
```

A Decision Tree makes predictions using a sequence of decision rules.

Conceptually:

```text
Transaction Amount > Threshold?
          ↓
       Yes / No
          ↓
   Another condition
          ↓
     Fraud / Legitimate
```

Decision Trees can model nonlinear relationships between features.

---

# 8. Random Forest

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
```

Random Forest is an ensemble of multiple Decision Trees.

Each tree produces a prediction, and the trees are combined to produce the final prediction.

### Important Parameters

* `n_estimators=200` → Creates 200 decision trees.
* `class_weight="balanced"` → Assigns greater importance to the minority class based on class frequencies.

---

# 9. XGBoost

```python
XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42
)
```

XGBoost is a gradient boosting algorithm that builds trees sequentially.

Each new tree attempts to improve the errors made by previous trees.

Important parameters include:

* `n_estimators` → number of boosting trees
* `max_depth` → maximum tree depth
* `learning_rate` → contribution of each tree
* `eval_metric` → evaluation metric used during training

---

# 10. LightGBM

```python
LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    class_weight="balanced"
)
```

LightGBM is another gradient boosting framework designed for efficient and scalable tree-based learning.

It uses a leaf-wise tree growth strategy and can perform well on structured/tabular datasets.

`class_weight="balanced"` helps account for the imbalanced target distribution.

---

# 11. CatBoost

```python
CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    verbose=0,
    auto_class_weights="Balanced"
)
```

CatBoost is a gradient boosting algorithm with robust regularization and high performance.

Important parameters include:

* `iterations` → number of boosting iterations
* `depth` → tree depth
* `learning_rate` → learning rate
* `auto_class_weights="Balanced"` → automatically adjusts class weights

---

# 12. Model Evaluation

Each model is evaluated using:

```text
Accuracy
Precision
Recall
F1-Score
ROC-AUC
```

---

## 12.1 Accuracy

Accuracy measures the proportion of all predictions that are correct.

```text
Accuracy = Correct Predictions / Total Predictions
```

However, accuracy can be misleading for highly imbalanced fraud datasets.

For example, if 99.8% of transactions are legitimate, a model that predicts almost everything as legitimate could achieve very high accuracy while detecting very little fraud.

Therefore, accuracy is not sufficient by itself.

---

## 12.2 Precision

Precision measures how many transactions predicted as fraud are actually fraudulent.

```text
Precision = TP / (TP + FP)
```

Where:

* TP = True Positive
* FP = False Positive

High precision means fewer legitimate transactions are incorrectly flagged as fraud.

---

## 12.3 Recall

Recall measures how many actual fraudulent transactions were successfully detected.

```text
Recall = TP / (TP + FN)
```

Where:

* TP = True Positive
* FN = False Negative

Recall is particularly important in fraud detection because a **False Negative** means that an actual fraudulent transaction was missed.

---

## 12.4 F1-Score

F1-score combines precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

It provides a balance between precision and recall when both false positives and false negatives matter.

---

## 12.5 ROC-AUC

ROC-AUC measures how well the model separates the two classes across different classification thresholds.

The model uses `predict_proba(X_test)[:, 1]` to obtain the predicted probability of the fraud class.

A higher ROC-AUC generally indicates better class discrimination.

---

# 13. Baseline Model Comparison

The trained models are stored and their evaluation results are collected:

```text
Logistic Regression
Decision Tree
Random Forest
XGBoost
LightGBM
CatBoost
        ↓
Performance Comparison
```

The model with the highest F1-score is selected as:

```python
best_model_name = results_df["f1"].idxmax()
```

---

# 14. Hyperparameter Optimization

After identifying the best baseline model, `RandomizedSearchCV` is used to search for better hyperparameter combinations.

```python
RandomizedSearchCV(
    pipeline,
    param_grid[best_model_name],
    n_iter=10,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    random_state=42
)
```

The optimization process uses:

```text
Parameter Search
       ↓
RandomizedSearchCV
       ↓
3-Fold Cross-Validation
       ↓
10 Random Parameter Combinations
       ↓
ROC-AUC Optimization
       ↓
Best Hyperparameters
```

---

# 15. Tuned Best Model

The best hyperparameter configuration is obtained using `search.best_params_`.

The optimized model is stored in:

```python
best_model = search.best_estimator_
```

---

# 16. Ensemble Learning

The project uses ensemble learning to combine multiple tree-based models:

```text
Soft Voting
Stacking
```

---

## 16.1 Soft Voting

The `VotingClassifier` combines:

```text
XGBoost
LightGBM
Random Forest
```

using `voting="soft"`.

Conceptually:

```text
              ┌── XGBoost ──────┐
Input ────────┼── LightGBM ─────┼──→ Probability Combination
              └── RandomForest ─┘
                         ↓
                   Final Prediction
```

Instead of simple hard voting, the ensemble combines their continuous probability estimates.

---

## 16.2 Stacking

Stacking uses several base estimators (`XGBoost`, `Random Forest`, `LightGBM`) passed to a meta-classifier:

```text
                 XGBoost
                    │
                 Random Forest
                    │
                 LightGBM
                    │
                    ↓
             Base Predictions
                    ↓
          Logistic Regression
                    ↓
             Final Prediction
```

The final estimator learns how to optimally combine the predictions from the base models.

---

# 17. SMOTE-Tomek

The project uses `SMOTETomek(random_state=42)` to address class imbalance:

```text
SMOTE (Synthetic Minority Over-sampling)
+
Tomek Links (Boundary Cleaning)
```

```text
SMOTE
   ↓
Increase Minority Samples
   ↓
Tomek Links
   ↓
Clean Overlapping Samples
   ↓
Balanced Training Data
```

---

# 18. Final Model Comparison

Four approaches are compared:

```text
Best Tuned Model
Voting
Stacking
SMOTE-Tomek Model
```

The results are organized into a DataFrame and sorted according to ROC-AUC:

```python
pd.DataFrame(final_results).T.sort_values("roc_auc", ascending=False)
```

---

# 19. Final Model

The final model in the code is explicitly assigned as:

```python
best_final_model = voting_pipe
```

Therefore, the **Voting ensemble is used as the final prediction model** in the subsequent confusion-matrix step:

```text
XGBoost + LightGBM + Random Forest
              ↓
         Soft Voting
              ↓
    Final Fraud Prediction
```

---

# 20. Confusion Matrix

The final model predictions are evaluated using a confusion matrix:

```python
cm = confusion_matrix(y_test, y_pred)
```

```text
                    Predicted
                 Legitimate   Fraud
Actual
Legitimate           TN         FP
Fraud                FN         TP
```

* **True Negative (TN):** Legitimate transaction correctly classified as legitimate.
* **False Positive (FP):** Legitimate transaction incorrectly flagged as fraud.
* **False Negative (FN):** Fraudulent transaction incorrectly missed as legitimate.
* **True Positive (TP):** Fraudulent transaction correctly detected as fraud.

---

# 21. SHAP Explainability

The project uses **SHAP (SHapley Additive exPlanations)** with `TreeExplainer` on the trained model to interpret individual and global feature contributions:

```python
best_shap_model = trained_models["LightGBM"]
light_model = best_shap_model.named_steps["model"]
explainer = shap.TreeExplainer(light_model)
shap_values = explainer.shap_values(X_transformed)
```

---

# Complete Architecture

```text
                         CREDIT CARD DATASET
                                │
                                ▼
                         ARFF DATA LOADING
                                │
                                ▼
                       PANDAS DATAFRAME
                                │
                                ▼
                       NUMERICAL FEATURES
                     (Time, V1–V28, Amount)
                                │
                                ▼
                        MEDIAN IMPUTATION
                                │
                                ▼
                         ROBUST SCALING
                                │
                                ▼
                           PREPROCESSOR
                                │
                                ▼
                       TRAIN-TEST SPLIT
                         /            \
                        /              \
                   80% TRAIN         20% TEST
                        │
                        ▼
                 BASELINE MODELS
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
 Logistic Regression  Decision Tree   Random Forest
        │
        ├──────── XGBoost
        │
        ├──────── LightGBM
        │
        └──────── CatBoost
                        │
                        ▼
                  MODEL EVALUATION
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
      Precision       Recall        F1-Score
          │             │              │
          └─────────────┼──────────────┘
                        │
                     ROC-AUC
                        │
                        ▼
                 BEST MODEL SELECTION
                        │
                        ▼
             HYPERPARAMETER OPTIMIZATION
                        │
                 RandomizedSearchCV
                        │
                 3-Fold Cross Validation
                        │
                        ▼
                  TUNED BEST MODEL
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
        SOFT VOTING             STACKING
             │                     │
      XGBoost                    XGBoost
      LightGBM                   Random Forest
      Random Forest              LightGBM
             │                     │
             │              Logistic Regression
             │                     │
             └──────────┬──────────┘
                        │
                        ▼
                  SMOTE-TOMEK
                        │
                        ▼
              FINAL MODEL COMPARISON
                        │
                        ▼
                  FINAL MODEL
                        │
                        ▼
                 CONFUSION MATRIX
                        │
                        ▼
                SHAP EXPLAINABILITY
                        │
                        ▼
               FEATURE IMPORTANCE
               & MODEL EXPLANATION
```

# Technologies Used

* Python
* NumPy
* Pandas
* SciPy
* Scikit-learn
* Imbalanced-learn
* XGBoost
* LightGBM
* CatBoost
* SHAP
* Matplotlib
* Joblib

# Key Techniques

* Numerical data preprocessing
* Missing-value median imputation
* Robust scaling (outlier-resistant)
* Stratified train-test splitting
* Multiple machine learning classifiers
* Hyperparameter optimization
* RandomizedSearchCV & Cross-validation
* Ensemble learning (Soft Voting & Stacking)
* SMOTE-Tomek class balancing
* Confusion matrix analysis
* SHAP explainability & feature importance

# Evaluation Strategy

Because credit card fraud detection is highly imbalanced, the project does not rely only on accuracy.

The main evaluation metrics are:

```text
Accuracy
Precision
Recall
F1-Score
ROC-AUC
```

---

# Decision Threshold & Fraud Sensitivity

Machine learning classifiers calculate a continuous probability score $P(\text{Fraud}) \in [0.0, 1.0]$ for each transaction. The **Decision Threshold** (Fraud Sensitivity) is the cutoff boundary used to classify a transaction as **FRAUD** vs. **LEGITIMATE**:

$$\text{Predicted Class} = \begin{cases} \text{FRAUD (1)}, & \text{if } P(\text{Fraud}) \ge \text{Threshold} \\ \text{LEGITIMATE (0)}, & \text{if } P(\text{Fraud}) < \text{Threshold} \end{cases}$$

```text
       0.0 ─────────────────── [ Threshold ] ─────────────────── 1.0
             Legitimate                      Fraudulent
```

### Impact on Fraud Predictions:

| Threshold Adjustment | Operational Impact | Metric Trade-off | Business Context |
| :--- | :--- | :--- | :--- |
| **Lower Threshold**<br>*(e.g., 0.20 – 0.35)*<br>**High Sensitivity** | Catches more fraud attempts; flags borderline suspicious transactions. | **High Recall (Sensitivity)**<br>Lower Precision (More False Positives) | Used for high-value transactions, cross-border payments, or strict fraud prevention policies where missing a fraud is costlier than a manual review. |
| **Default Threshold**<br>*(0.50)* | Standard balanced classification boundary. | Balanced Precision & Recall | Standard baseline across machine learning libraries. |
| **Higher Threshold**<br>*(e.g., 0.65 – 0.85)*<br>**Low Sensitivity** | Flags transactions only when confidence is extremely high; minimizes false alarms. | **High Precision**<br>Lower Recall (More False Negatives) | Used when false declines directly harm user checkout conversion or customer trust (e.g., instant micro-payments). |

---

# Explainable AI

SHAP is incorporated to make the machine learning model transparent and interpretable:

* Identifies which features drive predictions.
* Pinpoints factors pushing transactions toward fraud vs. legitimate classifications.
* Provides feature attribution rankings for regulatory compliance and auditing.

# Final Workflow

```text
Data Collection
      ↓
Data Preprocessing (Median Imputation + Robust Scaling)
      ↓
Train-Test Split
      ↓
Baseline Model Training
      ↓
Model Evaluation
      ↓
Best Model Selection
      ↓
Hyperparameter Optimization
      ↓
Ensemble Learning
      ↓
SMOTE-Tomek
      ↓
Final Model Comparison
      ↓
Final Model Selection
      ↓
Confusion Matrix
      ↓
SHAP Explainability
      ↓
Fraud Detection + Model Interpretation
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

