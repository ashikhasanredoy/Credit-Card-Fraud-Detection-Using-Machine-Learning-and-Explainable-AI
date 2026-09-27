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
Data Preprocessing
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

The project uses a `ColumnTransformer` to apply different preprocessing techniques to numerical and categorical features.

The preprocessing pipeline is divided into two parts:

```text
Numerical Features
       ↓
Median Imputation
       ↓
Robust Scaling

Categorical Features
       ↓
Most-Frequent Imputation
       ↓
One-Hot Encoding
```

---

## 2.1 Numerical Feature Processing

```python
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])
```

### Median Imputation

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

It is particularly useful when the dataset contains outliers.

Unlike standard scaling, RobustScaler is less sensitive to extreme observations.

---

# 3. Categorical Feature Processing

For categorical variables, the project uses:

```python
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse_output=False
    ))
])
```

---

## 3.1 Most-Frequent Imputation

```python
SimpleImputer(strategy="most_frequent")
```

Missing categorical values are replaced with the most frequently occurring category.

For example:

```text
Visa
Visa
MasterCard
Missing
Visa
```

The missing value becomes:

```text
Visa
```

because Visa is the most frequent category.

---

## 3.2 One-Hot Encoding

```python
OneHotEncoder()
```

Machine learning algorithms generally require numerical inputs.

One-hot encoding converts categorical values into numerical columns.

For example:

```text
Card Type

Visa
MasterCard
```

can become:

```text
Visa    MasterCard
1       0
0       1
```

`drop="first"` removes one category to reduce redundant information.

`handle_unknown="ignore"` prevents errors when the test set contains a category that was not present during training.

---

# 4. ColumnTransformer

The numerical and categorical preprocessing pipelines are combined using `ColumnTransformer`.

```python
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])
```

This allows the model to process different types of features correctly within a single pipeline.

---

# 5. Train-Test Split

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

# 6. Baseline Machine Learning Models

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

# 7. Logistic Regression

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

# 8. Decision Tree

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

# 9. Random Forest

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

`n_estimators=200`

Creates 200 decision trees.

`class_weight="balanced"`

Assigns greater importance to the minority class based on class frequencies.

This is useful for imbalanced fraud datasets.

---

# 10. XGBoost

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

# 11. LightGBM

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

# 12. CatBoost

```python
CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    verbose=0,
    auto_class_weights="Balanced"
)
```

CatBoost is a gradient boosting algorithm developed with strong support for categorical features.

Important parameters include:

* `iterations` → number of boosting iterations
* `depth` → tree depth
* `learning_rate` → learning rate
* `auto_class_weights="Balanced"` → automatically adjusts class weights

---

# 13. Model Evaluation

Each model is evaluated using:

```text
Accuracy
Precision
Recall
F1-Score
ROC-AUC
```

The evaluation function calculates these metrics.

---

## 13.1 Accuracy

Accuracy measures the proportion of all predictions that are correct.

```text
Accuracy =
Correct Predictions / Total Predictions
```

However, accuracy can be misleading for highly imbalanced fraud datasets.

For example, if 99.8% of transactions are legitimate, a model that predicts almost everything as legitimate could achieve very high accuracy while detecting very little fraud.

Therefore, accuracy is not sufficient by itself.

---

# 14. Precision

Precision measures how many transactions predicted as fraud are actually fraudulent.

```text
Precision =
TP / (TP + FP)
```

Where:

* TP = True Positive
* FP = False Positive

High precision means fewer legitimate transactions are incorrectly flagged as fraud.

---

# 15. Recall

Recall measures how many actual fraudulent transactions were successfully detected.

```text
Recall =
TP / (TP + FN)
```

Where:

* TP = True Positive
* FN = False Negative

Recall is particularly important in fraud detection because a **False Negative** means that an actual fraudulent transaction was missed.

---

# 16. F1-Score

F1-score combines precision and recall.

```text
F1 =
2 × Precision × Recall
-----------------------
Precision + Recall
```

It provides a balance between precision and recall.

This is useful when both false positives and false negatives matter.

---

# 17. ROC-AUC

ROC-AUC measures how well the model separates the two classes across different classification thresholds.

The model uses:

```python
predict_proba(X_test)[:, 1]
```

to obtain the predicted probability of the fraud class.

A higher ROC-AUC generally indicates better class discrimination.

---

# 18. Baseline Model Comparison

The trained models are stored and their evaluation results are collected.

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

This makes F1-score the criterion for selecting the best baseline model.

---

# 19. Hyperparameter Optimization

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

## 19.1 RandomizedSearchCV

Instead of testing every possible combination, RandomizedSearchCV randomly samples combinations from the parameter search space.

This can significantly reduce computational cost when many combinations are available.

---

## 19.2 3-Fold Cross-Validation

The training data is divided into three folds.

Conceptually:

```text
Fold 1 → Validation
Fold 2 + Fold 3 → Training

Fold 2 → Validation
Fold 1 + Fold 3 → Training

Fold 3 → Validation
Fold 1 + Fold 2 → Training
```

The process is repeated so that each fold serves as validation data.

---

# 20. Tuned Best Model

The best hyperparameter configuration is obtained using:

```python
search.best_params_
```

The optimized model is stored in:

```python
best_model = search.best_estimator_
```

This creates the tuned version of the selected baseline model.

---

# 21. Ensemble Learning

The project also uses ensemble learning to combine multiple tree-based models.

Two ensemble techniques are implemented:

```text
Soft Voting
Stacking
```

---

# 22. Soft Voting

The VotingClassifier combines:

```text
XGBoost
LightGBM
Random Forest
```

using:

```python
voting="soft"
```

Soft voting uses the predicted class probabilities from the individual models.

Conceptually:

```text
              ┌── XGBoost ──────┐
Input ────────┼── LightGBM ─────┼──→ Probability Combination
              └── RandomForest ─┘
                         ↓
                   Final Prediction
```

Instead of simply asking each model for a class label, the ensemble combines their probability estimates.

---

# 23. Stacking

Stacking uses several models as base estimators:

```text
XGBoost
Random Forest
LightGBM
```

Their outputs are then passed to a final Logistic Regression model.

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

The final estimator learns how to combine the predictions from the base models.

---

# 24. SMOTE-Tomek

The project also uses:

```python
SMOTETomek(random_state=42)
```

to address class imbalance.

SMOTE-Tomek combines two techniques:

```text
SMOTE
+
Tomek Links
```

---

## 24.1 SMOTE

SMOTE stands for:

**Synthetic Minority Over-sampling Technique**

It creates synthetic samples for the minority class.

Instead of simply duplicating existing fraud transactions, SMOTE generates new synthetic minority examples based on neighboring minority samples.

Conceptually:

```text
Few Fraud Samples
       ↓
      SMOTE
       ↓
Synthetic Fraud Samples
       ↓
More Balanced Training Data
```

---

## 24.2 Tomek Links

Tomek Links identify pairs of samples from different classes that are very close to each other.

Removing Tomek links can help clean ambiguous samples near the class boundary.

Therefore:

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

# 25. Final Model Comparison

Four approaches are compared:

```text
Best Tuned Model
Voting
Stacking
SMOTE-Tomek Model
```

The results are organized into a DataFrame:

```python
pd.DataFrame(final_results).T
```

and sorted according to ROC-AUC.

```python
.sort_values("roc_auc", ascending=False)
```

This allows the different approaches to be compared using the same evaluation metrics.

---

# 26. Final Model

The final model in the provided code is explicitly assigned as:

```python
best_final_model = voting_pipe
```

Therefore, the **Voting ensemble is used as the final prediction model** in the subsequent confusion-matrix step.

The final model consists of:

```text
XGBoost
+
LightGBM
+
Random Forest
        ↓
Soft Voting
        ↓
Final Fraud Prediction
```

---

# 27. Confusion Matrix

The final model's predictions are evaluated using a confusion matrix.

```python
cm = confusion_matrix(y_test, y_pred)
```

The confusion matrix contains four outcomes:

```text
                    Predicted
                 Legitimate   Fraud
Actual
Legitimate           TN         FP
Fraud                FN         TP
```

### True Negative (TN)

A legitimate transaction correctly classified as legitimate.

### False Positive (FP)

A legitimate transaction incorrectly classified as fraud.

### False Negative (FN)

A fraudulent transaction incorrectly classified as legitimate.

### True Positive (TP)

A fraudulent transaction correctly detected as fraud.

For fraud detection, **False Negatives are particularly important** because they represent fraudulent transactions that the system failed to detect.

---

# 28. SHAP Explainability

The project uses **SHAP (SHapley Additive exPlanations)** to explain model predictions.

The SHAP section uses the trained LightGBM model:

```python
best_shap_model = trained_models["LightGBM"]
```

The purpose is to understand which features contribute to the model's predictions.

Instead of only producing:

```text
Prediction = Fraud
```

SHAP can help explain:

```text
Why was this transaction classified as fraud?
```

---

# 29. Feature Transformation for SHAP

The test data is transformed using the same preprocessing pipeline:

```python
X_transformed = (
    best_shap_model
    .named_steps["preprocessor"]
    .transform(X_test)
)
```

This ensures that the data supplied to the LightGBM model has the same representation used during training.

---

# 30. Feature Name Extraction

After preprocessing, the original features may be transformed into multiple features, especially after one-hot encoding.

The transformed feature names are obtained using:

```python
feature_names = (
    best_shap_model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)
```

The number of resulting features is then checked:

```python
print("Number of features:", len(feature_names))
```

This is useful because the number of features after preprocessing may differ from the number of original dataset columns.

---

# 31. LightGBM Model Extraction

The trained LightGBM model is extracted from the pipeline:

```python
light_model = best_shap_model.named_steps["model"]
```

Its type is then checked:

```python
print(type(light_model))
```

This provides the underlying LightGBM estimator that can be passed to SHAP for model explanation.

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
                    FEATURE IDENTIFICATION
                         /              \
                        /                \
               Numerical Features    Categorical Features
                     │                       │
              Median Imputation       Most-Frequent Imputation
                     │                       │
              Robust Scaling            One-Hot Encoding
                     \                       /
                      \                     /
                       └──── PREPROCESSOR ─┘
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

* Data preprocessing
* Missing-value imputation
* Robust scaling
* One-hot encoding
* Stratified train-test splitting
* Multiple machine learning classifiers
* Hyperparameter optimization
* RandomizedSearchCV
* Cross-validation
* Ensemble learning
* Soft voting
* Stacking
* SMOTE-Tomek
* Confusion matrix analysis
* SHAP explainability

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

The confusion matrix is also used to analyze:

```text
True Positives
True Negatives
False Positives
False Negatives
```

This provides a more detailed view of how effectively the system identifies fraudulent transactions.

# Explainable AI

SHAP is incorporated to make the machine learning model more interpretable.

The explainability stage helps identify:

* Which features influence predictions
* Which features contribute toward fraud predictions
* Which features contribute toward legitimate predictions
* How strongly individual features affect model output

This makes the fraud detection system more transparent than a black-box prediction system alone.

# Final Workflow

```text
Data Collection
      ↓
Data Preprocessing
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
