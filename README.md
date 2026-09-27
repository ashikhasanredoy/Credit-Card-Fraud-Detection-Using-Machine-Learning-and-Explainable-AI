# Credit Card Fraud Detection Using Machine Learning and Explainable AI

## Project Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning and Explainable AI (XAI).

Credit card fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent only a very small portion of all transactions. Therefore, this project focuses not only on overall accuracy but also on **Precision, Recall, F1-Score, ROC-AUC, and PR-AUC**.

The project evaluates multiple machine learning algorithms, handles class imbalance using **SMOTE-Tomek**, performs hyperparameter optimization and ensemble learning, and uses **SHAP** to explain model predictions.

---

## Dataset

The dataset contains **284,807 credit card transactions** with **31 columns**.

### Features

| Feature      | Description                                                    |
| ------------ | -------------------------------------------------------------- |
| `Time`       | Time elapsed between the transaction and the first transaction |
| `V1` – `V28` | PCA-transformed numerical features                             |
| `Amount`     | Transaction amount                                             |
| `Class`      | Target variable                                                |

### Target Variable

The `Class` column represents whether a transaction is fraudulent.

```text
0 → Legitimate Transaction
1 → Fraudulent Transaction
```

### Input and Target Separation

```python
X = df.drop(columns=['Class'])
y = df['Class']
```

Here:

* `X` contains the input features.
* `y` contains the target variable.

---

# Machine Learning Pipeline

```text
                         ┌─────────────────────┐
                         │       Dataset       │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │    Data Cleaning    │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │ Exploratory Data    │
                         │      Analysis       │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │ Feature Identification│
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │   Train-Test Split  │
                         │       80 / 20       │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ↓                                   ↓
       ┌─────────────────────┐             ┌─────────────────────┐
       │    Preprocessing    │             │ Class Imbalance     │
       │                     │             │     Handling         │
       │ Imputation          │             │                     │
       │ Scaling             │             │ Original             │
       │ Encoding            │             │ SMOTE-Tomek          │
       └──────────┬──────────┘             └──────────┬──────────┘
                  └─────────────────┬─────────────────┘
                                    ↓
                         ┌─────────────────────┐
                         │   Baseline Models  │
                         └──────────┬──────────┘
                                    ↓
              ┌──────────┬──────────┼──────────┬──────────┬──────────┐
              ↓          ↓          ↓          ↓          ↓          ↓
             LR         DT         RF       XGBoost   LightGBM   CatBoost
              └──────────┴──────────┴──────────┴──────────┴──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │ Cross-Validation    │
                         │    Stratified CV    │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │ Performance         │
                         │ Evaluation          │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │ Model / Experiment  │
                         │     Selection       │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │ Hyperparameter      │
                         │   Optimization      │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │    Tuned Model      │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │ Ensemble Learning   │
                         ├─────────────────────┤
                         │ Voting              │
                         │ Stacking            │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │ Final Comparison    │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │ Final Model         │
                         │    Selection        │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │   Test Evaluation   │
                         └──────────┬──────────┘
                                    ↓
              ┌─────────────────────┼─────────────────────┐
              ↓                     ↓                     ↓
       Confusion Matrix          ROC-AUC                PR-AUC
              ↓                                           ↓
          Precision                                    Recall
                                                        ↓
                                                       F1
                                    ↓
                         ┌─────────────────────┐
                         │ SHAP Explainability │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │ Feature Importance │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │ Final Conclusions   │
                         └─────────────────────┘
```

---

# 1. Dataset Loading

The dataset is loaded from an ARFF file and converted into a Pandas DataFrame.

```python
from scipy.io import arff
import pandas as pd

data, meta = arff.loadarff("dataset.arff")

df = pd.DataFrame(data)
```

If the dataset contains byte strings, they are decoded before further processing.

---

# 2. Data Cleaning

The dataset is checked for:

* Missing values
* Duplicate records
* Incorrect data types
* Invalid values

The target column is:

```python
Class
```

The input and target variables are separated using:

```python
X = df.drop(columns=['Class'])
y = df['Class']
```

---

# 3. Exploratory Data Analysis

EDA is performed to understand the dataset before model training.

The analysis includes:

* Dataset shape
* Data types
* Missing values
* Duplicate records
* Class distribution
* Numerical feature statistics
* Feature distributions
* Outlier analysis
* Correlation analysis
* Fraud vs legitimate transaction comparison

Because fraud is rare, class distribution is particularly important.

---

# 4. Feature Identification

The dataset contains:

```text
Time
V1
V2
...
V28
Amount
```

These features are used as input variables.

The target variable is:

```text
Class
```

Therefore:

```python
X = df.drop(columns=['Class'])
y = df['Class']
```

---

# 5. Train-Test Split

The dataset is divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The `stratify=y` parameter preserves the class distribution between the training and testing sets.

The test set is kept separate for final model evaluation.

---

# 6. Preprocessing

A preprocessing pipeline is created for the numerical and categorical features.

### Numerical Features

Numerical preprocessing includes:

```text
Missing Value Imputation
        ↓
Median
        ↓
Robust Scaling
```

RobustScaler is useful when numerical variables contain outliers.

### Categorical Features

Categorical preprocessing includes:

```text
Missing Value Imputation
        ↓
Most Frequent Value
        ↓
One-Hot Encoding
```

All preprocessing operations are placed inside the machine learning pipeline to reduce the risk of data leakage.

---

# 7. Class Imbalance Handling

Fraud detection contains a highly imbalanced target distribution.

Two experimental settings are considered:

### Original Dataset

The original training data is used without synthetic oversampling.

### SMOTE-Tomek

SMOTE generates synthetic minority-class samples while Tomek Links removes overlapping or ambiguous samples.

```text
Original Training Data
          ↓
        SMOTE
          ↓
Synthetic Fraud Samples
          ↓
      Tomek Links
          ↓
Cleaned Balanced Training Data
```

SMOTE-Tomek is applied only to the training data through an `imblearn` pipeline.

---

# 8. Baseline Models

Six machine learning algorithms are evaluated:

### Logistic Regression

A linear classification algorithm used as a baseline.

### Decision Tree

A tree-based model that learns decision rules from the features.

### Random Forest

An ensemble of multiple decision trees.

### XGBoost

A gradient boosting algorithm based on sequential decision trees.

### LightGBM

A gradient boosting framework optimized for efficient tree-based learning.

### CatBoost

A gradient boosting algorithm designed to provide strong performance with minimal preprocessing.

---

# 9. Cross-Validation

Stratified K-Fold Cross-Validation is used to evaluate model performance on the training data.

```text
Training Data
     ↓
 ┌───┴───┐
 │       │
Fold 1  Fold 2 ... Fold 5
 │       │
 └───┬───┘
     ↓
Average CV Performance
```

Using stratification helps preserve the minority fraud class in each fold.

---

# 10. Performance Evaluation

Multiple metrics are used because accuracy alone is not sufficient for fraud detection.

### Accuracy

Measures the proportion of correctly classified transactions.

### Precision

Measures how many transactions predicted as fraud are actually fraudulent.

```text
Precision = TP / (TP + FP)
```

### Recall

Measures how many actual fraudulent transactions are detected.

```text
Recall = TP / (TP + FN)
```

Recall is particularly important because missing fraudulent transactions can be costly.

### F1-Score

The harmonic mean of Precision and Recall.

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

### ROC-AUC

Measures the model's ability to distinguish between fraudulent and legitimate transactions across classification thresholds.

### PR-AUC

Average Precision / PR-AUC focuses on the Precision-Recall relationship and is especially informative for highly imbalanced classification problems.

---

# 11. Model Selection

Model selection is performed using predefined performance criteria from the training-set cross-validation results.

For a highly imbalanced fraud detection problem, PR-AUC, Recall, Precision, and F1-Score are considered important evaluation metrics.

The test set should not be used repeatedly for model selection.

---

# 12. Hyperparameter Optimization

After selecting the model/experiment configuration using training-set cross-validation, hyperparameters are optimized using `RandomizedSearchCV`.

The optimization uses:

```text
5-Fold Stratified Cross-Validation
```

The scoring metric can be:

```python
scoring="average_precision"
```

This allows the optimization process to focus on Precision-Recall performance.

---

# 13. Tuned Model

The best hyperparameters obtained from cross-validation are used to train the optimized model.

The tuned model is then evaluated against the ensemble models.

---

# 14. Ensemble Learning

Two ensemble learning methods are implemented.

## Voting Ensemble

Voting combines predictions from multiple models.

```text
XGBoost
   │
LightGBM ───→ Voting Ensemble
   │
Random Forest
```

Soft voting combines predicted probabilities from the individual models.

---

## Stacking Ensemble

Stacking uses multiple base models and a meta-model.

```text
XGBoost ───────┐
               │
Random Forest ─┼──→ Logistic Regression
               │       Meta Model
LightGBM ──────┘
```

The base models generate predictions that are passed to the meta-model.

---

# 15. Final Model Comparison

The following configurations are compared:

```text
Baseline Models
       ↓
SMOTE-Tomek Models
       ↓
Tuned Model
       ↓
Voting Ensemble
       ↓
Stacking Ensemble
```

The comparison includes:

* Precision
* Recall
* F1-Score
* ROC-AUC
* PR-AUC

---

# 16. Final Model Selection

The final model is selected using the predefined evaluation criterion based on training-set cross-validation results.

The test set remains untouched until the final evaluation.

---

# 17. Final Test Evaluation

After the final model has been selected, it is evaluated on the untouched test set.

The main evaluation includes:

```text
Confusion Matrix
Precision
Recall
F1-Score
ROC-AUC
PR-AUC
```

---

# 18. Confusion Matrix

The confusion matrix contains four values:

```text
                 Predicted
                0        1
Actual  0      TN       FP
        1      FN       TP
```

Where:

* **TN** = True Negative
* **FP** = False Positive
* **FN** = False Negative
* **TP** = True Positive

For fraud detection, the number of **False Negatives** is particularly important because they represent fraudulent transactions that were not detected.

---

# 19. SHAP Explainability

SHAP (SHapley Additive exPlanations) is used to explain model predictions.

SHAP helps answer:

```text
Why did the model classify this transaction as fraud?
```

It provides information about how individual features contribute to predictions.

---

# 20. Feature Importance

SHAP feature importance can be used to identify the features that have the greatest influence on model predictions.

The analysis can provide:

* Global feature importance
* Positive/negative feature contribution
* Individual transaction explanations

For an ensemble final model, SHAP should be interpreted carefully. A tree-based component can be explained directly with `TreeExplainer`, while explaining the complete ensemble requires an appropriate model-agnostic SHAP approach.

---

# 21. Research Methodology Summary

The complete methodology is:

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Identification
   ↓
Train-Test Split
   ↓
Preprocessing
   ↓
Class Imbalance Experiments
   ├── Original
   └── SMOTE-Tomek
   ↓
Baseline Models
   ├── Logistic Regression
   ├── Decision Tree
   ├── Random Forest
   ├── XGBoost
   ├── LightGBM
   └── CatBoost
   ↓
Stratified Cross-Validation
   ↓
Performance Evaluation
   ↓
Model / Experiment Selection
   ↓
Hyperparameter Optimization
   ↓
Tuned Model
   ↓
Ensemble Learning
   ├── Voting
   └── Stacking
   ↓
Final Comparison
   ↓
Final Model Selection
   ↓
Untouched Test Set Evaluation
   ↓
Confusion Matrix
   ↓
ROC-AUC
   ↓
PR-AUC
   ↓
Precision / Recall / F1
   ↓
SHAP Explainability
   ↓
Feature Importance
   ↓
Final Conclusions
```

---

# 22. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* XGBoost
* LightGBM
* CatBoost
* SHAP
* Matplotlib
* Joblib
* SciPy

---

# 23. Project Objective

The primary objectives of this project are:

1. Detect fraudulent credit card transactions.
2. Address the severe class imbalance problem.
3. Compare multiple machine learning algorithms.
4. Optimize the selected model.
5. Investigate ensemble learning using Voting and Stacking.
6. Evaluate models using fraud-detection-oriented metrics.
7. Explain model predictions using SHAP.
8. Identify important features contributing to fraud detection.
9. Develop a reliable and interpretable machine learning pipeline.

---

# 24. Conclusion

This project develops a complete machine learning framework for credit card fraud detection by combining data preprocessing, exploratory analysis, class imbalance handling, machine learning, cross-validation, hyperparameter optimization, ensemble learning, and explainable AI.

The methodology emphasizes **Precision, Recall, F1-Score, ROC-AUC, and PR-AUC** rather than relying solely on accuracy. The use of SHAP further provides interpretability by showing how features influence fraud predictions.

The final model is selected using training-data validation and is evaluated only once on the untouched test set to provide an unbiased estimate of final performance.
