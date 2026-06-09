# Credit Card Fraud Detection Using Machine Learning and Explainable AI

## Overview

This project develops a machine learning-based fraud detection system capable of identifying fraudulent credit card transactions from highly imbalanced financial transaction data. Multiple classification algorithms, ensemble techniques, imbalance handling strategies, and Explainable AI (XAI) methods were implemented and evaluated to achieve robust fraud detection performance.

The project focuses not only on prediction accuracy but also on model interpretability using SHAP (SHapley Additive Explanations), enabling a deeper understanding of the factors influencing fraud predictions.

---

## Dataset Information

The dataset contains anonymized credit card transaction records with the following characteristics:

* Total Transactions: 284,807
* Features: 30 numerical features

  * Time
  * V1 – V28 (PCA-transformed features)
  * Amount
* Target Variable:

  * 0 → Legitimate Transaction
  * 1 → Fraudulent Transaction

### Class Distribution

| Class      | Percentage |
| ---------- | ---------- |
| Legitimate | 99.83%     |
| Fraud      | 0.17%      |

This extreme imbalance makes fraud detection a challenging classification problem.

---

## Project Workflow

```text
Dataset Collection
        ↓
Data Preprocessing
        ↓
Train-Test Split
        ↓
Model Training
        ↓
Hyperparameter Optimization
        ↓
Ensemble Learning
        ↓
SMOTE-Tomek Balancing
        ↓
Model Evaluation
        ↓
SHAP Explainability
        ↓
Model Deployment
```

---

## Data Preprocessing

### Numerical Features

* Missing Value Imputation (Median Strategy)
* RobustScaler for outlier-resistant scaling

### Train-Test Split

* Training Data: 80%
* Testing Data: 20%
* Stratified Sampling

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## Machine Learning Models

The following classification algorithms were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. LightGBM
6. CatBoost

---

## Baseline Model Performance

| Model               | Accuracy | Precision | Recall   | F1 Score | ROC-AUC  |
| ------------------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression | 0.999157 | 0.828947  | 0.642857 | 0.724138 | 0.959248 |
| Decision Tree       | 0.999140 | 0.752577  | 0.744898 | 0.748718 | 0.872238 |
| Random Forest       | 0.999526 | 0.961039  | 0.755102 | 0.845714 | 0.957188 |
| XGBoost             | 0.999526 | 0.908046  | 0.806122 | 0.854054 | 0.976979 |
| LightGBM            | 0.999508 | 0.857143  | 0.857143 | 0.857143 | 0.977775 |
| CatBoost            | 0.999175 | 0.710744  | 0.877551 | 0.785388 | 0.977896 |

### Best Baseline Model

**CatBoost Classifier**

* ROC-AUC: 0.977896
* Recall: 0.877551

---

## Hyperparameter Optimization

RandomizedSearchCV was used to optimize CatBoost.

### Best Parameters

```python
{
    'model__learning_rate': 0.01,
    'model__iterations': 300,
    'model__depth': 8
}
```

---

## Ensemble Learning

### Voting Ensemble

Base Models:

* Random Forest
* XGBoost
* LightGBM

### Stacking Ensemble

Base Models:

* Random Forest
* XGBoost
* LightGBM

Meta Learner:

* Logistic Regression

---

## Handling Class Imbalance

To address the severe class imbalance problem, SMOTE-Tomek was applied.

### SMOTE

Generates synthetic fraud samples.

### Tomek Links

Removes overlapping noisy samples between classes.

---

## Final Model Comparison

| Model             | Accuracy | Precision | Recall   | F1 Score | ROC-AUC  |
| ----------------- | -------- | --------- | -------- | -------- | -------- |
| SMOTE + CatBoost  | 0.997437 | 0.391892  | 0.887755 | 0.543750 | 0.982091 |
| Tuned CatBoost    | 0.998139 | 0.477778  | 0.877551 | 0.618705 | 0.980702 |
| Stacking Ensemble | 0.999561 | 0.939759  | 0.795918 | 0.861878 | 0.977423 |
| Voting Ensemble   | 0.999561 | 0.910112  | 0.826531 | 0.866310 | 0.977381 |

---

## Confusion Matrix (Stacking Ensemble)

```text
[[56859     5]
 [   20    78]]
```

### Interpretation

* True Negatives (TN): 56,859
* False Positives (FP): 5
* False Negatives (FN): 20
* True Positives (TP): 78

The Stacking Ensemble achieved an excellent balance between fraud detection capability and false alarm reduction.

---

## Explainable AI (SHAP)

SHAP was used to interpret the CatBoost model.

### Global Feature Importance

Top Features Influencing Fraud Detection:

| Rank | Feature | Importance |
| ---- | ------- | ---------- |
| 1    | V4      | 2.253770   |
| 2    | V14     | 2.171501   |
| 3    | V1      | 0.985991   |
| 4    | V10     | 0.772293   |
| 5    | V12     | 0.769748   |
| 6    | V3      | 0.623900   |
| 7    | V8      | 0.430380   |
| 8    | V11     | 0.407725   |
| 9    | Time    | 0.363391   |
| 10   | V18     | 0.317885   |

### SHAP Visualizations

* SHAP Summary Plot
* SHAP Feature Importance Plot
* SHAP Waterfall Plot
* SHAP Dependence Plot

These visualizations explain both global and local model behavior.

---

## Technologies Used

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* Scikit-Learn
* XGBoost
* LightGBM
* CatBoost
* Imbalanced-Learn
* SHAP
* Matplotlib
* Joblib

---

## Installation

```bash
git clone https://github.com/your-username/fraud-detection.git

cd fraud-detection

pip install -r requirements.txt
```

---

## Run Project

```bash
python fraud_detection.py
```

---

## Model Saving

```python
joblib.dump(best_final_model,
            "fraud_detection_model.pkl")
```

---

## Key Findings

* CatBoost achieved the best standalone ROC-AUC score.
* Stacking Ensemble provided the best balance of precision and recall.
* SMOTE-Tomek improved fraud detection recall significantly.
* V4 and V14 were identified as the most influential fraud indicators.
* Transaction Amount had relatively low predictive importance.
* SHAP successfully explained model predictions and feature contributions.

---

## Future Improvements

* Deep Learning Models (LSTM, Autoencoders)
* Real-Time Fraud Detection API
* Streamlit Dashboard Deployment
* Feature Engineering
* Cost-Sensitive Learning
* Advanced Explainability Techniques

---

# fydp-
