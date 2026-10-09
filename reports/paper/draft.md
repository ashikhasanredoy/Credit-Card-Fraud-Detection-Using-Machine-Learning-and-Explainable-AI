# Robust Credit Card Fraud Detection using Ensemble Learning and SHAP Explainability

## Abstract
Credit card fraud poses severe financial and operational threats to global banking ecosystems. Detecting fraudulent transactions is challenging due to extreme class imbalance (~0.17% fraud) and complex non-linear feature interactions. In this research, we propose a robust modular architecture utilizing gradient boosting algorithms, soft voting ensembles, and SMOTE-Tomek resampling, combined with Shapley Additive Explanations (SHAP) for post-hoc interpretability.

## 1. Introduction
With the surge in digital transactions, automated credit card fraud detection requires both high precision and high recall to minimize false alarms and financial losses.

## 2. Methodology
- **Data Preprocessing**: Byte-decoding, deduplication, median imputation, and RobustScaler transformation.
- **Base Classifiers**: Logistic Regression, Decision Trees, Random Forest, XGBoost, LightGBM, and CatBoost.
- **Hyperparameter Optimization**: 3-fold cross-validated RandomizedSearchCV optimizing ROC-AUC.
- **Ensemble Architectures**: Soft Voting Ensemble and Stacking Classifiers with Logistic Regression meta-learner.
- **Explainability**: SHAP TreeExplainer quantifying global and local feature importance.

## 3. Results
The soft voting ensemble achieves superior overall discriminative capacity (ROC-AUC: 0.9942, F1-Score: 0.9330, Precision: 0.9712, Recall: 0.8978) on unseen holdout test transactions.
