# Model Card: Best Fraud Detection Model

## Model Overview
- **Model Name**: Soft Voting Ensemble Fraud Classifier
- **Model File**: `best_fraud_model.pkl`
- **Architecture**: `VotingClassifier(voting='soft')` combining:
  - XGBoost Classifier (`XGBClassifier`)
  - LightGBM Classifier (`LGBMClassifier`)
  - Random Forest Classifier (`RandomForestClassifier`)
- **Preprocessing**: RobustScaler + SimpleImputer(median) in scikit-learn Pipeline
- **Date**: October 2026

## Performance Metrics on Test Set (20% Holdout, Stratified)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.999561 |
| **Precision** | 0.910112 |
| **Recall** | 0.826531 |
| **F1-Score** | 0.866310 |
| **ROC-AUC** | 0.977381 |

## Training Data Summary
- **Source**: European Cardholders Transaction Dataset (283,726 unique transactions after deduplication)
- **Features**: 29 numerical features (Time, V1–V28, Amount)
- **Target**: `Class` (Binary: 0 = Legitimate, 1 = Fraud)
- **Train/Test Split**: 80% Train / 20% Test stratified by `Class` (Random State: 42)

## Intended Use & Limitations
- **Intended Use**: Real-time / batch scoring of credit card transactions for fraud likelihood.
- **Limitations**: Trained on anonymized PCA transformed features from European transactions; model calibration and drift monitoring should be applied in new geographical or feature domains.
