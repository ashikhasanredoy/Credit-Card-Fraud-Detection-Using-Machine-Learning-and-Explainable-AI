# Credit Card Fraud Detection Dataset

## Overview
This dataset contains transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred over a two-day period, where 492 frauds were recorded out of 284,807 transactions.

The dataset is highly unbalanced: the positive class (frauds) accounts for only ~0.172% of all transactions.

## Feature Descriptions
- **Time**: Number of seconds elapsed between each transaction and the first transaction in the dataset.
- **V1 – V28**: Principal Component Analysis (PCA) features obtained due to confidentiality protection.
- **Amount**: Transaction amount.
- **Class**: Response variable (1 for fraud, 0 for legitimate transaction).

## Directory Structure
- `data/raw/dataset.arff`: Complete raw dataset in Attribute-Relation File Format (ARFF, 284,807 transactions) tracked via Git LFS. Loaded automatically via `src.data.load_data.load_arff`.
- `data/processed/eda.csv`: Cleaned dataset generated after decoding byte strings and removing duplicate records (283,726 transactions) tracked via Git LFS. Loaded automatically via `src.data.load_data.load_processed_csv`.

## Source & License
- Source: Machine Learning Group (MLG) - ULB (Université Libre de Bruxelles) / Kaggle Credit Card Fraud Detection.
- License: Open Database License (ODbL) / Database Contents License (DbCL).
