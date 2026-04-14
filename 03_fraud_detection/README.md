# 03 — Credit Card Fraud Detection

## Overview
Binary classification project detecting fraudulent credit card transactions.
Key challenge: extreme class imbalance with only 0.17% fraud cases.

## Problem
Identify fraudulent transactions from 284,807 real credit card transactions
made by European cardholders in September 2013.

## Dataset
- **Source:** Kaggle — Credit Card Fraud Detection (ULB)
- **Size:** 284,807 transactions × 30 features
- **Fraud cases:** 492 (0.17%)
- **Features:** V1-V28 (PCA anonymized), Amount, Class

## Key Challenge
Extreme class imbalance — a model predicting everything as legitimate
would achieve 99.83% accuracy but catch zero fraud. Standard accuracy
is meaningless here — Recall and F1 are the key metrics.

## Approach
1. Feature scaling (StandardScaler on Amount)
2. Stratified train/test split
3. SMOTE oversampling on training data only
4. Three models trained and compared
5. Confusion matrix analysis

## SMOTE Results
- Before: 394 fraud vs 227,451 legitimate
- After: 227,451 fraud vs 227,451 legitimate

## Models Compared

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Random Forest** ⭐ | 0.8710 | 0.8265 | **0.8482** | 0.9132 |
| XGBoost | 0.6855 | 0.8673 | 0.7658 | 0.9333 |
| Logistic Regression | 0.0563 | 0.9184 | 0.1061 | 0.9459 |

## Key Findings
- Accuracy is misleading with imbalanced data — always use F1 and Recall
- Logistic Regression achieved 92% Recall but flagged almost everything as fraud — unusable
- Random Forest achieved the best F1 balance — best production model
- SMOTE successfully balanced training data and improved model performance
- In fraud detection, missing fraud (low Recall) is more costly than false alarms

## Business Insight
Choosing the right model depends on business priorities:
- Minimize financial loss → prioritize Recall → XGBoost
- Minimize customer friction → prioritize Precision → Random Forest

## Tools
Python, scikit-learn, XGBoost, imbalanced-learn (SMOTE),
pandas, numpy, matplotlib

## How to Run
```bash
pip install -r requirements.txt
python fraud_detection.py
python test_fraud.py
```

## Files
- `fraud_detection.py` — full pipeline
- `test_fraud.py` — test model with custom transaction
- Dataset: Download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
