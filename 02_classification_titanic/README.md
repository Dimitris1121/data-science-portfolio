# 02 — Titanic Survival Classification

## Overview
Binary classification project predicting passenger survival on the Titanic.
Built to practice and demonstrate classification techniques, evaluation metrics,
and model explainability.

## Problem
Predict whether a passenger survived the Titanic disaster based on features
like sex, ticket class, age, and fare.

## Dataset
- **Source:** Titanic dataset (publicly available)
- **Size:** 891 passengers × 8 features after cleaning
- **Target:** Survived (1 = survived, 0 = died)
- **Class balance:** 62% died, 38% survived

## Data Cleaning
- Dropped irrelevant columns (PassengerId, Name, Ticket, Cabin)
- Filled missing Age with median
- Filled missing Embarked with mode
- Encoded Sex and Embarked numerically

## Models Compared

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Acc |
|---|---|---|---|---|---|---|
| **LightGBM (tuned)** ⭐ | 0.8380 | 0.8082 | 0.7973 | 0.8027 | 0.8320 | 0.8373 |
| Random Forest | 0.8212 | 0.8000 | 0.7568 | 0.7778 | 0.8117 | 0.8137 |
| CatBoost | 0.8101 | 0.8333 | 0.6757 | 0.7463 | 0.7902 | 0.8294 |
| XGBoost | 0.8045 | 0.7671 | 0.7568 | 0.7619 | 0.7974 | 0.8148 |
| Logistic Regression | 0.7989 | 0.7714 | 0.7297 | 0.7500 | 0.7887 | 0.7935 |

## Key Findings
- Sex was the most important feature — female passengers had significantly higher survival
- Pclass was second most important — 1st class passengers survived more
- Fare strongly correlated with survival — higher fare = higher survival
- Age and family size had minimal impact
- SHAP confirmed historical accuracy — women and upper class had highest survival rates

## Visualizations
- Confusion Matrix
- ROC Curve (AUC = 0.8889)
- SHAP Beeswarm Plot

## Tools
Python, scikit-learn, LightGBM, XGBoost, CatBoost, SHAP, pandas, matplotlib

## How to Run
```bash
pip install -r requirements.txt
python classification_titanic.py
python test_titanic.py
```

## Files
- `classification_titanic.py` — full pipeline
- `test_titanic.py` — test model with custom passenger
