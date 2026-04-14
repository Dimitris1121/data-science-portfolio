# 01 — California Housing Price Prediction

## Overview
End-to-end regression project predicting California house prices
using machine learning. Built as part of my ML learning journey
to refresh and expand on regression techniques from my MSc dissertation.

## Problem
Predict the median house price of a California block given features
like location, income level, house age, and occupancy.

## Dataset
- **Source:** California Housing Dataset (scikit-learn built-in)
- **Size:** 20,535 rows × 8 features after cleaning
- **Target:** Median house price in USD

## Models Compared

| Model | R² | MAE | RMSE | CV R² |
|---|---|---|---|---|
| CatBoost (tuned) ⭐ | 0.8541 | $27,949 | $44,033 | 0.6991 |
| LightGBM | 0.8344 | $30,864 | $46,918 | 0.6882 |
| XGBoost | 0.8273 | $30,721 | $47,909 | 0.6547 |
| Random Forest | 0.8029 | $33,048 | $51,185 | 0.6518 |
| Linear Regression | 0.6373 | $50,003 | $69,437 | 0.6097 |

## Key Findings
- Location (Latitude/Longitude) is the most important feature
- MedInc is the strongest single linear predictor (correlation 0.69)
- AveOccup negatively impacts price
- Boosting models significantly outperform Linear Regression
- SHAP revealed geographic patterns that correlation missed

## Tools
Python, scikit-learn, CatBoost, XGBoost, LightGBM, SHAP, pandas, matplotlib

## How to Run
pip install -r requirements.txt
python regression_refresh.py
python test_model.py
