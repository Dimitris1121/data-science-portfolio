# 05 — Revenue Forecasting with Prophet

## Overview
Time series forecasting project predicting daily revenue using Meta's Prophet library.
Built on real UK e-commerce transaction data with seasonal patterns and holiday effects.
Directly applicable to player activity and revenue forecasting in iGaming industry.

## Problem
Forecast future daily revenue based on historical sales patterns, accounting for
weekly seasonality, yearly trends, and holiday spikes (Black Friday, Christmas).

## Dataset
- **Source:** Online Retail Dataset (UCI ML Repository)
- **Size:** 541,909 transactions → 305 daily revenue observations
- **Period:** December 2010 — December 2011
- **Target:** Daily revenue (£)

## Key Observations from EDA
- Strong weekly seasonality — higher revenue on weekdays vs weekends
- Multiplicative seasonality — variance grows over time
- Quiet mid-year baseline (Feb-Aug): £20,000-£40,000/day
- Major spikes: Black Friday (£105,000), Christmas period (£184,000)
- 3 extreme outliers identified and removed before training

## Approach

### 1. Data Preparation
- Aggregated 541,909 transactions to daily revenue
- Removed cancellations and invalid records
- Removed 3 extreme outliers (>3 standard deviations from mean)

### 2. Model — Prophet
Used Meta's Prophet library with:
- **Yearly seasonality** — learns annual patterns
- **Weekly seasonality** — learns weekday vs weekend patterns
- **Multiplicative mode** — handles growing variance over time
- **Holiday regressors** — Black Friday, Cyber Monday, Christmas, New Year
- **changepoint_prior_scale=0.1** — stable trend, less overfitting

### 3. Forecast Horizon
- Original: 30 days
- Improved: 7 days — more realistic with one year of data

## Results

### Version 1 — Baseline
| Metric | Value |
|---|---|
| MAE | £21,457 |
| RMSE | £31,950 |
| MAPE | 48.91% ❌ |

### Version 2 — Improved (outlier removal + holidays + 7 day horizon)
| Metric | Value |
|---|---|
| MAE | £11,933 |
| RMSE | £15,441 |
| MAPE | 23.97% ✅ |

### Improvement
- MAE improved by **44%**
- RMSE improved by **52%**
- MAPE improved by **51%**

## Key Findings
- Outlier removal was the single biggest improvement — 3 extreme days were distorting all metrics
- Holiday regressors improved accuracy around Black Friday and Christmas
- 7 day horizon is significantly more accurate than 30 days with limited data
- With 2-3 years of data MAPE would likely drop below 10%

## Limitations
- Only one year of data — Prophet needs multiple years for reliable yearly seasonality
- Test period (Nov-Dec) is the most volatile period — hardest to forecast
- 24% MAPE is acceptable for a first version but needs more data to improve further

## Business Context
In iGaming (Allwyn use case) this approach would be used to:
- Forecast daily active players
- Predict revenue by game category
- Plan marketing spend around seasonal peaks
- Detect anomalies in player behavior

## Tools
Python, Prophet (Meta), pandas, numpy, matplotlib, scikit-learn

## How to Run
```bash
pip install -r requirements.txt
pip install prophet
python time_series_forecasting.py
```

## Files
- `time_series_forecasting.py` — full pipeline
- `revenue_forecast.csv` — forecast output with confidence intervals
- Dataset: Download from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/online+retail)
