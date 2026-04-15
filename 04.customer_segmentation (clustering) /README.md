# 04 — Customer Segmentation Using RFM & K-Means Clustering

## Overview
Unsupervised machine learning project segmenting customers based on 
purchasing behavior using RFM analysis and K-Means clustering.
Directly applicable to gaming/lottery player segmentation in iGaming industry.

## Problem
Understanding who your customers are and how they behave is critical 
for targeted marketing, retention, and revenue growth. This project 
groups 4,338 customers into meaningful segments based on their 
purchasing patterns.

## Dataset
- **Source:** Online Retail Dataset (UCI ML Repository)
- **Size:** 541,909 transactions → 4,338 customers after RFM engineering
- **Period:** 12 months of UK e-commerce transactions

## Approach

### 1. Data Cleaning
- Removed transactions with missing CustomerID
- Removed returns and cancellations (negative quantities)
- Removed invalid prices

### 2. RFM Feature Engineering
Created 3 behavioral features per customer:
- **Recency** — days since last purchase
- **Frequency** — number of unique purchases
- **Monetary** — total amount spent

### 3. Clustering
- Scaled features with StandardScaler
- Used Elbow Method to find optimal K=4
- Applied K-Means clustering
- Evaluated with Silhouette Score

## Results

### Silhouette Score: 0.6162 (Reasonable clusters ✅)

### Customer Segments

| Segment | Customers | Avg Recency | Avg Frequency | Avg Monetary |
|---|---|---|---|---|
| 🌟 Champions | 13 | 7 days | 82 purchases | $127,338 |
| 🔥 High Value Actives | 204 | 15 days | 22 purchases | $12,709 |
| 💛 Loyal Customers | 3,054 | 44 days | 4 purchases | $1,359 |
| 💀 Lost Customers | 1,067 | 248 days | 2 purchases | $481 |

## Key Findings
- 13 Champions (0.3% of customers) generate estimated $1.6M+ revenue
- 70% of customers are Loyal but low frequency — upsell opportunity
- 25% of customers are Lost — targeted win-back campaigns needed
- High Value Actives are priority retention targets

## Business Recommendations
- **Champions** — VIP treatment, dedicated account manager
- **High Value Actives** — exclusive offers, retain at all costs
- **Loyal Customers** — loyalty program, upsell campaigns
- **Lost Customers** — win-back discount or deprioritize

## Tools
Python, scikit-learn, pandas, numpy, matplotlib, seaborn

## How to Run
```bash
pip install -r requirements.txt
python customer_segmentation.py
```

## Files
- `customer_segmentation.py` — full pipeline
- `customer_segments.csv` — final segmented customer table
- Dataset: Download from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/online+retail)
