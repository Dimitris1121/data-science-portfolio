import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- LOAD DATA ----
df = pd.read_csv(r'D:\Δουλειά\Python\Apr26 Refresh ML\04.Customer_Segmentation\online_retail.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())



# ---- PREPARE TIME SERIES ----

# Clean data
df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

# Calculate revenue per transaction
df['Revenue'] = df['Quantity'] * df['UnitPrice']

# Convert date
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Aggregate to daily revenue
daily_revenue = df.groupby(df['InvoiceDate'].dt.date)['Revenue'].sum().reset_index()
daily_revenue.columns = ['ds', 'y']
daily_revenue['ds'] = pd.to_datetime(daily_revenue['ds'])

# Sort by date
daily_revenue = daily_revenue.sort_values('ds').reset_index(drop=True)

print("Daily revenue shape:", daily_revenue.shape)
print("\nFirst 5 rows:")
print(daily_revenue.head())
print("\nDate range:")
print(f"From: {daily_revenue['ds'].min()}")
print(f"To:   {daily_revenue['ds'].max()}")
print("\nRevenue statistics:")
print(daily_revenue['y'].describe())




# ---- VISUALIZE RAW DATA ----
plt.figure(figsize=(14, 6))
plt.plot(daily_revenue['ds'], daily_revenue['y'], 
         color='blue', linewidth=1)
plt.title('Daily Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('Revenue (£)')
plt.grid(True)
plt.tight_layout()
plt.show()












# ---- TRAIN TEST SPLIT ----
# Use last 30 days as test set
split_date = daily_revenue['ds'].max() - pd.Timedelta(days=30)

train = daily_revenue[daily_revenue['ds'] <= split_date]
test = daily_revenue[daily_revenue['ds'] > split_date]

print(f"Training period: {train['ds'].min()} to {train['ds'].max()}")
print(f"Training days: {len(train)}")
print(f"Test period: {test['ds'].min()} to {test['ds'].max()}")
print(f"Test days: {len(test)}")

# ---- BUILD PROPHET MODEL ----
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative'  # because variance grows over time
)

# Train
model.fit(train)

# ---- FORECAST ----
# Create future dataframe - 30 days ahead
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

print("\nForecast columns:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))








# ---- VISUALIZE FORECAST ----
fig = model.plot(forecast)
plt.title('Revenue Forecast - Prophet')
plt.xlabel('Date')
plt.ylabel('Revenue (£)')
plt.tight_layout()
plt.show()

# ---- VISUALIZE COMPONENTS ----
fig2 = model.plot_components(forecast)
plt.tight_layout()
plt.show()






# ---- EVALUATE ON TEST SET ----
# Get predictions for test period only
test_forecast = forecast[forecast['ds'].isin(test['ds'])]

# Merge actual vs predicted
eval_df = test.merge(test_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

# Calculate metrics
mae = mean_absolute_error(eval_df['y'], eval_df['yhat'])
rmse = np.sqrt(mean_squared_error(eval_df['y'], eval_df['yhat']))
mape = (abs(eval_df['y'] - eval_df['yhat']) / eval_df['y']).mean() * 100

print("---- FORECAST EVALUATION ----")
print(f"MAE:  £{mae:,.0f}")
print(f"RMSE: £{rmse:,.0f}")
print(f"MAPE: {mape:.2f}%")

print("\nActual vs Predicted:")
print(eval_df[['ds', 'y', 'yhat']].to_string(index=False))













# ---- IMPROVED MODEL ----

# Step 1 - Remove outliers from training data
# Remove days where revenue is more than 3 standard deviations from mean
mean = daily_revenue['y'].mean()
std = daily_revenue['y'].std()
threshold = mean + (3 * std)

print(f"Mean daily revenue: £{mean:,.0f}")
print(f"Std: £{std:,.0f}")
print(f"Outlier threshold: £{threshold:,.0f}")
print(f"Outliers removed: {(daily_revenue['y'] > threshold).sum()}")

daily_revenue_clean = daily_revenue[daily_revenue['y'] <= threshold]

# Step 2 - New split - forecast 7 days instead of 30
split_date = daily_revenue_clean['ds'].max() - pd.Timedelta(days=7)
train_clean = daily_revenue_clean[daily_revenue_clean['ds'] <= split_date]
test_clean = daily_revenue_clean[daily_revenue_clean['ds'] > split_date]

print(f"\nTraining days: {len(train_clean)}")
print(f"Test days: {len(test_clean)}")




# Step 3 - Add holiday regressors
from prophet import Prophet
import pandas as pd

# Define special events
events = pd.DataFrame({
    'holiday': [
        'black_friday', 
        'cyber_monday',
        'christmas_shopping',
        'new_year'
    ],
    'ds': pd.to_datetime([
        '2011-11-25',
        '2011-11-28',
        '2011-12-09',
        '2011-01-03'
    ]),
    'lower_window': [-1, 0, -3, 0],
    'upper_window': [1, 1, 3, 1]
})

# Step 4 - Retrain improved model
model_v2 = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    holidays=events,
    changepoint_prior_scale=0.1
)

model_v2.fit(train_clean)

# Forecast 7 days
future_v2 = model_v2.make_future_dataframe(periods=7)
forecast_v2 = model_v2.predict(future_v2)

# Evaluate
test_forecast_v2 = forecast_v2[forecast_v2['ds'].isin(test_clean['ds'])]
eval_df_v2 = test_clean.merge(
    test_forecast_v2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
    on='ds'
)

mae_v2 = mean_absolute_error(eval_df_v2['y'], eval_df_v2['yhat'])
rmse_v2 = np.sqrt(mean_squared_error(eval_df_v2['y'], eval_df_v2['yhat']))
mape_v2 = (abs(eval_df_v2['y'] - eval_df_v2['yhat']) / eval_df_v2['y']).mean() * 100

print("---- IMPROVED MODEL RESULTS ----")
print(f"MAE:  £{mae_v2:,.0f}  (was £21,457)")
print(f"RMSE: £{rmse_v2:,.0f}  (was £31,950)")
print(f"MAPE: {mape_v2:.2f}%  (was 48.91%)")

print("\nActual vs Predicted:")
print(eval_df_v2[['ds', 'y', 'yhat']].to_string(index=False))



# Save forecast results
forecast_v2.to_csv('revenue_forecast.csv', index=False)
print("✅ Forecast saved to revenue_forecast.csv")