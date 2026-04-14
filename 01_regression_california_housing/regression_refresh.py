import joblib
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from xgboost import XGBRegressor


# ---- LOAD DATA ----
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target

# ---- CONVERT TO READABLE NUMBERS ----
df['MedInc'] = df['MedInc'] * 10000
df['Price'] = df['Price'] * 100000

# ---- CLEAN OUTLIERS ----
df = df[df['AveOccup'] < 10]
df = df[df['AveRooms'] < 20]
df = df[df['AveBedrms'] < 10]

# ---- SPLIT INTO X AND y ----
X = df.drop('Price', axis=1)
y = df['Price']

# ---- TRAIN TEST SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data ready!")
print(f"Training rows: {X_train.shape[0]}")
print(f"Testing rows:  {X_test.shape[0]}")


# ---- MODELS ----
results = []

# -- Linear Regression --
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_cv = cross_val_score(lr_model, X, y, cv=5, scoring='r2').mean()
results.append({
    'Model': 'Linear Regression',
    'R²':    round(r2_score(y_test, lr_pred), 4),
    'MAE':   round(mean_absolute_error(y_test, lr_pred), 0),
    'RMSE':  round(np.sqrt(mean_squared_error(y_test, lr_pred)), 0),
    'CV R²': round(lr_cv, 4)
})
print("✅ Linear Regression done")

# -- Random Forest --
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_cv = cross_val_score(rf_model, X, y, cv=5, scoring='r2').mean()
results.append({
    'Model': 'Random Forest',
    'R²':    round(r2_score(y_test, rf_pred), 4),
    'MAE':   round(mean_absolute_error(y_test, rf_pred), 0),
    'RMSE':  round(np.sqrt(mean_squared_error(y_test, rf_pred)), 0),
    'CV R²': round(rf_cv, 4)
})
print("✅ Random Forest done")

# -- XGBoost --
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_cv = cross_val_score(xgb_model, X, y, cv=5, scoring='r2').mean()
results.append({
    'Model': 'XGBoost',
    'R²':    round(r2_score(y_test, xgb_pred), 4),
    'MAE':   round(mean_absolute_error(y_test, xgb_pred), 0),
    'RMSE':  round(np.sqrt(mean_squared_error(y_test, xgb_pred)), 0),
    'CV R²': round(xgb_cv, 4)
})
print("✅ XGBoost done")

# -- LightGBM --
lgbm_model = LGBMRegressor(random_state=42)
lgbm_model.fit(X_train, y_train)
lgbm_pred = lgbm_model.predict(X_test)
lgbm_cv = cross_val_score(lgbm_model, X, y, cv=5, scoring='r2').mean()
results.append({
    'Model': 'LightGBM',
    'R²':    round(r2_score(y_test, lgbm_pred), 4),
    'MAE':   round(mean_absolute_error(y_test, lgbm_pred), 0),
    'RMSE':  round(np.sqrt(mean_squared_error(y_test, lgbm_pred)), 0),
    'CV R²': round(lgbm_cv, 4)
})
print("✅ LightGBM done")

# -- CatBoost --
catboost_model = CatBoostRegressor(random_state=42, verbose=0)
catboost_model.fit(X_train, y_train)
catboost_pred = catboost_model.predict(X_test)
catboost_cv = cross_val_score(catboost_model, X, y, cv=5, scoring='r2').mean()
results.append({
    'Model': 'CatBoost',
    'R²':    round(r2_score(y_test, catboost_pred), 4),
    'MAE':   round(mean_absolute_error(y_test, catboost_pred), 0),
    'RMSE':  round(np.sqrt(mean_squared_error(y_test, catboost_pred)), 0),
    'CV R²': round(catboost_cv, 4)
})
print("✅ CatBoost done")


# ---- RESULTS TABLE ----
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('CV R²', ascending=False)
print("\n---- MODEL COMPARISON ----")
print(results_df.to_string(index=False))


# ---- SHAP ----
print("\nCalculating SHAP values...")

explainer = shap.Explainer(catboost_model)
shap_values = explainer(X_test)

print("Showing SHAP Summary Plot...")
shap.summary_plot(shap_values, X_test)


# ---- HYPERPARAMETER TUNING ----
print("\nTuning CatBoost...")

param_grid = {
    'iterations':    [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth':         [4, 6, 8],
    'l2_leaf_reg':   [1, 3, 5, 7]
}

tuned_catboost = CatBoostRegressor(random_state=42, verbose=0)

search = RandomizedSearchCV(
    tuned_catboost,
    param_grid,
    n_iter=20,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
print("Best parameters:", search.best_params_)

y_pred_tuned = best_model.predict(X_test)
r2_tuned   = r2_score(y_test, y_pred_tuned)
mae_tuned  = mean_absolute_error(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
cv_tuned   = cross_val_score(best_model, X, y, cv=5, scoring='r2').mean()

print("\n---- TUNED CATBOOST RESULTS ----")
print(f"R²:    {r2_tuned:.4f}  (was 0.8483)")
print(f"MAE:   ${mae_tuned:,.0f}")
print(f"RMSE:  ${rmse_tuned:,.0f}")
print(f"CV R²: {cv_tuned:.4f}  (was 0.7080)")


# ---- SAVE THE MODEL ----
joblib.dump(best_model, 'best_catboost_model.pkl')
print("\n✅ Model saved as best_catboost_model.pkl")
