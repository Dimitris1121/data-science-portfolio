import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# =============================================================================
# LOAD DATA
# =============================================================================

# Load Titanic dataset directly from URL - no download needed
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# First look
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn info:")
print(df.info())

# =============================================================================
# DATA CLEANING
# =============================================================================

# Drop useless columns
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Fill missing Age with mean
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Encode Sex (male=1, female=0)
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

# Encode Embarked
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Verify no nulls remain
print("Null values after cleaning:")
print(df.isnull().sum())
print("\nShape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Check class balance
print("Survival distribution:")
print(df['Survived'].value_counts())
print("\nPercentage:")
print(df['Survived'].value_counts(normalize=True) * 100)

# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================

# Split into X and y
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# =============================================================================
# MODELS
# =============================================================================

# ---- LOGISTIC REGRESSION ----
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_lr)
precision = precision_score(y_test, y_pred_lr)
recall = recall_score(y_test, y_pred_lr)
f1 = f1_score(y_test, y_pred_lr)
roc_auc = roc_auc_score(y_test, y_pred_lr)
cv = cross_val_score(lr, X, y, cv=5, scoring='accuracy').mean()

print("\n---- Logistic Regression ----")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"CV Acc:    {cv:.4f}")

# ---- RANDOM FOREST ----
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n---- Random Forest ----")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1:        {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_rf):.4f}")
print(f"CV Acc:    {cross_val_score(rf, X, y, cv=5, scoring='accuracy').mean():.4f}")

# ---- XGBOOST ----
xgb = XGBClassifier(random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("\n---- XGBoost ----")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_xgb):.4f}")
print(f"F1:        {f1_score(y_test, y_pred_xgb):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_xgb):.4f}")
print(f"CV Acc:    {cross_val_score(xgb, X, y, cv=5, scoring='accuracy').mean():.4f}")

# ---- LIGHTGBM ----
lgbm = LGBMClassifier(random_state=42, verbose=-1)
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)

print("\n---- LightGBM ----")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_lgbm):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lgbm):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_lgbm):.4f}")
print(f"F1:        {f1_score(y_test, y_pred_lgbm):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_lgbm):.4f}")
print(f"CV Acc:    {cross_val_score(lgbm, X, y, cv=5, scoring='accuracy').mean():.4f}")

# ---- CATBOOST ----
cat = CatBoostClassifier(random_state=42, verbose=0)
cat.fit(X_train, y_train)
y_pred_cat = cat.predict(X_test)

print("\n---- CatBoost ----")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_cat):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_cat):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_cat):.4f}")
print(f"F1:        {f1_score(y_test, y_pred_cat):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_cat):.4f}")
print(f"CV Acc:    {cross_val_score(cat, X, y, cv=5, scoring='accuracy').mean():.4f}")

# =============================================================================
# CONFUSION MATRIX
# =============================================================================

cm = confusion_matrix(y_test, y_pred_lgbm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Died', 'Survived'])

plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues')
plt.title('LightGBM - Confusion Matrix')
plt.show()

# =============================================================================
# ROC CURVE
# =============================================================================

y_prob_lgbm = lgbm.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_lgbm)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'LightGBM (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# =============================================================================
# SHAP
# =============================================================================

print("Calculating SHAP values...")
explainer = shap.Explainer(lgbm)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)

# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

print("\nTuning LightGBM...")
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'num_leaves': [20, 31, 50]
}

tuned_lgbm = LGBMClassifier(random_state=42, verbose=-1)

search = RandomizedSearchCV(
    tuned_lgbm,
    param_grid,
    n_iter=20,
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
print("Best parameters:", search.best_params_)

y_pred_tuned = best_model.predict(X_test)
print("\n---- TUNED LIGHTGBM ----")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_tuned):.4f}  (was 0.8268)")
print(f"Precision: {precision_score(y_test, y_pred_tuned):.4f}  (was 0.7945)")
print(f"Recall:    {recall_score(y_test, y_pred_tuned):.4f}  (was 0.7838)")
print(f"F1:        {f1_score(y_test, y_pred_tuned):.4f}  (was 0.7891)")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_tuned):.4f}  (was 0.8205)")
cv_tuned = cross_val_score(best_model, X, y, cv=5, scoring='accuracy').mean()
print(f"CV Acc:    {cv_tuned:.4f}  (was 0.8272)")

# =============================================================================
# SAVE MODEL
# =============================================================================

joblib.dump(best_model, 'best_lgbm_titanic.pkl')
print("\n✅ Model saved as best_lgbm_titanic.pkl")
