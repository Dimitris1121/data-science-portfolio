import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# ---- DATA EXPLORATION ----

# Load dataset
df = pd.read_csv(r'D:\Δουλειά\Python\Apr26 Refresh ML\03.Fraud Detection\creditcard.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df['Class'].value_counts())
print("\nClass percentage:")
print(df['Class'].value_counts(normalize=True) * 100)

# Check for nulls
print("Null values:")
print(df.isnull().sum())

# Basic statistics
print("\nAmount statistics:")
print(df['Amount'].describe())

print("\nTime statistics:")
print(df['Time'].describe())


# ---- CLEANING ----

# Drop Time - not useful
df = df.drop('Time', axis=1)

# Scale Amount
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Split X and y
X = df.drop('Class', axis=1)
y = df['Class']

print("Cleaned shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ---- SPLIT ----

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Before SMOTE:")
print(f"Legitimate: {sum(y_train == 0)}")
print(f"Fraud:      {sum(y_train == 1)}")


# ---- SMOTE ----

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(f"Legitimate: {sum(y_train_sm == 0)}")
print(f"Fraud:      {sum(y_train_sm == 1)}")


# ---- LOGISTIC REGRESSION ----

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_sm, y_train_sm)
y_pred_lr = lr.predict(X_test)

print("---- Logistic Regression ----")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1:        {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_lr):.4f}")


# ---- RANDOM FOREST ----

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_sm, y_train_sm)
y_pred_rf = rf.predict(X_test)

print("\n---- Random Forest ----")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1:        {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_rf):.4f}")


# ---- XGBOOST ----

xgb = XGBClassifier(random_state=42)
xgb.fit(X_train_sm, y_train_sm)
y_pred_xgb = xgb.predict(X_test)

print("\n---- XGBoost ----")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_xgb):.4f}")
print(f"F1:        {f1_score(y_test, y_pred_xgb):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_xgb):.4f}")


# ---- CONFUSION MATRIX - RANDOM FOREST ----

cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Fraud'])

plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues')
plt.title('Random Forest - Confusion Matrix')
plt.show()


# ---- SAVE MODEL ----

joblib.dump(rf, 'best_rf_fraud.pkl')
print("✅ Model saved as best_rf_fraud.pkl")
