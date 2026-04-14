import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load saved model
model = joblib.load('best_rf_fraud.pkl')

# Create a fake transaction
transaction = pd.DataFrame({
    'V1': [-1.359807], 'V2': [-0.072781], 'V3': [2.536347],
    'V4': [1.378155], 'V5': [-0.338321], 'V6': [0.462388],
    'V7': [0.239599], 'V8': [0.098698], 'V9': [0.363787],
    'V10': [0.090794], 'V11': [-0.551600], 'V12': [-0.617801],
    'V13': [-0.991390], 'V14': [-0.311169], 'V15': [1.468177],
    'V16': [-0.470401], 'V17': [0.207971], 'V18': [0.025791],
    'V19': [0.403993], 'V20': [0.251412], 'V21': [-0.018307],
    'V22': [0.277838], 'V23': [-0.110474], 'V24': [0.066928],
    'V25': [0.128539], 'V26': [-0.189115], 'V27': [0.133558],
    'V28': [-0.021053], 'Amount': [0.244964]  # already scaled
})

# Predict
prediction = model.predict(transaction)[0]
probability = model.predict_proba(transaction)[0]

print("\n---- Transaction Check ----")
print(f"Prediction:            {'🚨 FRAUD' if prediction == 1 else '✅ LEGITIMATE'}")
print(f"Fraud probability:     {probability[1]*100:.1f}%")
print(f"Legitimate probability:{probability[0]*100:.1f}%")