import pandas as pd
import joblib

# Load the saved model
model = joblib.load('best_lgbm_titanic.pkl')

# Create a fake passenger
# Let's try a 25 year old female, 1st class, paid high fare
passenger = pd.DataFrame({
    'Pclass': [1],        # 1st class
    'Sex': [0],           # female
    'Age': [25],          # 25 years old
    'SibSp': [0],         # no siblings/spouse
    'Parch': [0],         # no parents/children
    'Fare': [100],        # high fare
    'Embarked': [1]       # boarded at Cherbourg
})

# Predict
prediction = model.predict(passenger)[0]
probability = model.predict_proba(passenger)[0]

print("\n---- Passenger Details ----")
print("Class: 1st | Sex: Female | Age: 25")
print(f"\nPrediction: {'SURVIVED ✅' if prediction == 1 else 'DIED ❌'}")
print(f"Survival probability: {probability[1]*100:.1f}%")
print(f"Death probability:    {probability[0]*100:.1f}%")