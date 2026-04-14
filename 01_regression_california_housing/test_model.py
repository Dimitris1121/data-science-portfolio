import pandas as pd
import joblib

# Load the saved model
model = joblib.load('best_catboost_model.pkl')

# Create a fake house
test_house = pd.DataFrame({
    'MedInc': [75000],
    'HouseAge': [10],
    'AveRooms': [6],
    'AveBedrms': [3],
    'Population': [1500],
    'AveOccup': [3],
    'Latitude': [37.77],
    'Longitude': [-122.41]
})

# Predict
predicted_price = model.predict(test_house)[0]
print(f"\nPredicted house price: ${predicted_price:,.0f}")