"""
Clinical Diabetes Risk Prediction
"""

import joblib
import pandas as pd

# Load model
model = joblib.load("models/random_forest.pkl")

scaler = joblib.load("models/scaler.pkl")

patient = pd.DataFrame({

    "Pregnancies":[2],

    "Glucose":[165],

    "BloodPressure":[85],

    "SkinThickness":[28],

    "Insulin":[150],

    "BMI":[33.2],

    "DiabetesPedigreeFunction":[0.58],

    "Age":[51],

    "Risk_Score":[4]

})

patient_scaled = scaler.transform(patient)

probability = model.predict_proba(patient_scaled)[0][1]

prediction = model.predict(patient_scaled)[0]

print("="*40)

print("Clinical Diabetes Risk Prediction")

print("="*40)

print(f"Predicted probability: {probability:.2%}")

print()

if prediction == 1:

    print("Risk Category: HIGH RISK")

    print("Recommendation:")

    print("- Further diabetic assessment")

    print("- HbA1c testing")

    print("- Lifestyle intervention")

else:

    print("Risk Category: LOW RISK")

    print("Recommendation:")

    print("- Routine follow-up")