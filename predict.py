import joblib
import numpy as np

# Load model
model = joblib.load("../model.pkl")

def get_risk_category(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

# Example input (Modify according to dataset features)
sample_data = np.array([[25, 22.5, 1, 0, 1, 98, 15]])

probability = model.predict_proba(sample_data)[0][1]
risk_category = get_risk_category(probability)

print("PCOS Probability:", round(probability * 100, 2), "%")
print("Risk Level:", risk_category)
