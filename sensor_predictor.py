import firebase_admin
from firebase_admin import credentials, db
import joblib
import pandas as pd
import time
import numpy as np

# Load Firebase credentials
cred = credentials.Certificate("firebase_config.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://fypproject-1058c-default-rtdb.firebaseio.com/'
})

# Load ML model
model = joblib.load("crop_recommendation_best_model.pkl")
# Reference to original sensor data
ref = db.reference("/SoilData")
# Reference to push prediction result
prediction_ref = db.reference("/PredictionResult")

def fetch_and_predict():
    data = ref.get()
    print("Fetched Sensor Data:", data)

    if data:
        # Map Firebase keys (capitalized) to model feature names (lowercase)
        input_dict = {
            'Nitrogen': data.get("Nitrogen", 0),
            'phosphorus': data.get("Phosphorus", 0),
            'potassium': data.get("Potassium", 0),
            'temperature': data.get("Temperature", 0),
            'humidity': data.get("Humidity", 0),
            'ph': data.get("PH", 0)
        }

        # Create DataFrame for prediction
        input_df = pd.DataFrame([input_dict])
        
        # Get prediction probabilities
        probas = model.predict_proba(input_df)[0]
        # Get class labels
        classes = model.classes_
        # Get top 5 crops with their probabilities
        top_indices = np.argsort(probas)[-5:][::-1]  # Top 5 indices in descending order
        
        # Prepare data to send to Firebase in the desired structure
        output_dict = {
            'Nitrogen': data.get("Nitrogen", 0),
            'Phosphorus': data.get("Phosphorus", 0),
            'Potassium': data.get("Potassium", 0),
            'Temperature': data.get("Temperature", 0),
            'Humidity': data.get("Humidity", 0),
            'PH': data.get("PH", 0),
            'PredictedCrop': classes[top_indices[0]],  # Top crop
            'Crop_1': classes[top_indices[0]],
            'Score_1': float(round(probas[top_indices[0]] * 100, 2)),  # Convert to Python float
            'Crop_2': classes[top_indices[1]],
            'Score_2': float(round(probas[top_indices[1]] * 100, 2)),
            'Crop_3': classes[top_indices[2]],
            'Score_3': float(round(probas[top_indices[2]] * 100, 2)),
            'Crop_4': classes[top_indices[3]],
            'Score_4': float(round(probas[top_indices[3]] * 100, 2)),
            'Crop_5': classes[top_indices[4]],
            'Score_5': float(round(probas[top_indices[4]] * 100, 2))
        }
        prediction_ref.set(output_dict)

        print("✅ Top 5 Predictions Sent to Firebase:", output_dict)

if __name__ == "__main__":
    while True:
        fetch_and_predict()
        time.sleep(5)  # Fetch every 5 seconds