from flask import Flask, render_template, request
import firebase_admin
from firebase_admin import credentials, db
import joblib
import pandas as pd

app = Flask(__name__)

# Firebase initialization
cred = credentials.Certificate("firebase_config.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://fypproject-1058c-default-rtdb.firebaseio.com/'
})

# Load reverse model and label encoder
reverse_model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict')
def show_prediction():
    ref = db.reference("/PredictionResult")
    data = ref.get()

    if data:
        sensor_data = {
            'Nitrogen': data.get('Nitrogen', 0),
            'Phosphorus': data.get('Phosphorus', 0),
            'Potassium': data.get('Potassium', 0),
            'Temperature': data.get('Temperature', 0),
            'Humidity': data.get('Humidity', 0),
            'PH': data.get('PH', 0)
        }
        top_crops = [
            {'crop': data.get('Crop_1', 'Unknown'), 'probability': data.get('Score_1', 0)},
            {'crop': data.get('Crop_2', 'Unknown'), 'probability': data.get('Score_2', 0)},
            {'crop': data.get('Crop_3', 'Unknown'), 'probability': data.get('Score_3', 0)},
            {'crop': data.get('Crop_4', 'Unknown'), 'probability': data.get('Score_4', 0)},
            {'crop': data.get('Crop_5', 'Unknown'), 'probability': data.get('Score_5', 0)}
        ]
        return render_template("dashboard.html", sensor=sensor_data, top_crops=top_crops)
    else:
        return "<h3>No prediction data available.</h3>"

@app.route('/reverse', methods=["GET", "POST"])
def reverse_predict():
    if request.method == "POST":
        crop_name = request.form["crop"]
        crop_encoded = label_encoder.transform([crop_name])[0]
        predicted_features = reverse_model.predict([[crop_encoded]])[0]

        features = {
            'Nitrogen': round(predicted_features[0], 2),
            'Phosphorus': round(predicted_features[1], 2),
            'Potassium': round(predicted_features[2], 2),
            'Temperature': round(predicted_features[3], 2),
            'Humidity': round(predicted_features[4], 2),
            'pH': round(predicted_features[5], 2),
        }

        return render_template("reverse_result.html", crop=crop_name, features=features)

    return render_template("reverse_form.html")

if __name__ == '__main__':
    app.run(debug=True)