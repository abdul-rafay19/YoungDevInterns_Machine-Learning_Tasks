from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('wine_quality_model.pkl')

@app.route('/')
def home():
    return "✅ Flask app is running successfully!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features from the received JSON
        features = np.array([[
            data['fixed_acidity'],
            data['volatile_acidity'],
            data['citric_acid'],
            data['residual_sugar'],
            data['chlorides'],
            data['free_sulfur_dioxide'],
            data['total_sulfur_dioxide'],
            data['density'],
            data['pH'],
            data['sulphates'],
            data['alcohol']
        ]])

        # Make prediction
        prediction = model.predict(features)

        return jsonify({'predicted_quality': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
