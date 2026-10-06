from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "model", "fake_news_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR, "model", "tfidf_vectorizer.pkl"
)

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(VECTORIZER_PATH, "rb") as file:
    vectorizer = pickle.load(file)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Fake News Detection API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({
            "error": "Please provide news text"
        }), 400

    text = data["text"].strip()

    if not text:
        return jsonify({
            "error": "News text cannot be empty"
        }), 400

    transformed_text = vectorizer.transform([text])

    prediction = model.predict(transformed_text)[0]

    result = str(prediction)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed_text)[0]
        confidence = round(max(probabilities) * 100, 2)
    else:
        confidence = None

    return jsonify({
        "prediction": result,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)