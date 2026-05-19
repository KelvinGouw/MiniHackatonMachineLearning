import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request
import train_model

MODEL_PATH = os.path.join("models", "burnout_model.joblib")

app = Flask(__name__)


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train_model.train_and_save()


model_bundle = load_model()
model = model_bundle["model"]
features = model_bundle["features"]


def build_input(payload):
    data = {}
    for feature in features:
        value = payload.get(feature, None)
        if value is None or value == "":
            data[feature] = None
            continue
        if feature in model_bundle["numeric_cols"]:
            try:
                data[feature] = float(value)
            except ValueError:
                data[feature] = None
        else:
            data[feature] = value
    return pd.DataFrame([data])


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    input_df = build_input(payload)
    pred = model.predict(input_df)[0]
    proba = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        classes = model.classes_
        proba = {label: float(score) for label, score in zip(classes, probs)}
    return jsonify({"pred": pred, "proba": proba})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
