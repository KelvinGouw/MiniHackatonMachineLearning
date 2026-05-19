import os
import requests
from flask import Flask, render_template, request

MODEL_API_URL = os.getenv("MODEL_API_URL", "http://localhost:5001/predict")

FEATURES = [
    "daily_study_hours",
    "daily_sleep_hours",
    "screen_time_hours",
    "physical_activity_hours",
    "stress_level",
    "anxiety_score",
    "depression_score",
    "academic_pressure_score",
    "financial_stress_score",
    "social_support_score",
    "sleep_quality",
    "attendance_percentage",
]

CATEGORICAL_COLS = ["stress_level", "sleep_quality"]

app = Flask(__name__)


def build_payload(form):
    data = {}
    for feature in FEATURES:
        value = form.get(feature, "")
        if value is None or value == "":
            data[feature] = None
            continue
        if feature in CATEGORICAL_COLS:
            data[feature] = value
        else:
            try:
                data[feature] = float(value)
            except ValueError:
                data[feature] = None
    return data


def get_recommendation(label):
    if label == "High":
        return (
            "Prioritaskan tidur cukup, kurangi beban tugas sementara, dan coba konsultasi dengan"
            " konselor kampus jika memungkinkan."
        )
    if label == "Medium":
        return (
            "Atur jadwal istirahat, batasi screen time malam hari, dan sisihkan waktu"
            " olahraga ringan 3-4x seminggu."
        )
    if label == "Low":
        return "Pertahankan kebiasaan baik, jaga rutinitas tidur, dan lakukan aktivitas fisik teratur."
    return "Jaga pola tidur dan aktivitas seimbang, serta cari dukungan sosial saat diperlukan."


def to_binary_label(pred, proba):
    if not proba:
        return "High" if pred == "High" else "Low"
    scores = dict(proba)
    high_score = scores.get("High", 0) + scores.get("Medium", 0)
    low_score = scores.get("Low", 0)
    return "High" if high_score >= low_score else "Low"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, recommendation=None)


@app.route("/predict", methods=["POST"])
def predict():
    payload = build_payload(request.form)
    response = requests.post(MODEL_API_URL, json=payload, timeout=10)
    response.raise_for_status()
    data = response.json()
    pred = data.get("pred")
    proba = data.get("proba", {})
    binary_label = to_binary_label(pred, proba)
    recommendation = get_recommendation(binary_label)
    return render_template("index.html", result=binary_label, proba=None, recommendation=recommendation)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
