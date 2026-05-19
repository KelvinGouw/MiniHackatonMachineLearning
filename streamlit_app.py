import os
import requests
import streamlit as st

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
    high_score = proba.get("High", 0) + proba.get("Medium", 0)
    low_score = proba.get("Low", 0)
    return "High" if high_score >= low_score else "Low"


def build_payload(values):
    payload = {}
    for feature in FEATURES:
        value = values.get(feature)
        if value is None or value == "":
            payload[feature] = None
            continue
        if feature in CATEGORICAL_COLS:
            payload[feature] = value
        else:
            payload[feature] = float(value)
    return payload


st.set_page_config(page_title="Student Burnout Predictor", page_icon="🧠")

st.title("Student Burnout Predictor")
st.caption("Masukkan data mahasiswa lalu dapatkan prediksi burnout.")

with st.form("burnout_form"):
    daily_study_hours = st.number_input("Daily Study Hours", min_value=0.0, step=0.1)
    daily_sleep_hours = st.number_input("Daily Sleep Hours", min_value=0.0, step=0.1)
    screen_time_hours = st.number_input("Screen Time Hours", min_value=0.0, step=0.1)
    physical_activity_hours = st.number_input("Physical Activity Hours", min_value=0.0, step=0.1)
    stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    anxiety_score = st.number_input("Anxiety Score", min_value=0.0, step=1.0)
    depression_score = st.number_input("Depression Score", min_value=0.0, step=1.0)
    academic_pressure_score = st.number_input("Academic Pressure Score", min_value=0.0, step=1.0)
    financial_stress_score = st.number_input("Financial Stress Score", min_value=0.0, step=1.0)
    social_support_score = st.number_input("Social Support Score", min_value=0.0, step=1.0)
    sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good"])
    attendance_percentage = st.number_input("Attendance %", min_value=0.0, step=0.1)

    submitted = st.form_submit_button("Predict")

if submitted:
    values = {
        "daily_study_hours": daily_study_hours,
        "daily_sleep_hours": daily_sleep_hours,
        "screen_time_hours": screen_time_hours,
        "physical_activity_hours": physical_activity_hours,
        "stress_level": stress_level,
        "anxiety_score": anxiety_score,
        "depression_score": depression_score,
        "academic_pressure_score": academic_pressure_score,
        "financial_stress_score": financial_stress_score,
        "social_support_score": social_support_score,
        "sleep_quality": sleep_quality,
        "attendance_percentage": attendance_percentage,
    }
    payload = build_payload(values)
    try:
        response = requests.post(MODEL_API_URL, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        pred = data.get("pred")
        proba = data.get("proba", {})
        result = to_binary_label(pred, proba)
        st.subheader("Prediction")
        st.success(result)
        st.write(get_recommendation(result))
    except requests.RequestException as exc:
        st.error(f"Gagal memanggil model service: {exc}")
