from components.converter import convert_input
from components.converter import sex_types
from components.converter import cp_types
from components.converter import resting_ecg_types
from components.converter import slope_types
from components.converter import thal_types

import streamlit as st
import pandas as pd
import numpy as np
import joblib

MODEL_PATH = "./models/model-knn-tuned.pkl"

model = joblib.load(MODEL_PATH)

def predict_heart_disease(input_data: dict) -> int:
    model_input = convert_input(input_data)
    X = pd.DataFrame(model_input, index = [0])
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    return y_pred[0], np.round(y_prob[0][1], 4) * 100

st.set_page_config(page_title = "Heart Disease Prediction App", page_icon = "❤️")

st.title("Heart Disease Prediction")
st.markdown("This app predicts the **Heart Disease**")

st.sidebar.header("User Input Parameters")

age: int = st.sidebar.number_input("Age", min_value = 1, value = 25, step = 1, placeholder = "Enter your age")
sex: str = st.sidebar.selectbox("Gender", sex_types.keys())
chest_pain: str = st.sidebar.selectbox("Chest Pain Type", cp_types.keys())
resting_bps: int = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
cholesterol: int = st.sidebar.slider("Cholesterol Level (mg/dl)", 100, 600, 200)
fasting_blood_sugar: str = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
resting_ecg: str = st.sidebar.radio("Resting ECG Result", resting_ecg_types.keys())
max_heart_rate: int = st.sidebar.slider("Maximum Heart Rate", 60, 250, 150)
exercise_angina: str = st.sidebar.radio("Exercise Induced Angina", ["Yes", "No"])
st_depression: float = st.sidebar.number_input(
    label = "ST Depression",
    min_value = 0.0,
    value = 2.5,
    step = 0.1,
    format = "%.1f",
    help = "ST depression induced by exercise relative to rest"
)
st_slope: str = st.sidebar.selectbox("ST Slope", slope_types.keys())
num_major_vessels: int = st.sidebar.number_input("Number of Major Vessels", 0, 3, 0, 1)
thalassemia: str = st.sidebar.selectbox("Thalassemia", thal_types.keys())

user_input = {
    "age": age,
    "sex": sex,
    "cp": chest_pain,
    "trestbps": resting_bps,
    "chol": cholesterol,
    "fbs": fasting_blood_sugar,
    "restecg": resting_ecg,
    "thalach": max_heart_rate,
    "exang": exercise_angina,
    "oldpeak": st_depression,
    "slope": st_slope,
    "ca": num_major_vessels,
    "thal": thalassemia
}

if st.button("Predict"):
    result, result_prob = predict_heart_disease(user_input)
    st.write("### Prediction Result")
    st.write(f"Probability of Heart Disease: {result_prob}%")
    if result == 0:
        st.write("No Heart Disease")
    else:
        st.write("Heart Disease Detected")
