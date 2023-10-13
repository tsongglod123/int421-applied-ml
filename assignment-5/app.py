from components.converter import convert_input
from components.converter import sex_types
from components.converter import cp_types

import streamlit as st
import numpy as np
import joblib

MODEL_PATH = "./models/model-knn-tuned.pkl"

model = joblib.load(MODEL_PATH)

def predict_heart_disease(input_data: dict) -> int:
    model_input = convert_input(input_data)
    # code here
    return np.random.choice([0, 1])

st.set_page_config(page_title = "Heart Disease Prediction App", page_icon = "❤️")

st.title("Heart Disease Prediction")
st.markdown("This app predicts the **Heart Disease**")

st.sidebar.header("User Input Parameters")

age: int = st.sidebar.number_input("Age", 1, value = 25, step = 1, placeholder = "Enter your age")
gender: str = st.sidebar.selectbox("Gender", sex_types.keys())
chest_pain: str = st.sidebar.selectbox("Chest Pain Type", cp_types.keys())
resting_bp: int = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
cholesterol: int = st.sidebar.slider("Cholesterol Level (mg/dl)", 100, 400, 200)
max_heart_rate: int = st.sidebar.slider("Maximum Heart Rate", 60, 220, 150)
exercise_angina: str = st.sidebar.radio("Exercise-Induced Angina", ["Yes", "No"])

user_input = {
    "age": age,
    "gender": gender,
    "chest_pain": chest_pain,
    "resting_bp": resting_bp,
    "cholesterol": cholesterol,
    "max_heart_rate": max_heart_rate,
    "exercise_angina": exercise_angina
}

if st.button("Predict"):
    result = predict_heart_disease(user_input)
    st.write("### Prediction Result")
    if result == 0:
        st.write("No Heart Disease")
    else:
        st.write("Heart Disease Detected")
