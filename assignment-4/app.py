from reg_score import get_reg_score

import streamlit as st
import pandas as pd
import numpy as np
import joblib

MODEL_PATH = "models/house-price-regression.pkl"
TRAIN_PATH = "datasets/train.csv"

loaded_model = joblib.load(MODEL_PATH)

df_train = pd.read_csv(TRAIN_PATH)
df_train = df_train.drop("Id", axis = 1)

r2_score = np.round(get_reg_score(df_train), 4)

st.title("House Price Prediction")
st.subheader(f"R^2 score: {r2_score}")

mszoning_options = {
    "Agriculture": "A",
    "Commercial": "C",
    "Floating Village Residential": "FV",
    "Industrial": "I",
    "Residential High Density": "RH",
    "Residential Low Density": "RL",
    "Residential Low Density Park": "RP",
    "Residential Medium Density": "RM",
}

# Input features
overall_cond = st.slider("Overall Condition (1-10)", min_value = 1, max_value = 10, value = 5)
overall_qual = st.slider("Overall Quality (1-10)", min_value = 1, max_value = 10, value = 5)
grlivarea = st.number_input("Living Area (sqft)", min_value = 0.0, value = 1500.0)
mszoning = st.selectbox("Zoning", list(mszoning_options.keys()))
mszoning_value = mszoning_options.get(mszoning)

if st.button("Predict"):
    input_data = pd.DataFrame({
        "OverallCond": [str(overall_cond)],
        "OverallQual": [str(overall_qual)],
        "GrLivArea": [grlivarea],
        "MSZoning": [mszoning_value],
    })
    prediction = loaded_model.predict(input_data)

    st.write("### Predicted Price")
    st.write(f"Estimated price for the property is: **${prediction[0]: ,.2f}**")
