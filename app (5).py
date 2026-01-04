
import streamlit as st
import pandas as pd
import joblib
import json

model = joblib.load("adherence_model.pkl")
features = joblib.load("features.pkl")

with open("medicine_codes.json") as f:
    medicine_map = json.load(f)

st.title("Patient Adherence Risk Prediction App")

inputs = {}

for feature in features:
    if feature == "Medicine_Name":
        selected = st.selectbox(
            "Medicine Name",
            [f"{name} ({code})" for name, code in medicine_map.items()]
        )
        inputs[feature] = int(selected.split("(")[1].replace(")", ""))
    else:
        inputs[feature] = st.number_input(feature, value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([inputs])
    result = model.predict(input_df)[0]
    st.success(f"Predicted Adherence Risk Score: {round(result, 2)}")
