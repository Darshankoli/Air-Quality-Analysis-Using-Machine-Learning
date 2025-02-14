import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("air_quality_model.pkl")  # Ensure you have this saved from training
scaler = joblib.load("scaler.pkl")  # Load the scaler used for preprocessing

# Streamlit App UI
st.title("🌍 Air Quality Index (AQI) Prediction")
st.markdown("Enter pollutant levels to predict AQI and assess air quality.")

# User Input Fields
pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, step=0.1)
pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, step=0.1)
no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, step=0.1)
so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, step=0.1)
co = st.number_input("CO (mg/m³)", min_value=0.0, step=0.01)
o3 = st.number_input("O3 (µg/m³)", min_value=0.0, step=0.1)

# Prediction Function
def predict_aqi():
    features = np.array([[pm25, pm10, no2, so2, co, o3]])
    scaled_features = scaler.transform(features)  # Scale input data
    aqi = model.predict(scaled_features)[0]  # Predict AQI
    return round(aqi, 2)

# AQI Classification
def classify_aqi(aqi):
    if aqi <= 50:
        return "Good (Green)"
    elif aqi <= 100:
        return "Moderate (Yellow)"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups (Orange)"
    elif aqi <= 200:
        return "Unhealthy (Red)"
    elif aqi <= 300:
        return "Very Unhealthy (Purple)"
    else:
        return "Hazardous (Maroon)"

# Predict Button
if st.button("Predict AQI"):
    aqi = predict_aqi()
    category = classify_aqi(aqi)
    st.success(f"Predicted AQI: {aqi}")
    st.write(f"**Category:** {category}")
