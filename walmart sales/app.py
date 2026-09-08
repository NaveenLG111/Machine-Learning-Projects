import os
import ssl

# 1. Hotfix: Overwrite SSLContext initialization to bypass corrupted local root cert stores
def empty_load_certs(*args, **kwargs):
    pass
ssl.SSLContext.load_default_certs = empty_load_certs

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 2. Safely load pre-trained artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('walmart_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception:
        st.error("Error: Missing pipeline artifacts. Please run your model training file first to generate 'walmart_model.pkl' and 'scaler.pkl'.")
        return None, None

model, scaler = load_artifacts()

# 3. Application Layout Interface
st.set_page_config(page_title="Walmart Predictive Analytics", layout="centered")
st.title("🛒 Walmart Weekly Sales Prediction Dashboard")
st.write("Provide the store parameters below to generate machine learning baseline forecasts.")

# Split user input form fields across columns
col1, col2 = st.columns(2)

with col1:
    store = st.number_input("Store ID", min_value=1, max_value=45, value=1, step=1)
    holiday = st.selectbox("Holiday Week?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    temp = st.slider("Temperature (°F)", 0.0, 110.0, 60.0)
    fuel = st.slider("Fuel Price ($/Gal)", 2.0, 5.5, 3.3)

with col2:
    cpi = st.number_input("Consumer Price Index (CPI)", min_value=100.0, max_value=250.0, value=175.0)
    unemp = st.slider("Unemployment Rate (%)", 3.0, 15.0, 7.0)
    month = st.slider("Month", 1, 12, 6)
    year = st.selectbox("Year", [2010, 2011, 2012])

# 4. Predict Feature Array Logic
if st.button("🔮 Forecast Weekly Sales", use_container_width=True):
    if model is not None and scaler is not None:
        raw_features = pd.DataFrame(
            [[store, holiday, temp, fuel, cpi, unemp, month, year]], 
            columns=['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Month', 'Year']
        )
        scaled_features = scaler.transform(raw_features)
        prediction = model.predict(scaled_features)[0]
        
        st.success(f"### Estimated Weekly Sales Target: **${prediction:,.2f}**")