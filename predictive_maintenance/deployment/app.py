import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="nairsuj/predictive-maintenance", filename="predictive_maintenance_model.joblib")
model = joblib.load(model_path)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Maintaince Prediction App")

st.write("""
This application predicts when an engine requires maintenance by analyzing engine health parameters such as RPM, temperature, pressure, and other sensor readings.
Please enter **Engine Parameters** below to get a prediction.
""")

# ------------------------------
# User Inputs
# ------------------------------
st.subheader("Engine Parameters")

engine_rpm = st.number_input("Engine RPM", min_value=0, max_value=3000, value=700)
lub_oil_pressure = st.number_input("Lub Oil Pressure (kPa)", min_value=0, max_value=3000, value=1)
fuel_pressure = st.number_input("Fuel Pressure (kPa)", min_value=0, max_value=3000, value=11)
coolant_pressure = st.number_input("Coolant Pressure (kPa)", min_value=0, max_value=3000, value=3)
lub_oil_temperature = st.number_input("Lub Oil Temperature (°C)", min_value=0, max_value=3000, value=84)
coolant_temperature = st.number_input("Coolant Temperature (°C)", min_value=0, max_value=3000, value=81)

# ------------------------------
# Prepare Input for Prediction
# ------------------------------
input_data = {
    "Engine rpm": engine_rpm,
    "Lub oil pressure": lub_oil_pressure,
    "Fuel pressure": fuel_pressure,
    "Coolant pressure": coolant_pressure,
    "lub oil temp": lub_oil_temperature,
    "Coolant temp": coolant_temperature
}

input_df = pd.DataFrame([input_data])

# Set the classification threshold
classification_threshold = 0.45

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict"):
    probability = model.predict_proba(input_df)[0][1]
    #prediction = model.predict(input_df)[0]

    prediction = (probability >= classification_threshold).astype(int)

    if prediction == 1:
        st.success(f"❌ This engine is **likely to** fail.")
    else:
        st.error(f"✅ This engine is **unlikely to** fail and will perform normal operation.")
