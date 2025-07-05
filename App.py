import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_loader import load_openaq_data
from utils.ai_explainer import explain_trend
from ml.model import predict_pollution

st.set_page_config(page_title="Air Pollution from Space", layout="wide")

st.title("🛰️ Monitoring Air Pollution from Space")
st.markdown("Using Satellite, Ground, Reanalysis Data + AI/ML")

# Sidebar
location = st.sidebar.text_input("Enter a location (e.g., Delhi, Beijing)", "Delhi")
pollutant = st.sidebar.selectbox("Select Pollutant", ["PM2.5", "NO2", "O3"])

# Load and show data
st.subheader(f"📍 Air Pollution Data for {location}")
data = load_openaq_data(location, pollutant)

if data is not None:
    st.line_chart(data["value"])
    st.write(data.tail())

    # Forecasting
    if st.button("🔮 Predict Next 7 Days"):
        forecast = predict_pollution(data)
        st.line_chart(forecast)

    # AI Explanation
    if st.button("🧠 Explain This Trend (GPT)"):
        summary = explain_trend(data)
        st.success(summary)
else:
    st.error("Could not fetch data. Try a different location.")

st.markdown("---")
st.caption("Hackathon Project | Streamlit + OpenAI + Satellite + ML")
