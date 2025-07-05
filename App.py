import streamlit as st
import pandas as pd
import altair as alt

# Try importing your utility modules with error handling
try:
    from utils.data_loader import load_openaq_data
    from utils.ai_explainer import explain_trend
    from ml.model import predict_pollution
except ImportError as e:
    st.error(f"Error importing utility modules: {e}")
    st.stop()

# Set up the Streamlit page
st.set_page_config(page_title="Air Pollution from Space", layout="wide")
st.title("🛰️ Monitoring Air Pollution from Space")
st.markdown("Using Satellite, Ground, Reanalysis Data + AI/ML")

# Sidebar user input
location = st.sidebar.text_input("Enter a location (e.g., Delhi, Beijing)", "Delhi")
pollutant = st.sidebar.selectbox("Select Pollutant", ["PM2.5", "NO2", "O3"])

# Load pollution data
st.subheader(f"📍 Air Pollution Data for {location}")

with st.spinner("Loading data..."):
    data = load_openaq_data(location, pollutant)

# Validate and display data
if data is not None and not data.empty:
    if "date" not in data.columns or "value" not in data.columns:
        st.error("Expected columns 'date' and 'value' not found in data.")
    else:
        # Ensure datetime format and set index
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data = data.dropna(subset=['date'])  # Drop rows with invalid dates
        data = data.set_index('date')

        # Plot with Altair
        chart = alt.Chart(data.reset_index()).mark_line().encode(
            x='date:T',
            y='value:Q',
            tooltip=['date:T', 'value:Q']
        ).properties(
            title=f"{pollutant} Levels in {location}"
        )
        st.altair_chart(chart, use_container_width=True)

        st.write("📊 Latest Data")
        st.dataframe(data.tail())

        # Forecast Button
        if st.button("🔮 Predict Next 7 Days"):
            with st.spinner("Generating forecast..."):
                try:
                    forecast = predict_pollution(data)
                    if isinstance(forecast, pd.DataFrame) and not forecast.empty:
                        st.line_chart(forecast)
                    else:
                        st.warning("Forecast returned no data.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        # Explanation Button
        if st.button("🧠 Explain This Trend (GPT)"):
            with st.spinner("Generating explanation..."):
                try:
                    summary = explain_trend(data)
                    st.success(summary)
                except Exception as e:
                    st.error(f"Explanation failed: {e}")
else:
    st.error("❌ Could not fetch valid data. Try a different location or check your API source.")

# Footer
st.markdown("---")
st.caption("Hackathon Project | Streamlit + OpenAI + Satellite + ML")
