import sys
from pathlib import Path

# Add project root to Python path (MUST BE FIRST)
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hopsworks
import plotly.graph_objects as go
import shap
from utils.config import Config
from models.model_registry import ModelRegistry

# Streamlit page config
st.set_page_config(
    page_title=Config.DASHBOARD_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hopsworks connection
try:
    project = hopsworks.login(
        project=Config.HOPSWORKS_PROJECT_NAME,
        api_key_value=Config.HOPSWORKS_API_KEY,
        host=Config.HOPSWORKS_HOST,
        port=443
    )
    feature_store = project.get_feature_store()
    st.success("✅ Connected to Hopsworks")
except Exception as e:
    st.error(f"❌ Hopsworks connection failed: {str(e)}")
    st.stop()

# Initialize components
model_registry = ModelRegistry()

# Session states
if 'current_data' not in st.session_state:
    st.session_state.current_data = {}
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = {}

def load_cities():
    """Load cities from feature store with YAML fallback"""
    try:
        feature_view = feature_store.get_feature_view(name="karachi_aqi_features", version=1)
        cities = feature_view.select(["city"]).to_pandas()["city"].unique().tolist()
        return cities if cities else [c['name'] for c in Config.CITIES]
    except Exception as e:
        st.warning(f"Using YAML cities (Feature Store error: {str(e)})")
        return [c['name'] for c in Config.CITIES]

def load_forecast(city):
    try:
        # Get models
        model_pm25 = model_registry.get_latest_model("Karachi_pm25")
        model_pm10 = model_registry.get_latest_model("Karachi_pm10")
        
        # Prepare input
        current = st.session_state.current_data.get(city)
        if not current: return None
            
        features = np.array([
            current['pm25'], current['pm10'],
            current['temperature'], current['humidity'],
            current['wind_speed']
        ]).reshape(1, -1)

        # Generate forecast
        dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") 
                for i in range(1, 4)]
        
        return pd.DataFrame({
            'date': dates,
            'pm25': model_pm25.predict(features)[:, 0],
            'pm10': model_pm10.predict(features)[:, 0]
        })

    except Exception as e:
        st.error(f"Forecast error: {str(e)}")
        return None

# ========== UI Components ==========
st.sidebar.title("AQI Predictor")
cities = load_cities()
selected_city = st.sidebar.selectbox("City", cities)
selected_pollutant = st.sidebar.radio("Pollutant", ["PM2.5", "PM10"])

# Main display
st.title(f"Air Quality: {selected_city}")
if st.sidebar.button("🔄 Refresh Data"):
    st.session_state.current_data.pop(selected_city, None)

# ========== Main Layout ==========

st.title(f"Air Quality Forecast for {selected_city}")

if selected_city not in st.session_state.current_data:
    with st.spinner(f"Loading {selected_city} data..."):
        load_current_data(selected_city)

current = st.session_state.current_data.get(selected_city)

if current:
    # Stack current AQI level and weather info
    st.subheader("Current AQI Levels")
    pollutant_value = current.get('pm25') if selected_pollutant == "PM2.5" else current.get('pm10')
    if pollutant_value:
        st.metric(label=f"{selected_pollutant} (µg/m³)", value=round(pollutant_value, 1))
    else:
        st.warning("Pollutant data not available.")

    st.subheader("Current Weather Information")
    st.metric("Temperature (°C)", round(current.get('temperature', 0), 1))
    st.metric("Humidity (%)", current.get('humidity', 0))
    st.metric("Wind Speed (m/s)", round(current.get('wind_speed', 0), 1))

    # Forecast Graph
    forecast = load_forecast(selected_city)

    if forecast is not None and not forecast.empty:
        st.subheader("3-Day Forecast")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['pm25'] if selected_pollutant == "PM2.5" else forecast['pm10'],
            mode='lines+markers',
            name=f"Forecasted {selected_pollutant}",
            line=dict(color="#636EFA", width=3)
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=f"{selected_pollutant} (µg/m³)",
            plot_bgcolor="#f9f9f9",
            paper_bgcolor="#f9f9f9",
            font=dict(size=14)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No forecast data available.")
else:
    st.error(f"No current data for {selected_city}.")
