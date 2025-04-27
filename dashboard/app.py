import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import hopsworks

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config
from data_collection.data_collector import DataCollector
from models.model_registry import ModelRegistry
from evaluation.feature_importance import FeatureImportanceAnalyzer

# Streamlit page config
st.set_page_config(
    page_title="AQI PREDICTOR DASHBOARD",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API keys check
if not Config.AQICN_API_KEY or not Config.OPENWEATHER_API_KEY:
    st.error("API keys missing. Please set AQICN_API_KEY and OPENWEATHER_API_KEY.")
    st.stop()

# Hopsworks connection
try:
    project = hopsworks.login(
        project=Config.HOPSWORKS_PROJECT_NAME,
        api_key_value=Config.HOPSWORKS_API_KEY,
        host=Config.HOPSWORKS_HOST,
        port=443
    )
    feature_store = project.get_feature_store()
except Exception as e:
    st.error(f"Failed to connect to Hopsworks: {e}")
    st.stop()

# Initialize collectors
data_collector = DataCollector(
    api_key_aqicn=Config.AQICN_API_KEY,
    api_key_openweather=Config.OPENWEATHER_API_KEY
)
model_registry = ModelRegistry()

# Session states
if 'current_data' not in st.session_state:
    st.session_state.current_data = {}
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = {}
if 'feature_importances' not in st.session_state:
    st.session_state.feature_importances = {}

# Helper Functions

def load_cities():
    return [city['name'] for city in Config.CITIES]

def load_current_data(city):
    try:
        # Fetch the feature view for the specified city
        feature_view = feature_store.get_feature_view(name="karachi_aqi_features")
        
        # Retrieve the most recent data (latest row)
        feature_data = feature_view.get_data().tail(1)
        
        # Check if data is available
        if not feature_data.empty:
            aqi_data = feature_data.iloc[0]
            st.session_state.current_data[city] = {
                'pm25': aqi_data.get('pm25', None),
                'pm10': aqi_data.get('pm10', None),
                'temperature': aqi_data.get('temperature', None),
                'humidity': aqi_data.get('humidity', None),
                'wind_speed': aqi_data.get('wind_speed', None)
            }
            return st.session_state.current_data[city]
        else:
            st.error(f"No data found for {city} in the feature store.")
            return None
    except Exception as e:
        st.error(f"Error loading data for {city}: {e}")
        return None

def prepare_input_features(city):
    current = st.session_state.current_data.get(city)
    if not current:
        return None
    input_features = np.array([ 
        current.get('pm25', 0),
        current.get('pm10', 0),
        current.get('temperature', 0),
        current.get('humidity', 0),
        current.get('wind_speed', 0)
    ]).reshape(1, -1)
    return input_features

def load_forecast(city):
    if city not in st.session_state.forecasts:
        best_model = model_registry.get_best_model(name="air_quality_model")
        if best_model:
            input_features = prepare_input_features(city)
            if input_features is not None:
                forecast = best_model.predict(input_features)
                dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 4)]
                st.session_state.forecasts[city] = pd.DataFrame({
                    'date': dates,
                    'pm25': forecast[:, 0],
                    'pm10': forecast[:, 1]
                })
            else:
                st.warning("Missing input features.")
    return st.session_state.forecasts.get(city)

def get_feature_importance(city):
    if city not in st.session_state.feature_importances:
        analyzer = FeatureImportanceAnalyzer(model_registry)
        best_model = model_registry.get_best_model(name="air_quality_model")
        if best_model:
            importance_data = analyzer.analyze(best_model)
            st.session_state.feature_importances[city] = importance_data
        else:
            st.warning("Model not found for feature importance.")
            st.session_state.feature_importances[city] = {}
    return st.session_state.feature_importances.get(city)

# ========== Sidebar ==========

st.sidebar.title("Pearls AQI Predictor")
st.sidebar.markdown("---")
cities = load_cities()
selected_city = st.sidebar.selectbox("🏠 Select City", cities)
selected_pollutant = st.sidebar.radio("Select Pollutant", ["PM2.5", "PM10"])
if st.sidebar.button("🔄 Refresh"):
    load_current_data(selected_city)
    st.session_state.forecasts.pop(selected_city, None)
    st.session_state.feature_importances.pop(selected_city, None)

# ========== Main Layout ==========

st.title(f"Air Quality Forecast")

if selected_city not in st.session_state.current_data:
    with st.spinner(f"Loading {selected_city} data..."):
        load_current_data(selected_city)

current = st.session_state.current_data.get(selected_city)

if current:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.subheader("Current AQI Levels")
        pollutant_value = current.get('pm25') if selected_pollutant == "PM2.5" else current.get('pm10')
        if pollutant_value is not None and not np.isnan(pollutant_value):
            st.metric(label=f"{selected_pollutant} (µg/m³)", value=round(pollutant_value, 1))
        else:
            st.warning("Pollutant data not available.")

    with col2:
        st.subheader("Weather Info")
        st.metric("Temperature (°C)", round(current.get('temperature', 0), 1))
        st.metric("Humidity (%)", current.get('humidity', 0))
        st.metric("Wind Speed (m/s)", round(current.get('wind_speed', 0), 1))

    with col3:
        st.subheader("Feature Importance")
        feature_importance = get_feature_importance(selected_city)
        if feature_importance is not None and not feature_importance == {}:
            st.dataframe(feature_importance)
        else:
            st.info("No feature importance available.")

    st.markdown("---")
    
    forecast = load_forecast(selected_city)

    if forecast is not None:
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
    st.error(f"No current data for {selected_city}.")
