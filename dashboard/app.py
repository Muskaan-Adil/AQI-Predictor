import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yaml
import hopsworks
import os

# Import project modules
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

# Load city names from cities.yaml
def load_cities():
    try:
        with open("cities.yaml", "r") as file:
            cities = yaml.safe_load(file)
            return [city["name"] for city in cities]
    except Exception as e:
        st.error(f"Error loading cities from YAML: {e}")
        return []

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

def load_feature_view_data(city):
    feature_view = feature_store.get_feature_view("karachi_aqi_features")
    feature_data = feature_view.select(["city", "pm25", "pm10", "temperature", "humidity", "wind_speed"]).filter("city == ?", city).to_pandas()
    return feature_data

def load_current_data(city):
    city_data = load_feature_view_data(city)
    if city_data is not None and not city_data.empty:
        st.session_state.current_data[city] = {
            'pm25': city_data['pm25'].iloc[-1],
            'pm10': city_data['pm10'].iloc[-1],
            'temperature': city_data['temperature'].iloc[-1],
            'humidity': city_data['humidity'].iloc[-1],
            'wind_speed': city_data['wind_speed'].iloc[-1]
        }
        return st.session_state.current_data[city]
    else:
        st.error(f"Failed to collect data for {city}.")
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
selected_city = st.sidebar.selectbox("Select City", cities)
selected_pollutant = st.sidebar.radio("Select Pollutant", ["PM2.5", "PM10"])
if st.sidebar.button("Refresh"):
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
        if pollutant_value:
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
