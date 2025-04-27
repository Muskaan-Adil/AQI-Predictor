import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import hopsworks

# Ensure the correct configuration is loaded
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config
from data_collection.data_collector import DataCollector
from models.model_registry import ModelRegistry
from evaluation.feature_importance import FeatureImportanceAnalyzer

# Set up Streamlit page configuration
st.set_page_config(
    page_title="AQI PREDICTOR DASHBOARD",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check for required API keys
if not Config.AQICN_API_KEY or not Config.OPENWEATHER_API_KEY:
    st.error("API keys are missing. Please set AQICN_API_KEY and OPENWEATHER_API_KEY in your environment.")
    st.stop()

# Initialize the Hopsworks project and feature store
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

# Initialize DataCollector and ModelRegistry with required API keys
data_collector = DataCollector(
    api_key_aqicn=Config.AQICN_API_KEY,
    api_key_openweather=Config.OPENWEATHER_API_KEY
)
model_registry = ModelRegistry()

# Initialize session state variables
if 'current_data' not in st.session_state:
    st.session_state.current_data = {}
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = {}
if 'feature_importances' not in st.session_state:
    st.session_state.feature_importances = {}

def load_cities():
    """Load list of cities from config."""
    return [city['name'] for city in Config.CITIES]

def load_current_data(city):
    """Load current AQI data for a city."""
    city_info = next((c for c in Config.CITIES if c['name'] == city), None)
    if not city_info:
        st.error(f"City '{city}' not found in configuration.")
        return None
    try:
        raw_data = data_collector.collect_data(city_info)
        if raw_data:
            aqi_data = raw_data.get('aqi', {})
            weather_data = raw_data.get('weather', {})
            pm25 = aqi_data.get(data_collector.default_parameter)
            pm10 = aqi_data.get('pm10')
            temperature = weather_data.get('main', {}).get('temp')
            humidity = weather_data.get('main', {}).get('humidity')
            wind_speed = weather_data.get('wind', {}).get('speed')

            # Store extracted data in session state
            st.session_state.current_data[city] = {
                'pm25': pm25,
                'pm10': pm10,
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed
            }
            return st.session_state.current_data[city]
        else:
            st.error(f"Failed to collect data for {city}.")
            return None
    except Exception as e:
        st.error(f"Failed to load data for {city}: {e}")
        return None

def load_forecast(city):
    """Load forecasts for a city."""
    if city not in st.session_state.forecasts:
        # Load the best model from the registry
        best_model = model_registry.get_best_model(name="air_quality_model")
        if best_model:
            # Prepare input features for prediction
            input_features = prepare_input_features(city)
            forecast = best_model.predict(input_features)
            dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 4)]
            st.session_state.forecasts[city] = pd.DataFrame({
                'date': dates,
                'pm25': forecast[:, 0],  # Assuming PM2.5 predictions
                'pm10': forecast[:, 1]   # Assuming PM10 predictions
            })
        else:
            st.warning(f"No model available for {city}. Using default forecasts.")
            dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 4)]
            st.session_state.forecasts[city] = pd.DataFrame({
                'date': dates,
                'pm25': [50, 55, 60],  # Default values
                'pm10': [30, 35, 40]   # Default values
            })
    return st.session_state.forecasts[city]

def get_feature_importance(city):
    """Get feature importance for a city."""
    if city not in st.session_state.feature_importances:
        analyzer = FeatureImportanceAnalyzer(model_registry)
        importance_data = analyzer.analyze(city)  # Define this method elsewhere
        st.session_state.feature_importances[city] = importance_data
    return st.session_state.feature_importances[city]

def get_aqi_category_pm25(pm25_value):
    """Get AQI category based on PM2.5 value."""
    if pm25_value is None:
        return "Unknown"
    elif pm25_value <= 12:
        return "Good"
    elif pm25_value <= 35.4:
        return "Moderate"
    elif pm25_value <= 55.4:
        return "Unhealthy for Sensitive Groups"
    elif pm25_value <= 150.4:
        return "Unhealthy"
    elif pm25_value <= 250.4:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def get_aqi_color_pm25(pm25_value):
    """Get color for AQI category based on PM2.5 value."""
    if pm25_value is None:
        return "#CCCCCC"
    elif pm25_value <= 12:
        return "#00E400"
    elif pm25_value <= 35.4:
        return "#FFFF00"
    elif pm25_value <= 55.4:
        return "#FF7E00"
    elif pm25_value <= 150.4:
        return "#FF0000"
    elif pm25_value <= 250.4:
        return "#8F3F97"
    else:
        return "#7E0023"

def get_aqi_category_pm10(pm10_value):
    """Get AQI category based on PM10 value."""
    if pm10_value is None:
        return "Unknown"
    elif pm10_value <= 54:
        return "Good"
    elif pm10_value <= 154:
        return "Moderate"
    elif pm10_value <= 254:
        return "Unhealthy for Sensitive Groups"
    elif pm10_value <= 354:
        return "Unhealthy"
    elif pm10_value <= 424:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def get_aqi_color_pm10(pm10_value):
    """Get color for AQI category based on PM10 value."""
    if pm10_value is None:
        return "#CCCCCC"
    elif pm10_value <= 54:
        return "#00E400"
    elif pm10_value <= 154:
        return "#FFFF00"
    elif pm10_value <= 254:
        return "#FF7E00"
    elif pm10_value <= 354:
        return "#FF0000"
    elif pm10_value <= 424:
        return "#8F3F97"
    else:
        return "#7E0023"

# Sidebar configuration
st.sidebar.title("Pearls AQI Predictor")
st.sidebar.markdown("### Select a City")
cities = load_cities()
selected_city = st.sidebar.selectbox("City", cities)

selected_pollutant = st.sidebar.selectbox(
    "Select Pollutant",
    ["PM2.5", "PM10"]
)

if st.sidebar.button("Refresh Data"):
    load_current_data(selected_city)
    if selected_city in st.session_state.forecasts:
        del st.session_state.forecasts[selected_city]
    if selected_city in st.session_state.feature_importances:
        del st.session_state.feature_importances[selected_city]

# Main content
st.title(f"Air Quality Prediction for {selected_city}")

if selected_city not in st.session_state.current_data:
    with st.spinner(f"Loading current data for {selected_city}..."):
        load_current_data(selected_city)

current_data = st.session_state.current_data.get(selected_city, {})

if current_data:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### Current {selected_pollutant}")
        pollutant_value = current_data.get('pm25') if selected_pollutant == "PM2.5" else current_data.get('pm10')
        category = get_aqi_category_pm25(pollutant_value) if selected_pollutant == "PM2.5" else get_aqi_category_pm10(pollutant_value)
        color = get_aqi_color_pm25(pollutant_value) if selected_pollutant == "PM2.5" else get_aqi_color_pm10(pollutant_value)
        st.markdown(f"<h1 style='color: {color};'>{pollutant_value:.1f}</h1>", unsafe_allow_html=True)
        st.markdown(f"**Category**: {category}")
    with col2:
        st.markdown("### Weather Conditions")
        temp = current_data.get('temperature')
        humidity = current_data.get('humidity')
        wind = current_data.get('wind_speed')
        if temp is not None:
            st.markdown(f"**Temperature**: {temp:.1f}°C")
        if humidity is not None:
            st.markdown(f"**Humidity**: {humidity}%")
        if wind is not None:
            st.markdown(f"**Wind Speed**: {wind:.1f} m/s")

    with col3:
        st.markdown("### Feature Importance")
        feature_importance = get_feature_importance(selected_city)
        st.write(feature_importance)

    st.markdown("---")

    # Forecast section
    forecast_data = load_forecast(selected_city)
    st.subheader("Forecast for Next 3 Days")
    st.dataframe(forecast_data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_data['date'],
        y=forecast_data['pm25'] if selected_pollutant == "PM2.5" else forecast_data['pm10'],
        mode='lines+markers',
        name=selected_pollutant,
        line=dict(width=3),
        marker=dict(color=forecast_data['pm25'] if selected_pollutant == "PM2.5" else forecast_data['pm10'])
    ))
    st.plotly_chart(fig)
else:
    st.error(f"No data available for {selected_city}.")
