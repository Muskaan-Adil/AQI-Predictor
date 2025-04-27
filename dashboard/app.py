import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import hopsworks
import plotly.graph_objects as go
import shap
from utils.config import Config
from data_collection.data_collector import DataCollector
from models.model_registry import ModelRegistry

# Streamlit page config
st.set_page_config(
    page_title="AQI Predictor Dashboard",
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
except Exception as e:
    st.error(f"Failed to connect to Hopsworks: {e}")
    st.stop()

# Initialize collectors
model_registry = ModelRegistry()

# Session states
if 'current_data' not in st.session_state:
    st.session_state.current_data = {}
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = {}

# Helper Functions
def load_cities_from_feature_store():
    # Query the feature store to get unique city names
    feature_view = feature_store.get_feature_view(name="karachi_aqi_features", version=1)
    city_data = feature_view.select(["city"]).to_pandas()  # Select only the 'city' column
    unique_cities = city_data["city"].unique()  # Get unique city names
    return unique_cities.tolist()  # Convert to list and return

def load_feature_view_data(city):
    feature_view = feature_store.get_feature_view(name="karachi_aqi_features", version=1)
    feature_data = feature_view.select(["city", "pm25", "pm10", "temperature", "humidity", "wind_speed"]) \
                              .filter("city == ?", city).to_pandas()
    return feature_data

def load_current_data(city):
    try:
        feature_data = load_feature_view_data(city)
        if not feature_data.empty:
            st.session_state.current_data[city] = {
                'pm25': feature_data['pm25'].iloc[-1],
                'pm10': feature_data['pm10'].iloc[-1],
                'temperature': feature_data['temperature'].iloc[-1],
                'humidity': feature_data['humidity'].iloc[-1],
                'wind_speed': feature_data['wind_speed'].iloc[-1]
            }
            return st.session_state.current_data[city]
        else:
            st.error(f"No feature data available for {city}.")
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
    forecast = {}
    try:
        best_model_pm25 = model_registry.get_latest_model(name="Karachi_pm25")
        best_model_pm10 = model_registry.get_latest_model(name="Karachi_pm10")
        
        if best_model_pm25 and best_model_pm10:
            input_features = prepare_input_features(city)
            if input_features is not None:
                forecast_pm25 = best_model_pm25.predict(input_features)
                forecast_pm10 = best_model_pm10.predict(input_features)
                dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 4)]
                forecast = pd.DataFrame({
                    'date': dates,
                    'pm25': forecast_pm25[:, 0],
                    'pm10': forecast_pm10[:, 0]
                })
                
                # SHAP Explanation
                explainer_pm25 = shap.Explainer(best_model_pm25)
                shap_values_pm25 = explainer_pm25(input_features)
                
                # Display SHAP Summary Plot
                st.subheader(f"SHAP Explanation for {selected_pollutant} Prediction")
                shap.summary_plot(shap_values_pm25, input_features)
                
            else:
                st.warning("Missing input features.")
        else:
            st.warning("No models found for prediction.")
    except Exception as e:
        st.error(f"Error generating forecast: {e}")
    return forecast

# ========== Sidebar ==========

st.sidebar.title("AQI Predictor")
st.sidebar.markdown("---")
cities = load_cities_from_feature_store()
selected_city = st.sidebar.selectbox("Select City", cities)
selected_pollutant = st.sidebar.radio("Select Pollutant", ["PM2.5", "PM10"])
if st.sidebar.button("🔄 Refresh"):
    load_current_data(selected_city)
    st.session_state.forecasts.pop(selected_city, None)

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
