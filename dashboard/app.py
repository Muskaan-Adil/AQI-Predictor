import sys
from pathlib import Path
import logging

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(
    page_title=Config.DASHBOARD_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize all required session state variables"""
    if 'current_data' not in st.session_state:
        st.session_state.current_data = {}
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = {}

def connect_to_hopsworks():
    """Establish connection to Hopsworks feature store"""
    try:
        project = hopsworks.login(
            project=Config.HOPSWORKS_PROJECT_NAME,
            api_key_value=Config.HOPSWORKS_API_KEY,
            host=Config.HOPSWORKS_HOST,
            port=443
        )
        feature_store = project.get_feature_store()
        st.success("✅ Connected to Hopsworks")
        return feature_store
    except Exception as e:
        st.error(f"❌ Hopsworks connection failed: {str(e)}")
        st.stop()
        return None

def load_cities(feature_store):
    """Load cities from feature store with YAML fallback"""
    try:
        feature_view = feature_store.get_feature_view(name="karachi_aqi_features", version=1)
        cities = feature_view.select(["city"]).to_pandas()["city"].unique().tolist()
        return cities if cities else [c['name'] for c in Config.CITIES]
    except Exception as e:
        st.warning(f"Using YAML cities (Feature Store error: {str(e)})")
        return [c['name'] for c in Config.CITIES]

def load_current_data(feature_store, city):
    """Load current AQI and weather data for a city"""
    try:
        # Get latest data from feature store
        feature_view = feature_store.get_feature_view(
            name="karachi_aqi_features", 
            version=1
        )
        
        # Get most recent record for selected city
        df = feature_view.filter(
            feature_view.city == city
        ).to_pandas().sort_values('date', ascending=False).head(1)
        
        if not df.empty:
            st.session_state.current_data[city] = {
                'pm25': df['pm25'].values[0],
                'pm10': df['pm10'].values[0],
                'temperature': df['temperature'].values[0],
                'humidity': df['humidity'].values[0],
                'wind_speed': df['wind_speed'].values[0],
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            logger.info(f"Loaded data for {city}")
        else:
            st.error(f"No data available for {city}")
            
    except Exception as e:
        st.error(f"Failed to load data for {city}: {str(e)}")
        logger.error(f"Data loading error: {str(e)}")

def load_forecast(city):
    """Generate pollution forecast for selected city"""
    try:
        # Get models
        model_pm25 = model_registry.get_latest_model("Karachi_pm25")
        model_pm10 = model_registry.get_latest_model("Karachi_pm10")
        
        # Prepare input
        current = st.session_state.current_data.get(city)
        if not current:
            return None
            
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
        logger.error(f"Forecast generation failed: {str(e)}")
        return None

def display_current_metrics(city, pollutant):
    """Display current AQI and weather metrics"""
    current = st.session_state.current_data.get(city)
    
    if not current:
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Current AQI Levels")
        pollutant_value = current.get('pm25') if pollutant == "PM2.5" else current.get('pm10')
        if pollutant_value is not None:
            st.metric(label=f"{pollutant} (µg/m³)", value=round(pollutant_value, 1))
        else:
            st.warning("Pollutant data not available")
        st.caption(f"Last updated: {current.get('last_updated', 'N/A')}")
    
    with col2:
        st.subheader("Weather Metrics")
        st.metric("Temperature (°C)", round(current.get('temperature', 0), 1))
        st.metric("Humidity (%)", current.get('humidity', 0))
    
    with col3:
        st.subheader("Wind")
        st.metric("Wind Speed (m/s)", round(current.get('wind_speed', 0), 1))

def display_forecast_chart(city, pollutant):
    """Display forecast visualization"""
    forecast = load_forecast(city)
    
    if forecast is None or forecast.empty:
        st.warning("No forecast data available.")
        return
    
    st.subheader("3-Day Forecast")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast['date'],
        y=forecast['pm25'] if pollutant == "PM2.5" else forecast['pm10'],
        mode='lines+markers',
        name=f"Forecasted {pollutant}",
        line=dict(color="#636EFA", width=3)
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=f"{pollutant} (µg/m³)",
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="#f9f9f9",
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    initialize_session_state()
    
    # Connect to Hopsworks
    feature_store = connect_to_hopsworks()
    if not feature_store:
        return
    
    # Initialize model registry
    global model_registry
    model_registry = ModelRegistry()
    
    # Sidebar controls
    st.sidebar.title("AQI Predictor")
    cities = load_cities(feature_store)
    selected_city = st.sidebar.selectbox("City", cities)
    selected_pollutant = st.sidebar.radio("Pollutant", ["PM2.5", "PM10"])
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.session_state.current_data.pop(selected_city, None)
        st.rerun()
    
    # Main display
    st.title(f"Air Quality: {selected_city}")
    
    # Load data if not already loaded
    if selected_city not in st.session_state.current_data:
        with st.spinner(f"Loading {selected_city} data..."):
            load_current_data(feature_store, selected_city)
    
    # Display content
    current_data = st.session_state.current_data.get(selected_city)
    
    if current_data:
        display_current_metrics(selected_city, selected_pollutant)
        display_forecast_chart(selected_city, selected_pollutant)
    else:
        st.error(f"No current data available for {selected_city}.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.exception("Application crashed")
        st.stop()
