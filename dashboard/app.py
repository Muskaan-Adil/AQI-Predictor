import sys
from pathlib import Path
import logging
import time

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import hopsworks
import plotly.graph_objects as go
from utils.config import Config
from models.model_registry import ModelRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CONNECTION_TIMEOUT = 30  # seconds
DATA_LOAD_TIMEOUT = 20  # seconds

# Streamlit config
st.set_page_config(
    page_title="AQI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize all session state variables"""
    if 'feature_store' not in st.session_state:
        st.session_state.feature_store = None
    if 'cities' not in st.session_state:
        st.session_state.cities = []
    if 'current_data' not in st.session_state:
        st.session_state.current_data = {}
    if 'last_update' not in st.session_state:
        st.session_state.last_update = {}

def connect_to_hopsworks():
    """Establish connection to Hopsworks with timeout"""
    start_time = time.time()
    
    try:
        project = hopsworks.login(
            project=Config.HOPSWORKS_PROJECT_NAME,
            api_key_value=Config.HOPSWORKS_API_KEY,
            host=Config.HOPSWORKS_HOST,
            port=443,
            hostname_verification=False
        )
        st.session_state.feature_store = project.get_feature_store()
        logger.info("Successfully connected to Hopsworks")
        return True
    except Exception as e:
        logger.error(f"Hopsworks connection failed: {str(e)}")
        st.error(f"Failed to connect to Hopsworks: {str(e)}")
        return False

def load_cities_from_hopsworks():
    """Load cities from Hopsworks feature store"""
    if not st.session_state.feature_store:
        return []
    
    try:
        start_time = time.time()
        feature_view = st.session_state.feature_store.get_feature_view(
            name="karachi_aqi_features", 
            version=1
        )
        cities_df = feature_view.select(["city"]).to_pandas()
        
        if time.time() - start_time > DATA_LOAD_TIMEOUT:
            raise TimeoutError("City data loading timed out")
            
        cities = cities_df["city"].unique().tolist()
        st.session_state.cities = cities
        logger.info(f"Loaded {len(cities)} cities from Hopsworks")
        return cities
    except Exception as e:
        logger.error(f"Failed to load cities: {str(e)}")
        st.error(f"City data unavailable: {str(e)}")
        return []

def load_city_data(city):
    """Load current data for a specific city"""
    if not st.session_state.feature_store:
        return None
    
    try:
        feature_view = st.session_state.feature_store.get_feature_view(
            name="karachi_aqi_features", 
            version=1
        )
        
        # Get most recent record for the city
        df = feature_view.filter(
            feature_view.city == city
        ).to_pandas().sort_values('date', ascending=False).head(1)
        
        if df.empty:
            logger.warning(f"No data found for city: {city}")
            return None
            
        current_data = {
            'pm25': df['pm25'].values[0],
            'pm10': df['pm10'].values[0],
            'temperature': df['temperature'].values[0],
            'humidity': df['humidity'].values[0],
            'wind_speed': df['wind_speed'].values[0]
        }
        
        st.session_state.last_update[city] = datetime.now()
        return current_data
        
    except Exception as e:
        logger.error(f"Failed to load data for {city}: {str(e)}")
        return None

def display_city_metrics(city, data):
    """Display metrics for a city"""
    st.subheader(f"Current Air Quality: {city}")
    
    cols = st.columns(4)
    with cols[0]:
        st.metric("PM2.5", f"{data['pm25']:.1f} µg/m³")
    with cols[1]:
        st.metric("PM10", f"{data['pm10']:.1f} µg/m³")
    with cols[2]:
        st.metric("Temperature", f"{data['temperature']:.1f} °C")
    with cols[3]:
        st.metric("Humidity", f"{data['humidity']:.0f}%")
    
    if city in st.session_state.last_update:
        st.caption(f"Last updated: {st.session_state.last_update[city].strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    initialize_session_state()
    
    # Connect to Hopsworks if not already connected
    if st.session_state.feature_store is None:
        with st.spinner("Connecting to Hopsworks..."):
            if not connect_to_hopsworks():
                st.error("Critical error: Could not connect to Hopsworks")
                st.stop()
    
    # Load cities
    if not st.session_state.cities:
        with st.spinner("Loading city list..."):
            cities = load_cities_from_hopsworks()
            if not cities:
                st.error("No cities available - please check your feature store")
                st.stop()
    
    # Sidebar controls
    st.sidebar.title("AQI Dashboard")
    selected_city = st.sidebar.selectbox(
        "Select City", 
        st.session_state.cities,
        key="city_select"
    )
    
    refresh_button = st.sidebar.button("Refresh Data")
    
    # Main content
    if refresh_button or selected_city not in st.session_state.current_data:
        with st.spinner(f"Loading data for {selected_city}..."):
            st.session_state.current_data[selected_city] = load_city_data(selected_city)
    
    if st.session_state.current_data.get(selected_city):
        display_city_metrics(selected_city, st.session_state.current_data[selected_city])
    else:
        st.error(f"Could not load data for {selected_city}")

if __name__ == "__main__":
    main()
