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
from utils.config import Config

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG for more details
logger = logging.getLogger(__name__)

# Streamlit config
st.set_page_config(
    page_title="AQI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'feature_store': None,
        'cities': [],
        'current_data': {},
        'last_update': {},
        'connection_status': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def connect_to_hopsworks():
    """Establish connection to Hopsworks with detailed error handling"""
    try:
        logger.debug("Attempting to connect to Hopsworks...")
        project = hopsworks.login(
            project=Config.HOPSWORKS_PROJECT_NAME,
            api_key_value=Config.HOPSWORKS_API_KEY,
            host=Config.HOPSWORKS_HOST,
            port=443,
            hostname_verification=False
        )
        st.session_state.feature_store = project.get_feature_store()
        st.session_state.connection_status = "connected"
        logger.info("Successfully connected to Hopsworks")
        return True
    except Exception as e:
        logger.error(f"Hopsworks connection failed: {str(e)}", exc_info=True)
        st.session_state.connection_status = f"failed: {str(e)}"
        return False

def load_feature_data():
    """Load all feature data with comprehensive error handling"""
    try:
        logger.debug("Loading feature data...")
        feature_view = st.session_state.feature_store.get_feature_view(
            name="karachi_aqi_features", 
            version=1
        )
        
        # Get data with timeout protection
        start_time = time.time()
        feature_data = feature_view.get_batch_data()
        
        # Convert to pandas if needed
        if not isinstance(feature_data, pd.DataFrame):
            feature_data = feature_data.to_pandas()
        
        logger.debug(f"Data loaded successfully. Columns: {feature_data.columns.tolist()}")
        logger.debug(f"Sample data:\n{feature_data.head(2)}")
        
        return feature_data
    
    except Exception as e:
        logger.error(f"Failed to load feature data: {str(e)}", exc_info=True)
        st.error(f"Data loading error: {str(e)}")
        return None

def load_cities():
    """Load cities from feature data with validation"""
    feature_data = load_feature_data()
    if feature_data is None:
        return []
    
    try:
        if 'city' not in feature_data.columns:
            raise ValueError("'city' column not found in feature data")
            
        cities = feature_data['city'].unique().tolist()
        if not cities:
            raise ValueError("No cities found in feature data")
            
        logger.debug(f"Found cities: {cities}")
        return cities
        
    except Exception as e:
        logger.error(f"Failed to extract cities: {str(e)}")
        st.error(f"City data error: {str(e)}")
        return []

def load_city_data(city):
    """Load data for specific city with validation"""
    feature_data = load_feature_data()
    if feature_data is None:
        return None
    
    try:
        # Validate city exists
        if city not in feature_data['city'].values:
            raise ValueError(f"City '{city}' not found in data")
        
        # Filter and sort data
        city_data = feature_data[feature_data['city'] == city]
        if city_data.empty:
            raise ValueError(f"No records found for city '{city}'")
            
        # Get latest record
        latest_record = city_data.sort_values('date', ascending=False).iloc[0]
        
        # Validate required fields
        required_fields = ['pm25', 'pm10', 'temperature', 'humidity', 'wind_speed']
        for field in required_fields:
            if field not in latest_record:
                raise ValueError(f"Missing required field: {field}")
        
        data = {
            'pm25': float(latest_record['pm25']),
            'pm10': float(latest_record['pm10']),
            'temperature': float(latest_record['temperature']),
            'humidity': float(latest_record['humidity']),
            'wind_speed': float(latest_record['wind_speed'])
        }
        
        logger.debug(f"Loaded data for {city}: {data}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load data for {city}: {str(e)}")
        return None

def display_ui():
    """Main UI rendering with status information"""
    st.sidebar.title("AQI Dashboard")
    st.sidebar.write(f"**Connection Status:** {st.session_state.connection_status or 'Not attempted'}")
    
    if not st.session_state.cities:
        st.sidebar.warning("No cities available")
    else:
        selected_city = st.sidebar.selectbox(
            "Select City", 
            st.session_state.cities,
            key="city_select"
        )
        
        if st.sidebar.button("🔄 Refresh Data"):
            st.session_state.current_data.pop(selected_city, None)
            st.rerun()
            
        if selected_city not in st.session_state.current_data:
            with st.spinner(f"Loading data for {selected_city}..."):
                st.session_state.current_data[selected_city] = load_city_data(selected_city)
        
        if st.session_state.current_data.get(selected_city):
            display_city_metrics(selected_city)
        else:
            st.error(f"Could not load data for {selected_city}")
            st.write("Possible issues:")
            st.write("- The city exists but has no records")
            st.write("- Required fields are missing in the data")
            st.write("- Feature store permissions may be insufficient")

def display_city_metrics(city):
    """Display metrics for a city"""
    data = st.session_state.current_data[city]
    
    st.title(f"Air Quality: {city}")
    
    # Current Conditions
    st.header("Current Conditions")
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
            connect_to_hopsworks()
    
    # Load cities if not already loaded
    if not st.session_state.cities and st.session_state.feature_store:
        with st.spinner("Loading city data..."):
            st.session_state.cities = load_cities()
    
    # Main UI
    display_ui()

if __name__ == "__main__":
    main()
