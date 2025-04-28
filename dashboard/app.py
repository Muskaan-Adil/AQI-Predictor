import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import hopsworks
import plotly.graph_objects as go
import streamlit as st

# ---- CRITICAL PATH CONFIGURATION ----
# Get the absolute path to the project root
project_root = Path(__file__).parents[1]  # Go up two levels from app.py
sys.path.insert(0, str(project_root))  # Insert at start of PATH

try:
    from utils.config import Config
except ImportError as e:
    st.error(f"""
    ❌ Import Error: {str(e)}
    Current Python path: {sys.path}
    Project root: {project_root}
    Please verify:
    1. You have a utils/ directory with config.py at: {project_root/'utils/config.py'}
    2. The directory structure is correct
    """)
    st.stop()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def init_session_state():
    """Initialize all session variables with type hints"""
    if 'fs' not in st.session_state:
        st.session_state.fs = None  # type: hopsworks.feature_store.FeatureStore
    if 'cities' not in st.session_state:
        st.session_state.cities = []  # type: list[str]
    if 'current_data' not in st.session_state:
        st.session_state.current_data = {}  # type: dict[str, dict]
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = {}  # type: dict[str, pd.DataFrame]

def connect_to_hopsworks() -> bool:
    """Connect to Hopsworks with validation"""
    try:
        logger.debug("Attempting Hopsworks connection...")
        project = hopsworks.login(
            project=Config.HOPSWORKS_PROJECT_NAME,
            api_key_value=Config.HOPSWORKS_API_KEY,
            host=Config.HOPSWORKS_HOST,
            port=443,
            hostname_verification=False
        )
        st.session_state.fs = project.get_feature_store()
        
        # Verify connection works
        test_fg = st.session_state.fs.get_feature_groups()
        logger.debug(f"Connected successfully. Found {len(test_fg)} feature groups")
        return True
    except Exception as e:
        logger.error(f"Connection failed: {str(e)}", exc_info=True)
        st.error(f"""
        ❌ Hopsworks Connection Failed
        Error: {str(e)}
        Verify:
        1. API Key is correct
        2. Project name is correct
        3. Host URL is accessible
        """)
        return False

def load_cities() -> list[str]:
    """Load cities with robust error handling"""
    try:
        logger.debug("Loading cities...")
        fv = st.session_state.fs.get_feature_view("karachi_aqi_features", 1)
        
        # Method 1: Direct query
        try:
            cities_df = fv.select(["city"]).distinct().read()
            cities = cities_df["city"].tolist()
        except:
            # Method 2: Fallback to batch data
            data = fv.get_batch_data()
            if not isinstance(data, pd.DataFrame):
                data = data.to_pandas()
            cities = data["city"].unique().tolist()
        
        if not cities:
            raise ValueError("No cities found in feature view")
            
        logger.debug(f"Found cities: {cities}")
        return cities
        
    except Exception as e:
        logger.error(f"City loading failed: {str(e)}", exc_info=True)
        st.error(f"""
        ⚠️ Failed to load cities
        Error: {str(e)}
        Verify:
        1. Feature view 'karachi_aqi_features' exists
        2. It contains a 'city' column
        """)
        return []

def load_city_data(city: str) -> dict:
    """Load city data with validation"""
    try:
        logger.debug(f"Loading data for {city}...")
        fv = st.session_state.fs.get_feature_view("karachi_aqi_features", 1)
        
        # Get most recent record
        query = fv.select_all().filter(fv.city == city).order_by("date", False).limit(1)
        record = query.read().iloc[0]
        
        # Validate fields
        required = {
            'pm25': float, 'pm10': float,
            'temperature': float, 'humidity': float,
            'wind_speed': float, 'date': 'datetime'
        }
        
        result = {}
        for field, dtype in required.items():
            if field not in record:
                raise ValueError(f"Missing field: {field}")
            result[field] = dtype(record[field]) if dtype != 'datetime' else record[field]
        
        result['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return result
        
    except Exception as e:
        logger.error(f"Data load failed for {city}: {str(e)}", exc_info=True)
        st.error(f"""
        ⚠️ Failed to load {city} data
        Error: {str(e)}
        Verify:
        1. City exists in feature view
        2. All required fields are present
        3. There are records for this city
        """)
        return None

def main():
    init_session_state()
    st.set_page_config(
        page_title="AQI Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Connection
    if st.session_state.fs is None:
        with st.spinner("Connecting to Hopsworks..."):
            if not connect_to_hopsworks():
                st.stop()
    
    # Load cities
    if not st.session_state.cities:
        with st.spinner("Loading cities..."):
            st.session_state.cities = load_cities()
            if not st.session_state.cities:
                st.stop()
    
    # UI
    st.sidebar.title("Controls")
    city = st.sidebar.selectbox("City", st.session_state.cities)
    pollutant = st.sidebar.radio("Pollutant", ["PM2.5", "PM10"])
    
    # Load data
    if city not in st.session_state.current_data:
        with st.spinner(f"Loading {city} data..."):
            st.session_state.current_data[city] = load_city_data(city)
    
    # Display
    data = st.session_state.current_data.get(city)
    if data:
        st.title(f"Air Quality: {city}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("PM2.5", f"{data['pm25']:.1f} µg/m³")
            st.metric("Temperature", f"{data['temperature']:.1f} °C")
        with col2:
            st.metric("PM10", f"{data['pm10']:.1f} µg/m³")
            st.metric("Humidity", f"{data['humidity']:.0f}%")
        
        st.caption(f"Last updated: {data['last_updated']}")
    else:
        st.error(f"Could not load data for {city}")

if __name__ == "__main__":
    main()
