import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import hopsworks
import plotly.graph_objects as go
import streamlit as st
from utils.config import Config

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for troubleshooting
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Initialize session state
def init_session_state():
    if 'fs' not in st.session_state:
        st.session_state.fs = None
    if 'cities' not in st.session_state:
        st.session_state.cities = []
    if 'current_data' not in st.session_state:
        st.session_state.current_data = {}
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = {}
    if 'pollutant' not in st.session_state:
        st.session_state.pollutant = "PM2.5"

# Connect to Hopsworks with robust error handling
def connect_to_hopsworks():
    try:
        logger.debug("Attempting to connect to Hopsworks...")
        project = hopsworks.login(
            project=Config.HOPSWORKS_PROJECT_NAME,
            api_key_value=Config.HOPSWORKS_API_KEY,
            host=Config.HOPSWORKS_HOST,
            port=443,
            hostname_verification=False
        )
        fs = project.get_feature_store()
        st.session_state.fs = fs
        logger.info("Successfully connected to Hopsworks")
        return True
    except Exception as e:
        logger.error(f"Connection failed: {str(e)}", exc_info=True)
        st.error(f"❌ Failed to connect to Hopsworks: {str(e)}")
        return False

# Load cities with multiple retrieval strategies
def load_cities():
    try:
        logger.debug("Loading cities from feature store...")
        
        # Method 1: Try direct feature view access
        fv = st.session_state.fs.get_feature_view("karachi_aqi_features", 1)
        data = fv.get_batch_data()
        
        # Convert to DataFrame if needed
        if not isinstance(data, pd.DataFrame):
            data = data.to_pandas()
        
        logger.debug(f"Data columns: {data.columns.tolist()}")
        logger.debug(f"Sample data:\n{data.head(2)}")
        
        # Verify city column exists
        if 'city' not in data.columns:
            raise ValueError("'city' column not found in feature data")
            
        cities = data['city'].unique().tolist()
        
        if not cities:
            raise ValueError("No cities found in feature data")
            
        logger.info(f"Loaded {len(cities)} cities from feature store")
        return cities
        
    except Exception as e:
        logger.error(f"Failed to load cities: {str(e)}", exc_info=True)
        st.error(f"⚠️ City loading error: {str(e)}")
        return []

# Get city data with comprehensive validation
def get_city_data(city):
    try:
        logger.debug(f"Loading data for {city}...")
        fv = st.session_state.fs.get_feature_view("karachi_aqi_features", 1)
        
        # Method 1: Optimized query
        try:
            query = fv.select_all().filter(fv.city == city)
            city_data = query.read()
        except:
            # Method 2: Fallback to pandas filtering
            all_data = fv.get_batch_data()
            if not isinstance(all_data, pd.DataFrame):
                all_data = all_data.to_pandas()
            city_data = all_data[all_data['city'] == city]
        
        # Validation
        if city_data.empty:
            raise ValueError(f"No records found for {city}")
            
        if 'date' not in city_data.columns:
            raise ValueError("'date' column missing")
            
        # Get most recent record
        latest = city_data.sort_values('date', ascending=False).iloc[0]
        logger.debug(f"Latest record:\n{latest}")
        
        # Validate all required fields
        required_fields = {
            'pm25': float,
            'pm10': float,
            'temperature': float, 
            'humidity': float,
            'wind_speed': float
        }
        
        result = {'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        for field, dtype in required_fields.items():
            if field not in latest:
                raise ValueError(f"Missing field: {field}")
            result[field] = dtype(latest[field])
        
        return result
        
    except Exception as e:
        logger.error(f"Data loading failed for {city}: {str(e)}", exc_info=True)
        st.error(f"⚠️ Data load error for {city}: {str(e)}")
        return None

# Generate forecast (replace with your actual model)
def generate_forecast(current_data, pollutant):
    try:
        logger.debug(f"Generating {pollutant} forecast...")
        
        # Get base value
        pollutant_key = pollutant.lower().replace(".", "")
        base_value = current_data[f'pm{pollutant_key}']
        
        # Generate dates (next 3 days)
        dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") 
                for i in range(1, 4)]
        
        # Mock predictions - REPLACE WITH ACTUAL MODEL
        predictions = [
            base_value * (1 + np.random.uniform(-0.1, 0.2)),
            base_value * (1 + np.random.uniform(-0.15, 0.25)),
            base_value * (1 + np.random.uniform(-0.2, 0.3))
        ]
        
        forecast_df = pd.DataFrame({
            'date': dates,
            f'predicted_{pollutant_key}': predictions
        })
        
        logger.debug(f"Forecast data:\n{forecast_df}")
        return forecast_df
        
    except Exception as e:
        logger.error(f"Forecast failed: {str(e)}", exc_info=True)
        st.error(f"⚠️ Forecast error: {str(e)}")
        return None

# UI Components
def display_current_metrics(city, data):
    st.subheader(f"Current Air Quality in {city}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PM2.5", f"{data['pm25']:.1f} µg/m³")
        st.metric("Temperature", f"{data['temperature']:.1f} °C")
    with col2:
        st.metric("PM10", f"{data['pm10']:.1f} µg/m³")
        st.metric("Humidity", f"{data['humidity']:.0f}%")
    
    st.caption(f"Last updated: {data['last_updated']}")

def display_forecast_chart(forecast, pollutant):
    pollutant_key = pollutant.lower().replace(".", "")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast['date'],
        y=forecast[f'predicted_{pollutant_key}'],
        mode='lines+markers',
        name=f"{pollutant} Forecast",
        line=dict(color='#FFA15A', width=3)
    ))
    
    fig.update_layout(
        title=f"3-Day {pollutant} Forecast",
        xaxis_title='Date',
        yaxis_title=f'{pollutant} (µg/m³)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Main App
def main():
    init_session_state()
    
    # Page config
    st.set_page_config(
        page_title="AQI Prediction Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Connect to Hopsworks
    if st.session_state.fs is None:
        with st.spinner("Connecting to Hopsworks..."):
            if not connect_to_hopsworks():
                st.stop()
    
    # Load cities
    if not st.session_state.cities:
        with st.spinner("Loading cities..."):
            st.session_state.cities = load_cities()
            if not st.session_state.cities:
                st.error("No cities available - check feature store")
                st.stop()
    
    # Sidebar controls
    st.sidebar.title("Settings")
    selected_city = st.sidebar.selectbox(
        "Select City", 
        st.session_state.cities,
        key='city_select'
    )
    
    st.session_state.pollutant = st.sidebar.selectbox(
        "Select Pollutant",
        ['PM2.5', 'PM10'],
        key='pollutant_select'
    )
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.session_state.current_data.pop(selected_city, None)
        st.session_state.forecasts.pop(selected_city, None)
        st.rerun()
    
    # Main content
    st.title(f"Air Quality Dashboard: {selected_city}")
    
    # Load current data
    if selected_city not in st.session_state.current_data:
        with st.spinner(f"Loading {selected_city} data..."):
            st.session_state.current_data[selected_city] = get_city_data(selected_city)
    
    # Display data
    current_data = st.session_state.current_data.get(selected_city)
    if current_data:
        display_current_metrics(selected_city, current_data)
        
        # Generate forecast
        if selected_city not in st.session_state.forecasts:
            with st.spinner("Generating forecast..."):
                st.session_state.forecasts[selected_city] = generate_forecast(
                    current_data,
                    st.session_state.pollutant
                )
        
        forecast = st.session_state.forecasts.get(selected_city)
        if forecast is not None:
            display_forecast_chart(forecast, st.session_state.pollutant)
    else:
        st.error(f"Failed to load data for {selected_city}")

if __name__ == "__main__":
    main()
