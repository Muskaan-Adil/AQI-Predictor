import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import hopsworks
import plotly.graph_objects as go
import streamlit as st

# ---- PATH CONFIGURATION ----
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from utils.config import Config
except ImportError as e:
    st.error(f"❌ Import Error: {str(e)}")
    st.error("Please ensure:")
    st.error("1. You have a utils/ directory with config.py")
    st.error("2. The directory structure is correct")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_session_state():
    session_defaults = {
        'fs': None,
        'cities': [],
        'current_data': {},
        'forecasts': {},
        'pollutant': 'PM2.5'
    }
    for key, val in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def connect_to_hopsworks():
    try:
        project = hopsworks.login(
            project=Config.HOPSWORKS_PROJECT_NAME,
            api_key_value=Config.HOPSWORKS_API_KEY,
            host=Config.HOPSWORKS_HOST,
            port=443,
            hostname_verification=False
        )
        fs = project.get_feature_store()
        
        # Verify connection by checking feature view exists
        try:
            fs.get_feature_view("karachi_aqi_features", 1)
            return fs
        except Exception as e:
            st.error(f"❌ Feature view not found: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"❌ Hopsworks connection failed: {str(e)}")
        return None

def load_cities(fs):
    try:
        fv = fs.get_feature_view("karachi_aqi_features", 1)
        data = fv.get_batch_data()
        if not isinstance(data, pd.DataFrame):
            data = data.to_pandas()
            
        if 'city' not in data.columns:
            st.error("⚠️ 'city' column not found in feature data")
            return []
            
        return data["city"].unique().tolist()
    except Exception as e:
        st.error(f"⚠️ Failed to load cities: {str(e)}")
        return []

def load_city_data(fs, city):
    try:
        fv = fs.get_feature_view("karachi_aqi_features", 1)
        data = fv.get_batch_data()
        if not isinstance(data, pd.DataFrame):
            data = data.to_pandas()
        
        # Validate required columns exist
        required_columns = ['city', 'timestamp', 'pm25', 'pm10', 
                           'temperature', 'humidity', 'wind_speed']
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            st.error(f"⚠️ Missing columns: {', '.join(missing)}")
            return None
            
        city_data = data[data["city"] == city]
        if city_data.empty:
            st.warning(f"No records found for {city}")
            return None
            
        # Sort by timestamp instead of date
        latest = city_data.sort_values("timestamp", ascending=False).iloc[0]
        
        # Convert timestamp to readable format if needed
        if isinstance(latest['timestamp'], (int, float)):
            last_updated = datetime.fromtimestamp(latest['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
        else:
            last_updated = str(latest['timestamp'])
        
        return {
            'pm25': float(latest['pm25']),
            'pm10': float(latest['pm10']),
            'temperature': float(latest['temperature']),
            'humidity': float(latest['humidity']),
            'wind_speed': float(latest['wind_speed']),
            'last_updated': last_updated
        }
    except Exception as e:
        st.error(f"⚠️ Failed to load {city} data: {str(e)}")
        return None

def generate_forecast(current_data, pollutant):
    try:
        base_value = current_data[f'pm{pollutant.replace(".", "")}']
        dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") 
                for i in range(1, 4)]
        return pd.DataFrame({
            'date': dates,
            f'predicted_{pollutant.replace(".", "")}': [
                base_value * (1 + np.random.uniform(-0.1, 0.2)),
                base_value * (1 + np.random.uniform(-0.15, 0.25)),
                base_value * (1 + np.random.uniform(-0.2, 0.3))
            ]
        })
    except Exception as e:
        st.error(f"⚠️ Forecast failed: {str(e)}")
        return None

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

def display_forecast(forecast, pollutant):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast['date'],
        y=forecast[f'predicted_{pollutant.replace(".", "")}'],
        mode='lines+markers',
        line=dict(color='#FFA15A', width=3)
    ))
    fig.update_layout(
        title=f"3-Day {pollutant} Forecast",
        xaxis_title="Date",
        yaxis_title=f"{pollutant} (µg/m³)"
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    init_session_state()
    st.set_page_config(
        page_title="AQI Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Connect to Hopsworks
    if st.session_state.fs is None:
        with st.spinner("Connecting to Hopsworks..."):
            st.session_state.fs = connect_to_hopsworks()
            if not st.session_state.fs:
                st.stop()

    # Debug: Show feature view columns
    try:
        fv = st.session_state.fs.get_feature_view("karachi_aqi_features", 1)
        data_sample = fv.get_batch_data()
        if not isinstance(data_sample, pd.DataFrame):
            data_sample = data_sample.to_pandas()
        st.sidebar.write("Feature view columns:", data_sample.columns.tolist())
    except:
        pass

    # Load cities
    if not st.session_state.cities:
        with st.spinner("Loading cities..."):
            st.session_state.cities = load_cities(st.session_state.fs)
            if not st.session_state.cities:
                st.error("No cities available")
                st.stop()

    # UI Controls
    st.sidebar.title("Controls")
    selected_city = st.sidebar.selectbox("City", st.session_state.cities)
    st.session_state.pollutant = st.sidebar.selectbox("Pollutant", ["PM2.5", "PM10"])
    
    if st.sidebar.button("🔄 Refresh"):
        st.session_state.current_data.pop(selected_city, None)
        st.session_state.forecasts.pop(selected_city, None)
        st.rerun()

    # Main content
    st.title(f"Air Quality: {selected_city}")

    # Load data
    if selected_city not in st.session_state.current_data:
        with st.spinner(f"Loading {selected_city} data..."):
            st.session_state.current_data[selected_city] = load_city_data(
                st.session_state.fs,
                selected_city
            )

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
            display_forecast(forecast, st.session_state.pollutant)
    else:
        st.error(f"Could not load data for {selected_city}")

if __name__ == "__main__":
    main()
