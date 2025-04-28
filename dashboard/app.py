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
        'pollutant': 'PM2.5',
        'data_loaded': False
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
        
        # Verify connection works
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
            
        cities = data["city"].unique().tolist()
        
        # Ensure we have valid city names
        cities = [c for c in cities if pd.notna(c) and str(c).strip() != ""]
        
        if not cities:
            st.error("⚠️ No valid cities found in feature data")
            return []
            
        return cities
    except Exception as e:
        st.error(f"⚠️ Failed to load cities: {str(e)}")
        return []

def load_city_data(fs, city):
    try:
        fv = fs.get_feature_view("karachi_aqi_features", 1)
        data = fv.get_batch_data()
        if not isinstance(data, pd.DataFrame):
            data = data.to_pandas()
        
        # Validate city exists
        if city not in data["city"].values:
            st.error(f"⚠️ City '{city}' not found in data")
            return None
            
        city_data = data[data["city"] == city]
        if city_data.empty:
            st.warning(f"⚠️ No records found for {city}")
            return None
            
        # Get most recent record
        if 'timestamp' in city_data.columns:
            latest = city_data.sort_values("timestamp", ascending=False).iloc[0]
        elif 'date' in city_data.columns:  # Fallback to date if timestamp doesn't exist
            latest = city_data.sort_values("date", ascending=False).iloc[0]
        else:
            st.error("⚠️ Neither 'timestamp' nor 'date' column found")
            return None
        
        # Prepare result with validation
        result = {
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for field in ['pm25', 'pm10', 'temperature', 'humidity', 'wind_speed']:
            if field in latest:
                try:
                    result[field] = float(latest[field])
                except:
                    st.warning(f"⚠️ Invalid value for {field}")
                    result[field] = 0.0
            else:
                st.warning(f"⚠️ Missing field: {field}")
                result[field] = 0.0
                
        return result
        
    except Exception as e:
        st.error(f"⚠️ Failed to load {city} data: {str(e)}")
        return None

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

    # Load cities if not loaded yet
    if not st.session_state.cities:
        with st.spinner("Loading cities..."):
            st.session_state.cities = load_cities(st.session_state.fs)
            if not st.session_state.cities:
                st.error("❌ No cities available - check feature store data")
                st.stop()

    # Sidebar controls
    st.sidebar.title("Controls")
    
    # Ensure we have valid cities before showing selectbox
    if st.session_state.cities:
        selected_city = st.sidebar.selectbox(
            "City", 
            st.session_state.cities,
            index=0  # Always select first city by default
        )
    else:
        st.sidebar.error("No cities available")
        selected_city = None
    
    st.session_state.pollutant = st.sidebar.radio(
        "Pollutant",
        ["PM2.5", "PM10"],
        index=0
    )
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.session_state.current_data = {}
        st.session_state.forecasts = {}
        st.rerun()

    # Main content
    if selected_city:
        st.title(f"Air Quality: {selected_city}")
        
        # Load data if not loaded or city changed
        if selected_city not in st.session_state.current_data:
            with st.spinner(f"Loading {selected_city} data..."):
                st.session_state.current_data[selected_city] = load_city_data(
                    st.session_state.fs,
                    selected_city
                )
        
        # Display data
        current_data = st.session_state.current_data.get(selected_city)
        if current_data:
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("PM2.5", f"{current_data['pm25']:.1f} µg/m³")
                st.metric("Temperature", f"{current_data['temperature']:.1f} °C")
            with col2:
                st.metric("PM10", f"{current_data['pm10']:.1f} µg/m³")
                st.metric("Humidity", f"{current_data['humidity']:.0f}%")
            st.caption(f"Last updated: {current_data['last_updated']}")
            
            # Generate and display forecast
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
    else:
        st.error("No valid city selected")

def generate_forecast(current_data, pollutant):
    try:
        pollutant_key = pollutant.lower().replace(".", "")
        base_value = current_data[f'pm{pollutant_key}']
        
        dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") 
                for i in range(1, 4)]
        
        return pd.DataFrame({
            'date': dates,
            f'predicted_{pollutant_key}': [
                base_value * (1 + np.random.uniform(-0.1, 0.2)),
                base_value * (1 + np.random.uniform(-0.15, 0.25)),
                base_value * (1 + np.random.uniform(-0.2, 0.3))
            ]
        })
    except Exception as e:
        st.error(f"⚠️ Forecast failed: {str(e)}")
        return None

def display_forecast(forecast, pollutant):
    pollutant_key = pollutant.lower().replace(".", "")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast['date'],
        y=forecast[f'predicted_{pollutant_key}'],
        mode='lines+markers',
        line=dict(color='#FFA15A', width=3)
    ))
    fig.update_layout(
        title=f"3-Day {pollutant} Forecast",
        xaxis_title="Date",
        yaxis_title=f"{pollutant} (µg/m³)"
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
