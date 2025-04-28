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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Streamlit page config
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    if 'feature_store' not in st.session_state:
        st.session_state.feature_store = None
    if 'cities' not in st.session_state:
        st.session_state.cities = []
    if 'current_data' not in st.session_state:
        st.session_state.current_data = {}
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = {}

# Connect to Hopsworks
def connect_to_hopsworks():
    try:
        project = hopsworks.login(
            project=Config.HOPSWORKS_PROJECT_NAME,
            api_key_value=Config.HOPSWORKS_API_KEY,
            host=Config.HOPSWORKS_HOST,
            port=443,
            hostname_verification=False
        )
        st.session_state.feature_store = project.get_feature_store()
        st.success("✅ Connected to Hopsworks")
        return True
    except Exception as e:
        st.error(f"❌ Hopsworks connection failed: {str(e)}")
        return False

# Load cities from feature store
def load_cities():
    try:
        feature_view = st.session_state.feature_store.get_feature_view(
            name="karachi_aqi_features", 
            version=1
        )
        feature_data = feature_view.get_batch_data()
        if not isinstance(feature_data, pd.DataFrame):
            feature_data = feature_data.to_pandas()
        return feature_data["city"].unique().tolist()
    except Exception as e:
        st.error(f"Failed to load cities: {str(e)}")
        return []

# Load most recent data for a city
def load_current_data(city):
    try:
        feature_view = st.session_state.feature_store.get_feature_view(
            name="karachi_aqi_features", 
            version=1
        )
        feature_data = feature_view.get_batch_data()
        if not isinstance(feature_data, pd.DataFrame):
            feature_data = feature_data.to_pandas()
        
        # Get most recent record for the city
        city_data = feature_data[feature_data["city"] == city]
        if city_data.empty:
            return None
            
        latest_record = city_data.sort_values("date", ascending=False).iloc[0]
        
        return {
            'pm25': latest_record['pm25'],
            'pm10': latest_record['pm10'],
            'temperature': latest_record['temperature'],
            'humidity': latest_record['humidity'],
            'wind_speed': latest_record['wind_speed'],
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        st.error(f"Failed to load data for {city}: {str(e)}")
        return None

# Generate 3-day forecast (mock function - replace with your actual model)
def generate_forecast(city_data, pollutant):
    try:
        # This is a mock forecast - replace with your actual model prediction
        base_value = city_data[f'pm{pollutant.replace(".", "")}']
        dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 4)]
        
        # Simple trend: slight random variation around current value
        forecast_values = [base_value * (1 + np.random.uniform(-0.1, 0.2)) for _ in range(3)]
        
        return pd.DataFrame({
            'date': dates,
            f'predicted_{pollutant.replace(".", "")}': forecast_values
        })
    except Exception as e:
        st.error(f"Forecast generation failed: {str(e)}")
        return None

# Display current metrics
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

# Display forecast chart
def display_forecast_chart(forecast, pollutant):
    fig = go.Figure()
    
    pollutant_col = f'predicted_{pollutant.replace(".", "")}'
    
    fig.add_trace(go.Scatter(
        x=forecast['date'],
        y=forecast[pollutant_col],
        mode='lines+markers',
        name=f"Forecasted {pollutant}",
        line=dict(color="#636EFA", width=3)
    ))
    
    fig.update_layout(
        title=f"3-Day {pollutant} Forecast",
        xaxis_title="Date",
        yaxis_title=f"{pollutant} (µg/m³)",
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="#f9f9f9"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Main app function
def main():
    init_session_state()
    
    # Connect to Hopsworks
    if st.session_state.feature_store is None:
        with st.spinner("Connecting to Hopsworks..."):
            if not connect_to_hopsworks():
                st.stop()
    
    # Load cities
    if not st.session_state.cities:
        with st.spinner("Loading cities..."):
            st.session_state.cities = load_cities()
            if not st.session_state.cities:
                st.error("No cities found in feature store")
                st.stop()
    
    # Sidebar controls
    st.sidebar.title("AQI Predictor")
    selected_city = st.sidebar.selectbox("Select City", st.session_state.cities)
    selected_pollutant = st.sidebar.selectbox("Predict", ["PM2.5", "PM10"])
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.session_state.current_data.pop(selected_city, None)
        st.session_state.forecasts.pop(selected_city, None)
        st.rerun()
    
    # Main content
    st.title(f"Air Quality Dashboard: {selected_city}")
    
    # Load current data if not already loaded
    if selected_city not in st.session_state.current_data:
        with st.spinner(f"Loading current data for {selected_city}..."):
            st.session_state.current_data[selected_city] = load_current_data(selected_city)
    
    current_data = st.session_state.current_data.get(selected_city)
    
    if current_data:
        display_current_metrics(selected_city, current_data)
        
        # Generate forecast if not already generated
        if selected_city not in st.session_state.forecasts:
            with st.spinner("Generating forecast..."):
                st.session_state.forecasts[selected_city] = generate_forecast(
                    current_data, 
                    selected_pollutant
                )
        
        forecast = st.session_state.forecasts.get(selected_city)
        
        if forecast is not None:
            display_forecast_chart(forecast, selected_pollutant)
        else:
            st.warning("Could not generate forecast")
    else:
        st.error(f"Could not load data for {selected_city}")

if __name__ == "__main__":
    main()
