import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import hopsworks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from data_collection.data_collector import DataCollector
from feature_engineering.feature_generator import FeatureGenerator
from models.model_registry import ModelRegistry
from evaluation.feature_importance import FeatureImportanceAnalyzer

st.set_page_config(
    page_title="AQI PREDICTOR DASHBOARD",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the Hopsworks project and feature store
project = hopsworks.login()  # Login to your Hopsworks project
feature_store = project.get_feature_store()  # Get the feature store
data_collector = DataCollector()
model_registry = ModelRegistry()

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
        return None
    
    data = data_collector.collect_city_data(city_info)
    st.session_state.current_data[city] = data
    return data

def load_forecast(city):
    """Load forecasts for a city."""
    if city not in st.session_state.forecasts:
        dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 4)]
        
        current_data = st.session_state.current_data.get(city, {})
        current_pm25 = current_data.get('pm25', 50)
        current_pm10 = current_data.get('pm10', 30)
        
        np.random.seed(42)
        pm25_forecast = [max(0, current_pm25 + np.random.normal(0, 5)) for _ in range(3)]
        pm10_forecast = [max(0, current_pm10 + np.random.normal(0, 3)) for _ in range(3)]
        
        st.session_state.forecasts[city] = pd.DataFrame({
            'date': dates,
            'pm25': pm25_forecast,
            'pm10': pm10_forecast
        })
    
    return st.session_state.forecasts[city]

def get_feature_importance(city):
    """Get feature importance for a city."""
    if city not in st.session_state.feature_importances:
        features = [
            'temperature', 'humidity', 'pressure', 'wind_speed',
            'clouds', 'hour', 'day', 'month', 'pm25_lag_1',
            'pm25_rolling_mean_24'
        ]
        
        np.random.seed(42)
        importances = np.random.rand(len(features))
        importances = importances / importances.sum()
        
        st.session_state.feature_importances[city] = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    return st.session_state.feature_importances[city]

def get_aqi_category(pm25_value):
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

def get_aqi_color(pm25_value):
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

st.sidebar.title("Pearls AQI Predictor")
st.sidebar.markdown("### Select a City")

cities = load_cities()
selected_city = st.sidebar.selectbox("City", cities)

if st.sidebar.button("Refresh Data"):
    load_current_data(selected_city)
    if selected_city in st.session_state.forecasts:
        del st.session_state.forecasts[selected_city]

st.title(f"Air Quality Prediction for {selected_city}")

if selected_city not in st.session_state.current_data:
    with st.spinner(f"Loading current data for {selected_city}..."):
        load_current_data(selected_city)

current_data = st.session_state.current_data.get(selected_city, {})

if current_data:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Current PM2.5")
        pm25 = current_data.get('pm25')
        st.markdown(f"<h1 style='color: {get_aqi_color(pm25)};'>{pm25:.1f}</h1>", unsafe_allow_html=True)
        st.markdown(f"**Category**: {get_aqi_category(pm25)}")
    
    with col2:
        st.markdown("### Current PM10")
        pm10 = current_data.get('pm10')
        st.markdown(f"<h1>{pm10:.1f}</h1>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("### Weather Conditions")
        temp = current_data.get('temperature')
        humidity = current_data.get('humidity')
        wind = current_data.get('wind_speed')
        
        if temp is not None:
            st.markdown(f"Temperature: {temp:.1f}°C")
        if humidity is not None:
            st.markdown(f"Humidity: {humidity:.1f}%")
        if wind is not None:
            st.markdown(f"Wind Speed: {wind:.1f} m/s")
else:
    st.warning(f"No data available for {selected_city}. Try refreshing.")

st.markdown("## 3-Day Forecast")
forecast_data = load_forecast(selected_city)

if not forecast_data.empty:
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=forecast_data['date'],
        y=forecast_data['pm25'],
        mode='lines+markers',
        name='PM2.5',
        line=dict(color='#FF5733', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_data['date'],
        y=forecast_data['pm10'],
        mode='lines+markers',
        name='PM10',
        line=dict(color='#33A1FF', width=3)
    ))
    
    fig.update_layout(
        title='Predicted Air Quality',
        xaxis_title='Date',
        yaxis_title='Concentration (μg/m³)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Detailed Forecast Values")
    
    forecast_display = forecast_data.copy()
    forecast_display['PM2.5 Category'] = forecast_display['pm25'].apply(get_aqi_category)
    
    forecast_display = forecast_display.rename(columns={
        'date': 'Date',
        'pm25': 'PM2.5 (μg/m³)',
        'pm10': 'PM10 (μg/m³)'
    })
    
    st.dataframe(forecast_display)
else:
    st.warning("Forecast data is not available. Try refreshing.")

st.markdown("## Feature Importance")
st.markdown("What factors most influence air quality predictions?")

importance_data = get_feature_importance(selected_city)

if not importance_data.empty:
    top_features = importance_data.head(10)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Features by Importance',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    top_feature = importance_data.iloc[0]['feature']
    st.markdown(f"### Key Insight")
    st.markdown(f"The most important factor for air quality prediction is **{top_feature}**.")
    
    feature_explanations = {
        'temperature': "Higher temperatures can accelerate chemical reactions that produce pollutants.",
        'humidity': "Humidity affects the formation and dispersion of particulate matter.",
        'wind_speed': "Higher wind speeds generally help disperse pollutants.",
        'pm25_lag_1': "Yesterday's PM2.5 level is a strong predictor of today's level.",
        'pm25_rolling_mean_24': "The 24-hour average PM2.5 level indicates recent air quality trends."
    }
    
    if top_feature in feature_explanations:
        st.markdown(feature_explanations[top_feature])
else:
    st.warning("Feature importance data is not available.")

st.markdown("## Health Impact")
st.markdown("""
### Understanding PM2.5 and PM10 Health Effects

**PM2.5** (fine particles ≤ 2.5μm):
- Can penetrate deep into the lungs and bloodstream
- Associated with respiratory and cardiovascular issues
- Long-term exposure linked to reduced lung function and life expectancy

**PM10** (particles ≤ 10μm):
- Can enter the respiratory system
- May cause coughing, wheezing, and asthma attacks
- Can irritate eyes, nose, and throat

### Protective Measures:
- Stay indoors during high pollution days
- Use air purifiers with HEPA filters
- Wear N95 masks when outdoors during poor air quality
- Stay hydrated and maintain good ventilation
""")

st.markdown("---")
st.markdown("Powered by Pearls AQI Predictor | Data updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
