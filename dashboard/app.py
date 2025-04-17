import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.config import Config
from src.data_collection.data_collector import DataCollector
from src.feature_engineering.feature_generator import FeatureGenerator
from src.evaluation.model_selector import ModelSelector
from src.evaluation.feature_importance import FeatureImportanceAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

data_collector = DataCollector()
feature_generator = FeatureGenerator()
model_selector = ModelSelector()
feature_analyzer = FeatureImportanceAnalyzer()

st.set_page_config(
    page_title=Config.DASHBOARD_TITLE,
    page_icon="🌬️",
    layout="wide"
)

st.title("🌬️ Pearls AQI Predictor")
st.markdown("### Predicting Air Quality Index (PM2.5/PM10) for the next 3 days")

city_options = [city['name'] for city in Config.CITIES]
selected_city = st.sidebar.selectbox("Select a City", city_options)

selected_city_data = next((city for city in Config.CITIES if city['name'] == selected_city), None)

target_options = ['pm25', 'pm10']
selected_target = st.sidebar.radio("Select Pollutant to Predict", target_options)

if st.sidebar.button("Refresh Data"):
    st.sidebar.success("Data refreshed!")

@st.cache_data(ttl=3600)
def get_current_data(city):
    """Get current air quality data for a city.
    
    Args:
        city (dict): City dictionary with name, lat, lon.
        
    Returns:
        dict: Current data or empty dict if retrieval fails.
    """
    try:
        data = data_collector.collect_city_data(city)
        return data
    except Exception as e:
        logger.error(f"Error fetching current data: {e}")
        return {}

@st.cache_data(ttl=3600)
def generate_forecast(city, target_col, days=3):
    """Generate forecast for a city.
    
    Args:
        city (dict): City dictionary with name, lat, lon.
        target_col (str): Target column to forecast.
        days (int, optional): Number of days to forecast. Defaults to 3.
        
    Returns:
        pd.DataFrame: Forecast data or None if forecasting fails.
    """
    try:
        current_data = get_current_data(city)
        if not current_data:
            return None
        
        current_df = pd.DataFrame([current_data])
        
        feature_df = feature_generator.generate_all_features(current_df)
        
        model = model_selector.get_best_model(city['name'], target_col)
        
        if model is None:
            logger.warning(f"No model available for {city['name']}, {target_col}")
            return None
        
        future_dates = [datetime.now() + timedelta(days=i) for i in range(1, days+1)]
        
        future_data = []
        for date in future_dates:
            future_row = feature_df.iloc[-1].copy()
            
            future_row['timestamp'] = date
            future_row['hour'] = date.hour
            future_row['day'] = date.day
            future_row['month'] = date.month
            future_row['year'] = date.year
            future_row['dayofweek'] = date.weekday()
            future_row['is_weekend'] = 1 if date.weekday() >= 5 else 0
            future_row['quarter'] = (date.month - 1) // 3 + 1
            future_row['dayofyear'] = date.timetuple().tm_yday
            future_row['weekofyear'] = date.isocalendar()[1]
            
            future_data.append(future_row)
        
        future_df = pd.DataFrame(future_data)
        
        future_df[target_col] = model.predict(future_df)
        
        future_df['date'] = future_df['timestamp'].dt.date if pd.api.types.is_datetime64_any_dtype(future_df['timestamp']) else pd.to_datetime(future_df['timestamp']).dt.date
        
        return future_df
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        return None

@st.cache_data(ttl=86400)
def get_historical_data(city, days=30):
    """Get historical data for a city.
    
    Args:
        city (dict): City dictionary with name, lat, lon.
        days (int, optional): Number of days of history. Defaults to 30.
        
    Returns:
        pd.DataFrame: Historical data or None if retrieval fails.
    """
    try:
        historical_df = data_collector.collect_historical_data(city, days)
        return historical_df
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return None

@st.cache_data(ttl=86400)
def get_feature_importance(city, target_col):
    """Get feature importance for a city and target.
    
    Args:
        city (str): City name.
        target_col (str): Target column.
        
    Returns:
        pd.DataFrame: Feature importance or None if retrieval fails.
    """
    try:
        model = model_selector.get_best_model(city, target_col)
        
        if model is None:
            return None
        
        city_dict = next((c for c in Config.CITIES if c['name'] == city), None)
        if not city_dict:
            return None
        
        historical_df = get_historical_data(city_dict, days=7)
        
        if historical_df is None or historical_df.empty:
            return None
        
        feature_df = feature_generator.generate_all_features(historical_df)
        
        importance_df = feature_analyzer.get_model_feature_importance(model, feature_df)
        
        if importance_df is None:
            shap_values, X_sample = feature_analyzer.get_shap_values(model, feature_df)
            if shap_values is not None:
                importance_df = feature_analyzer.get_top_features(shap_values, X_sample)
        
        return importance_df
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        return None

try:
    if selected_city_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Conditions")
            
            current_data = get_current_data(selected_city_data)
            
            if current_data:
                pm25 = current_data.get('pm25')
                pm10 = current_data.get('pm10')
                
                if pm25 is not None:
                    st.metric("PM2.5", f"{pm25:.1f} µg/m³")
                
                if pm10 is not None:
                    st.metric("PM10", f"{pm10:.1f} µg/m³")
                
                temp = current_data.get('temperature')
                humidity = current_data.get('humidity')
                
                weather_cols = st.columns(2)
                with weather_cols[0]:
                    if temp is not None:
                        st.metric("Temperature", f"{temp:.1f} °C")
                
                with weather_cols[1]:
                    if humidity is not None:
                        st.metric("Humidity", f"{humidity:.1f}%")
                
                timestamp = current_data.get('timestamp')
                if timestamp:
                    st.caption(f"Last updated: {pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.warning("No current data available for this city")
        
        with col2:
            st.subheader(f"{selected_target.upper()} Forecast (Next 3 Days)")
            
            forecast_df = generate_forecast(selected_city_data, selected_target)
            
            if forecast_df is not None and not forecast_df.empty:
                fig = px.line(
                    forecast_df, 
                    x='date', 
                    y=selected_target,
                    markers=True,
                    labels={
                        'date': 'Date',
                        selected_target: f"{selected_target.upper()} (µg/m³)"
                    },
                    title=f"{selected_target.upper()} Forecast"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(
                    forecast_df[['date', selected_target]].set_index('date'),
                    use_container_width=True
                )
            else:
                st.warning("Unable to generate forecast. This could be due to missing models or data.")
        
        st.subheader("Historical Data")
        
        historical_df = get_historical_data(selected_city_data)
        
        if historical_df is not None and not historical_df.empty:
            if 'date' in historical_df.columns:
                historical_df['date'] = pd.to_datetime(historical_df['date'])
            elif 'timestamp' in historical_df.columns:
                historical_df['date'] = pd.to_datetime(historical_df['timestamp'])
            
            if selected_target in historical_df.columns:
                fig = px.line(
                    historical_df, 
                    x='date', 
                    y=selected_target,
                    labels={
                        'date': 'Date',
                        selected_target: f"{selected_target.upper()} (µg/m³)"
                    },
                    title=f"Historical {selected_target.upper()}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No historical {selected_target} data available for this city")
        else:
            st.warning("No historical data available for this city")
        
        st.subheader("Feature Importance")
        
        importance_df = get_feature_importance(selected_city_data['name'], selected_target)
        
        if importance_df is not None and not importance_df.empty:
            fig = px.bar(
                importance_df.head(10), 
                x='importance', 
                y='feature',
                orientation='h',
                labels={
                    'importance': 'Importance',
                    'feature': 'Feature'
                },
                title=f"Top 10 Features for {selected_target.upper()} Prediction"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance data not available")
    else:
        st.error("City not found in configuration")

except Exception as e:
    st.error(f"An error occurred: {e}")
    logger.exception("Dashboard error")

st.sidebar.subheader("About")
st.sidebar.info(
    """
    This dashboard predicts air quality (PM2.5 and PM10) for different cities using 
    machine learning models trained on historical data. The predictions are based on 
    weather conditions and historical pollution patterns.
    """
)

st.sidebar.subheader("AQI Alert Thresholds")
alert_df = pd.DataFrame({
    "Level": ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"],
    "PM2.5 (µg/m³)": ["0-12", "12.1-35.4", "35.5-55.4", "55.5-150.4", "150.5-250.4", ">250.5"],
    "PM10 (µg/m³)": ["0-54", "55-154", "155-254", "255-354", "355-424", ">425"]
})
st.sidebar.dataframe(alert_df, use_container_width=True)

if __name__ == "__main__":
    pass
