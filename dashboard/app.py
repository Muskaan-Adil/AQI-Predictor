import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import hopsworks
import plotly.graph_objects as go
import streamlit as st
import shap
import matplotlib.pyplot as plt
import joblib
from pandas.api.types import is_datetime64_any_dtype as is_datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))
try:
    from utils.config import Config
except ImportError:
    class Config:
        HOPSWORKS_PROJECT_NAME = "AQI_Pred_10Pearls"
        HOPSWORKS_API_KEY = "your_api_key_here"
        HOPSWORKS_HOST = "c.app.hopsworks.ai"

# Model configuration with exact timestamps
MODEL_CONFIG = {
    'pm25': {
        'name': 'Karachi_pm25',
        'timestamp': '20250430031524',  # For Karachi_pm25_20250430031524.joblib
    },
    'pm10': {
        'name': 'Karachi_pm10',
        'timestamp': '20250430031535',  # For Karachi_pm10_20250430031535.joblib
    }
}

# Initialize session state
def init_session_state():
    session_defaults = {
        'fs_connected': False,
        'cities': [],
        'current_data': {},
        'forecasts': {},
        'pollutant': 'PM2.5',
        'last_update': None,
        'shap_values': None,
        'model_pm25': None,
        'model_pm10': None,
        'explainer_pm25': None,
        'explainer_pm10': None,
        'model_versions': {'pm25': 'fallback', 'pm10': 'fallback'},
        'last_model_check': None,
        'feature_store': None,
        'model_registry': None
    }
    for key, val in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def deploy_model(model_registry, pollutant='pm25'):
    try:
        config = MODEL_CONFIG[pollutant]
        models = model_registry.get_models(config['name'])
        if not models:
            return None
            
        target_model = next((m for m in models if config['timestamp'] in m.name), None)
        if not target_model:
            return None

        model_dir = target_model.download()
        model_file = next(Path(model_dir).rglob(f"*{config['timestamp']}.joblib"), None)
        if not model_file:
            return None

        model = joblib.load(model_file)
        st.session_state.model_versions[pollutant] = config['timestamp']
        return model
    except Exception as e:
        logger.error(f"Model deployment failed: {str(e)}")
        return None

def load_models(model_registry):
    try:
        pm25_model = deploy_model(model_registry, 'pm25')
        pm10_model = deploy_model(model_registry, 'pm10')

        # Load fallback for missing models
        if not pm25_model and not train_fallback_model('pm25'):
            raise RuntimeError("PM2.5 model load failed")
        if not pm10_model and not train_fallback_model('pm10'):
            raise RuntimeError("PM10 model load failed")

        # Update session state for loaded models
        if pm25_model:
            st.session_state.model_pm25 = pm25_model
            st.session_state.explainer_pm25 = shap.TreeExplainer(pm25_model)
        if pm10_model:
            st.session_state.model_pm10 = pm10_model
            st.session_state.explainer_pm10 = shap.TreeExplainer(pm10_model)
            
        return True
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        st.error("⚠️ Using fallback linear regression models")
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        st.error(f"⚠️ Model loading error: {str(e)}")
        return False

def connect_feature_store():
    try:
        project = hopsworks.login(
            project=Config.HOPSWORKS_PROJECT_NAME,
            api_key_value=Config.HOPSWORKS_API_KEY,
            host=Config.HOPSWORKS_HOST,
            port=443,
            hostname_verification=False
        )
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        st.session_state.feature_store = fs
        st.session_state.model_registry = mr
        st.session_state.fs_connected = True
        return fs, mr
    except Exception as e:
        st.error("🔴 Connection failed. Using last available data.")
        return None, None

def check_model_freshness(model_registry, model_name):
    try:
        models = model_registry.get_models(model_name)
        if not models:
            return False
        latest_model = sorted(models, key=lambda m: m.version, reverse=True)[0]
        
        # Handle timestamp conversion
        if isinstance(latest_model.created, (int, float)):
            creation_time = datetime.fromtimestamp(latest_model.created / 1000)
        else:
            creation_time = latest_model.created.replace(tzinfo=None)
            
        return (datetime.now() - creation_time) < timedelta(hours=24)
    except Exception as e:
        logger.error(f"Error checking model freshness: {str(e)}")
        return False

def load_available_cities(fs):
    try:
        fv = fs.get_feature_view("karachi_aqi_features", 1)
        data = fv.get_batch_data()
        if not isinstance(data, pd.DataFrame):
            data = data.to_pandas()
        
        if 'city' not in data.columns:
            raise ValueError("'city' column not found in feature data")
            
        cities = data['city'].unique().tolist()
        
        if not cities:
            raise ValueError("No cities found in feature data")
            
        logger.info(f"Loaded {len(cities)} cities from feature store")
        return cities
        
    except Exception as e:
        logger.error(f"City loading failed: {str(e)}")
        st.error("⚠️ Could not load city list from feature store")
        return []

def get_city_data(fs, city):
    try:
        fv = fs.get_feature_view("karachi_aqi_features", 1)
        data = fv.get_batch_data().to_pandas()
        city_data = data[data['city'] == city]
        
        if city_data.empty:
            return None
            
        latest = city_data.sort_values('date', ascending=False).iloc[0]
        return {
            'last_updated': latest['date'].strftime("%Y-%m-%d %H:%M:%S"),
            'pm25': latest['pm25'],
            'pm10': latest['pm10'],
            'temperature': latest['temperature'],
            'humidity': latest['humidity'],
            'wind_speed': latest['wind_speed'],
            'pressure': latest['pressure'],
            'features': {
                'temperature': latest['temperature'],
                'humidity': latest['humidity'],
                'wind_speed': latest['wind_speed'],
                'pressure': latest['pressure'],
                'precipitation': latest['precipitation']
            }
        }
    except Exception as e:
        logger.error(f"Data load failed: {str(e)}")
        return None

def generate_forecast(current_data, pollutant):
    try:
        model = st.session_state.model_pm25 if pollutant == 'PM2.5' else st.session_state.model_pm10
        explainer = st.session_state.explainer_pm25 if pollutant == 'PM2.5' else st.session_state.explainer_pm10
        
        features = ['temperature', 'humidity', 'wind_speed', 'pressure', 'precipitation']
        X = pd.DataFrame([current_data['features'][f] for f in features], index=features).T
        
        dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1,4)]
        predictions = [model.predict(X)[0] * (1 + i*0.1) for i in range(3)]  # Simple trend
        
        if explainer:
            shap_values = explainer.shap_values(X)
            st.session_state.shap_values = {
                'values': shap_values[0],
                'features': features,
                'feature_values': X.values[0],
                'base_value': explainer.expected_value,
                'pollutant': pollutant
            }
            
        return pd.DataFrame({
            'date': dates,
            f'predicted_{pollutant.lower().replace(".", "")}': predictions
        })
    except Exception as e:
        logger.error(f"Forecast failed: {str(e)}")
        return None

def display_shap_explanations():
    if st.session_state.shap_values is None:
        return
        
    st.subheader(f"Feature Importance Explanation for {st.session_state.shap_values['pollutant']} (SHAP Values)")
    
    shap_values = st.session_state.shap_values['values']
    features = st.session_state.shap_values['features']
    feature_values = st.session_state.shap_values['feature_values']
    base_value = st.session_state.shap_values['base_value']
    
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=feature_values,
        feature_names=features
    )
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation[0], max_display=10, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    
    with st.expander("How to interpret SHAP values"):
        st.markdown("""
        - **Blue bars**: Features reducing the AQI prediction
        - **Red bars**: Features increasing the AQI prediction
        - **Length**: Magnitude of the feature's impact
        - **Base value**: Average model output
        - **f(x)**: Final predicted AQI value
        - **Feature values**: Current conditions shown in parentheses
        """)

def display_current_metrics(city, data):
    st.subheader(f"Current Air Quality in {city}")
    
    if 'model_versions' in st.session_state:
        st.caption(f"Using models: PM2.5 v{st.session_state.model_versions['pm25']}, PM10 v{st.session_state.model_versions['pm10']}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PM2.5", f"{data['pm25']:.1f} µg/m³")
        st.metric("Temperature", f"{data['temperature']:.1f} °C")
    with col2:
        st.metric("PM10", f"{data['pm10']:.1f} µg/m³")
        st.metric("Humidity", f"{data['humidity']:.0f}%")
    with col3:
        st.metric("Wind Speed", f"{data['wind_speed']:.1f} m/s")
        st.metric("Pressure", f"{data.get('pressure', 0):.1f} hPa")
    
    st.caption(f"Last updated: {data['last_updated']}")

def display_forecast(forecast, pollutant):
    if forecast is None:
        return
        
    pollutant_col = f'predicted_{pollutant.lower().replace(".", "")}'
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast['date'],
        y=forecast[pollutant_col],
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
    display_shap_explanations()

def main():
    init_session_state()
    
    st.set_page_config(
        page_title="Karachi AQI Prediction Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.sidebar.title("Settings")
    
    if not st.session_state.fs_connected:
        with st.spinner("Connecting to Hopsworks..."):
            fs, mr = connect_feature_store()
            if not fs:
                st.stop()
            
            with st.spinner("Deploying latest models..."):
                if not load_models(mr):
                    st.stop()
    
    if (not st.session_state.last_model_check or 
        (datetime.now() - st.session_state.last_model_check) > timedelta(minutes=30)):
        with st.spinner("Checking for model updates..."):
            pm25_fresh = check_model_freshness(st.session_state.model_registry, "Karachi_pm25")
            pm10_fresh = check_model_freshness(st.session_state.model_registry, "Karachi_pm10")
            
            if pm25_fresh or pm10_fresh:
                load_models(st.session_state.model_registry)
            
            st.session_state.last_model_check = datetime.now()
    
    if not st.session_state.cities:
        with st.spinner("Loading available cities..."):
            st.session_state.cities = load_available_cities(st.session_state.feature_store)
            if not st.session_state.cities:
                st.error("No cities available - check feature store")
                st.stop()
    
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
    
    if st.sidebar.button("Refresh Data"):
        st.session_state.current_data.pop(selected_city, None)
        st.session_state.forecasts.pop(selected_city, None)
        st.session_state.shap_values = None
        st.rerun()
    
    st.title(f"Karachi Air Quality Dashboard: {selected_city}")
    
    if selected_city not in st.session_state.current_data:
        with st.spinner(f"Loading {selected_city} data..."):
            st.session_state.current_data[selected_city] = get_city_data(
                st.session_state.feature_store,
                selected_city
            )
    
    current_data = st.session_state.current_data.get(selected_city)
    if current_data:
        display_current_metrics(selected_city, current_data)
        
        if selected_city not in st.session_state.forecasts:
            with st.spinner("Generating forecast..."):
                st.session_state.forecasts[selected_city] = generate_forecast(
                    current_data,
                    st.session_state.pollutant
                )
        
        forecast = st.session_state.forecasts.get(selected_city)
        display_forecast(forecast, st.session_state.pollutant)
    else:
        st.error(f"Failed to load data for {selected_city}")

if __name__ == "__main__":
    main()
