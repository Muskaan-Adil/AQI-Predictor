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
import shap
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor  # Example model, replace with yours

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

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
        'model': None,
        'explainer': None
    }
    for key, val in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# Connect to Hopsworks with robust error handling
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
        logger.info("Successfully connected to Hopsworks")
        return fs, mr
    except Exception as e:
        logger.error(f"Hopsworks connection failed: {str(e)}")
        st.error("🔴 Failed to connect to Hopsworks Feature Store")
        st.error(f"Error details: {str(e)}")
        return None, None

# Load ML model from Hopsworks model registry
def load_models(model_registry):
    try:
        # Get the latest versions of both models
        pm25_models = model_registry.get_models("Karachi_pm25")
        pm10_models = model_registry.get_models("Karachi_pm10")
        
        if not pm25_models or not pm10_models:
            raise ValueError("One or both models not found in registry")
        
        # Sort models by version and get the latest
        latest_pm25 = sorted(pm25_models, key=lambda m: m.version, reverse=True)[0]
        latest_pm10 = sorted(pm10_models, key=lambda m: m.version, reverse=True)[0]
        
        # Download and load models
        model_dir_pm25 = latest_pm25.download()
        model_pm25 = joblib.load(model_dir_pm25 + "/model.pkl")
        explainer_pm25 = shap.TreeExplainer(model_pm25)
        
        model_dir_pm10 = latest_pm10.download()
        model_pm10 = joblib.load(model_dir_pm10 + "/model.pkl")
        explainer_pm10 = shap.TreeExplainer(model_pm10)
        
        # Store in session state with version info
        st.session_state.update({
            'model_pm25': model_pm25,
            'model_pm10': model_pm10,
            'explainer_pm25': explainer_pm25,
            'explainer_pm10': explainer_pm10,
            'model_versions': {
                'pm25': latest_pm25.version,
                'pm10': latest_pm10.version
            }
        })
        
        logger.info(f"Loaded PM2.5 model v{latest_pm25.version} and PM10 model v{latest_pm10.version}")
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        st.error("⚠️ Could not load models from registry")
        return False

# Load available cities with multiple fallback strategies
def load_available_cities(fs):
    try:
        # Try direct feature view access
        fv = fs.get_feature_view("aqi_features", 1)
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

# Get most recent data for a city with comprehensive validation
def get_city_data(fs, city):
    try:
        fv = fs.get_feature_view("aqi_features", 1)
        
        # Method 1: Optimized Hopsworks query
        try:
            query = fv.select_all().filter(fv.city == city)
            city_data = query.read()
        except:
            # Method 2: Fallback to pandas filtering
            all_data = fv.get_batch_data()
            if not isinstance(all_data, pd.DataFrame):
                all_data = all_data.to_pandas()
            city_data = all_data[all_data['city'] == city]
        
        # Validation checks
        if city_data.empty:
            raise ValueError(f"No records found for {city}")
            
        if 'date' not in city_data.columns:
            raise ValueError("'date' column missing")
            
        # Get most recent record
        latest = city_data.sort_values('date', ascending=False).iloc[0]
        
        # Validate all required fields
        required_fields = {
            'pm25': float,
            'pm10': float,
            'temperature': float,
            'humidity': float,
            'wind_speed': float,
            'pressure': float,
            'precipitation': float
        }
        
        result = {
            'last_updated': latest['date'].strftime("%Y-%m-%d %H:%M:%S"),
            'features': {}
        }
        
        for field, dtype in required_fields.items():
            if field not in latest:
                raise ValueError(f"Missing field: {field}")
            result['features'][field] = dtype(latest[field])
            result[field] = dtype(latest[field])  # Also keep in root for backward compat
        
        return result
        
    except Exception as e:
        logger.error(f"Data loading failed for {city}: {str(e)}")
        st.error(f"⚠️ Could not load data for {city}")
        return None

# Generate forecast with SHAP explanations
def generate_forecast(current_data, pollutant):
    try:
        if st.session_state.model is None or st.session_state.explainer is None:
            raise ValueError("Model or explainer not loaded")
            
        # Prepare features for prediction
        feature_names = ['temperature', 'humidity', 'wind_speed', 'pressure', 'precipitation']
        feature_values = [current_data['features'][f] for f in feature_names]
        X = pd.DataFrame([feature_values], columns=feature_names)
        
        # Generate forecast (next 3 days)
        dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") 
                for i in range(1, 4)]
        
        # Predict for each day (in reality you might have different features for future days)
        predictions = []
        shap_values_list = []
        
        for _ in range(3):
            # Predict (replace with your actual forecasting logic)
            pred = st.session_state.model.predict(X)[0]
            predictions.append(pred)
            
            # Calculate SHAP values
            shap_values = st.session_state.explainer.shap_values(X)
            shap_values_list.append(shap_values)
            
            # For demo, modify features slightly for next prediction
            X['temperature'] *= 1.02
            X['humidity'] *= 0.98
        
        # Store SHAP values for the first prediction
        st.session_state.shap_values = {
            'values': shap_values_list[0][0],  # First prediction's SHAP values
            'features': feature_names,
            'feature_values': feature_values,
            'base_value': st.session_state.explainer.expected_value
        }
        
        return pd.DataFrame({
            'date': dates,
            f'predicted_{pollutant.lower().replace(".", "")}': predictions
        })
        
    except Exception as e:
        logger.error(f"Forecast failed: {str(e)}")
        st.error("⚠️ Forecast generation failed")
        return None

# Display SHAP explanation plot
def display_shap_explanations():
    if st.session_state.shap_values is None:
        return
        
    st.subheader("Feature Importance Explanation (SHAP Values)")
    
    # Create SHAP explanation object
    shap_values = st.session_state.shap_values['values']
    features = st.session_state.shap_values['features']
    feature_values = st.session_state.shap_values['feature_values']
    base_value = st.session_state.shap_values['base_value']
    
    # Create SHAP explanation object
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=feature_values,
        feature_names=features
    )
    
    # Plot using matplotlib
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation[0], max_display=10, show=False)
    plt.tight_layout()
    
    # Display in Streamlit
    st.pyplot(fig)
    
    # Add interpretation help
    with st.expander("How to interpret SHAP values"):
        st.markdown("""
        - **Blue bars**: Features reducing the prediction
        - **Red bars**: Features increasing the prediction
        - **Length**: Magnitude of the feature's impact
        - **Base value**: Average model output
        - **f(x)**: Final prediction value
        """)

# UI Components
def display_current_metrics(city, data):
    # Add model version info to the display
    st.subheader(f"Current Air Quality in {city}")
    
    if 'model_versions' in st.session_state:
        st.caption(f"Using models: PM2.5 v{st.session_state.model_versions['pm25']}, PM10 v{st.session_state.model_versions['pm10']}")

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
    
    # Display SHAP explanations after forecast
    display_shap_explanations()

# Main App
def main():
    init_session_state()
    
    # Page config
    st.set_page_config(
        page_title="AQI Prediction Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Connect to Hopsworks
    if not st.session_state.fs_connected:
        with st.spinner("Connecting to Feature Store..."):
            fs, mr = connect_feature_store()
            if not fs:
                st.stop()
            
            # Load model after connecting
            with st.spinner("Loading prediction model..."):
                load_model(mr)
    
    # Load cities
    if not st.session_state.cities:
        with st.spinner("Loading available cities..."):
            st.session_state.cities = load_available_cities(st.session_state.feature_store)
            if not st.session_state.cities:
                st.error("No cities available - check feature store")
                st.stop()
    
    # City selection
    selected_city = st.sidebar.selectbox(
        "Select City", 
        st.session_state.cities,
        key='city_select'
    )
    
    # Pollutant selection
    st.session_state.pollutant = st.sidebar.selectbox(
        "Select Pollutant",
        ['PM2.5', 'PM10'],
        key='pollutant_select'
    )
    
    # Refresh button
    if st.sidebar.button("Refresh Data"):
        st.session_state.current_data.pop(selected_city, None)
        st.session_state.forecasts.pop(selected_city, None)
        st.session_state.shap_values = None
        st.rerun()
    
    # Main content
    st.title(f"Air Quality Dashboard: {selected_city}")
    
    # Load current data
    if selected_city not in st.session_state.current_data:
        with st.spinner(f"Loading {selected_city} data..."):
            st.session_state.current_data[selected_city] = get_city_data(
                st.session_state.feature_store,
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
        display_forecast(forecast, st.session_state.pollutant)
    else:
        st.error(f"Failed to load data for {selected_city}")

if __name__ == "__main__":
    main()
