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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
        HOPSWORKS_API_KEY = "ouFZ8BhcXFbDQy7S.GDkj3eGXwA4BgwzKSWqeEi53jUsd1fYSf22pxCnqG0tBZTM9RSTE2z1T64N7SErS"
        HOPSWORKS_HOST = "c.app.hopsworks.ai"

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
        'model_versions': {'pm25': None, 'pm10': None},
        'last_model_check': None,
        'feature_store': None,
        'model_registry': None
    }
    for key, val in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def deploy_model(feature_store, model_registry, pollutant='pm25'):
    try:
        st.info(f"⚙️ Deploying new {pollutant} model...")
        
        # 1. Get training data
        fv = feature_store.get_feature_view("karachi_aqi_features", 1)
        df = fv.get_batch_data()
        if not isinstance(df, pd.DataFrame):
            df = df.to_pandas()
        
        # 2. Prepare data
        features = df[['temperature', 'humidity', 'wind_speed', 'pressure', 'precipitation']]
        target = df[pollutant.lower()]
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        # 3. Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 4. Evaluate
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        
        # 5. Save model
        model_dir = f"{pollutant}_model"
        Path(model_dir).mkdir(exist_ok=True)
        joblib.dump(model, f"{model_dir}/model.pkl")
        
        # 6. Register model
        model_spec = {
            "name": f"Karachi_{pollutant}",
            "metrics": {"rmse": rmse},
            "description": f"Random Forest model for {pollutant} prediction",
            "input_example": X_train.iloc[[0]].to_dict()
        }
        model_registry.python.create_model(**model_spec)
        
        # 7. Upload model
        model_registry.get_model(f"Karachi_{pollutant}", version=1).upload(model_dir)
        
        st.success(f"✅ Successfully deployed {pollutant} model (RMSE: {rmse:.2f})!")
        return True
    except Exception as e:
        st.error(f"🚨 Model deployment failed: {str(e)}")
        logger.error(f"Model deployment error: {str(e)}")
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
        logger.info("Successfully connected to Hopsworks")
        return fs, mr
    except Exception as e:
        logger.error(f"Hopsworks connection failed: {str(e)}")
        st.error("🔴 Failed to connect to Hopsworks Feature Store")
        st.error(f"Error details: {str(e)}")
        return None, None

def check_model_freshness(model_registry, model_name):
    try:
        models = model_registry.get_models(model_name)
        if not models:
            return False
        latest_model = sorted(models, key=lambda m: m.version, reverse=True)[0]
        creation_time = latest_model.created.replace(tzinfo=None)
        return (datetime.now() - creation_time) < timedelta(hours=24)
    except Exception as e:
        logger.error(f"Error checking model freshness: {str(e)}")
        return False

def load_models(model_registry, feature_store):
    try:
        # Try to load existing models
        pm25_models = model_registry.get_models("Karachi_pm25")
        pm10_models = model_registry.get_models("Karachi_pm10")
        
        # Deploy if missing
        if not pm25_models:
            if not deploy_model(feature_store, model_registry, 'pm25'):
                return False
            pm25_models = model_registry.get_models("Karachi_pm25")
            
        if not pm10_models:
            if not deploy_model(feature_store, model_registry, 'pm10'):
                return False
            pm10_models = model_registry.get_models("Karachi_pm10")
        
        # Get latest versions
        latest_pm25 = sorted(pm25_models, key=lambda m: m.version, reverse=True)[0]
        latest_pm10 = sorted(pm10_models, key=lambda m: m.version, reverse=True)[0]
        
        # Check if update needed
        current_versions = st.session_state.model_versions
        new_versions = {
            'pm25': latest_pm25.version,
            'pm10': latest_pm10.version
        }
        
        # Load PM2.5 model if needed
        if current_versions['pm25'] != new_versions['pm25'] or not st.session_state.model_pm25:
            model_dir = latest_pm25.download()
            try:
                model = joblib.load(f"{model_dir}/artifacts/model.pkl")
            except FileNotFoundError:
                model = joblib.load(f"{model_dir}/model.pkl")
                
            st.session_state.model_pm25 = model
            st.session_state.explainer_pm25 = shap.TreeExplainer(model)
            st.session_state.model_versions['pm25'] = latest_pm25.version
            logger.info(f"Loaded PM2.5 model v{latest_pm25.version}")
        
        # Load PM10 model if needed
        if current_versions['pm10'] != new_versions['pm10'] or not st.session_state.model_pm10:
            model_dir = latest_pm10.download()
            try:
                model = joblib.load(f"{model_dir}/artifacts/model.pkl")
            except FileNotFoundError:
                model = joblib.load(f"{model_dir}/model.pkl")
                
            st.session_state.model_pm10 = model
            st.session_state.explainer_pm10 = shap.TreeExplainer(model)
            st.session_state.model_versions['pm10'] = latest_pm10.version
            logger.info(f"Loaded PM10 model v{latest_pm10.version}")
        
        # Notify if updated
        if current_versions != new_versions:
            st.toast("New model versions loaded! Forecasts will be regenerated.", icon="🔄")
            st.session_state.forecasts = {}
            
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        st.error("⚠️ Could not load models from registry")
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
        
        try:
            query = fv.select_all().filter(fv.city == city)
            city_data = query.read()
        except:
            all_data = fv.get_batch_data()
            if not isinstance(all_data, pd.DataFrame):
                all_data = all_data.to_pandas()
            city_data = all_data[all_data['city'] == city]
        
        if city_data.empty:
            raise ValueError(f"No records found for {city}")
            
        if 'date' not in city_data.columns:
            raise ValueError("'date' column missing")
            
        latest = city_data.sort_values('date', ascending=False).iloc[0]
        
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
            result[field] = dtype(latest[field])
        
        return result
        
    except Exception as e:
        logger.error(f"Data loading failed for {city}: {str(e)}")
        st.error(f"⚠️ Could not load data for {city}")
        return None

def generate_forecast(current_data, pollutant):
    try:
        if pollutant == 'PM2.5':
            model = st.session_state.model_pm25
            explainer = st.session_state.explainer_pm25
        else:
            model = st.session_state.model_pm10
            explainer = st.session_state.explainer_pm10
            
        if model is None or explainer is None:
            raise ValueError("Model or explainer not loaded")
            
        feature_names = ['temperature', 'humidity', 'wind_speed', 'pressure', 'precipitation']
        feature_values = [current_data['features'][f] for f in feature_names]
        X = pd.DataFrame([feature_values], columns=feature_names)
        
        dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") 
                for i in range(1, 4)]
        
        predictions = []
        shap_values_list = []
        
        for i in range(3):
            pred = model.predict(X)[0]
            predictions.append(pred)
            shap_values = explainer.shap_values(X)
            shap_values_list.append(shap_values)
            
            if i == 0:
                X['temperature'] *= 1.02
                X['humidity'] *= 0.98
            elif i == 1:
                X['wind_speed'] *= 1.1
        
        st.session_state.shap_values = {
            'values': shap_values_list[0][0],
            'features': feature_names,
            'feature_values': feature_values,
            'base_value': explainer.expected_value,
            'pollutant': pollutant
        }
        
        return pd.DataFrame({
            'date': dates,
            f'predicted_{pollutant.lower().replace(".", "")}': predictions
        })
        
    except Exception as e:
        logger.error(f"Forecast failed: {str(e)}")
        st.error("⚠️ Forecast generation failed")
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
        with st.spinner("Connecting to Feature Store..."):
            fs, mr = connect_feature_store()
            if not fs:
                st.stop()
            
            with st.spinner("Loading prediction models..."):
                if not load_models(mr, fs):
                    st.stop()
    
    if (not st.session_state.last_model_check or 
        (datetime.now() - st.session_state.last_model_check) > timedelta(minutes=30)):
        with st.spinner("Checking for model updates..."):
            pm25_fresh = check_model_freshness(st.session_state.model_registry, "Karachi_pm25")
            pm10_fresh = check_model_freshness(st.session_state.model_registry, "Karachi_pm10")
            
            if pm25_fresh or pm10_fresh:
                load_models(st.session_state.model_registry, st.session_state.feature_store)
            
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
