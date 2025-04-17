import os
import yaml
from dotenv import load_dotenv

load_dotenv()

def load_cities():
    """Load cities from YAML configuration file."""
    yaml_path = 'cities.yaml'
    
    try:
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)
                if data and 'cities' in data and isinstance(data['cities'], list):
                    print(f"Loaded {len(data['cities'])} cities from YAML")
                    return data['cities']
        print("Cities YAML not found, using default cities")
        return DEFAULT_CITIES
    except Exception as e:
        print(f"Error loading cities from YAML: {e}")
        return DEFAULT_CITIES
class Config:
    """Configuration handler for the application."""
    
    AQICN_API_KEY = os.getenv('AQICN_API_KEY')
    OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
    HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
    
    HOPSWORKS_PROJECT_NAME = os.getenv('HOPSWORKS_PROJECT_NAME', 'AQI_PRED_10PEARLS')
    
    CITIES = load_cities()
    
    BACKFILL_DAYS = 365
    
    FEATURE_STORE_NAME = 'aqi_feature_store'
    
    MODEL_REGISTRY_NAME = 'aqi_model_registry'
    
    FORECAST_DAYS = 3
    
    DASHBOARD_TITLE = 'Pearls AQI Predictor'
