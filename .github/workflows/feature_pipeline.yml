import os
import yaml
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration handler for the application."""
    
    # Define default cities as a class variable
    default_cities = [
        {'name': 'Karachi', 'lat': 24.8607, 'lon': 67.0011},
        {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
        {'name': 'London', 'lat': 51.5074, 'lon': -0.1278},
        # Add more default cities if needed
    ]
    
    AQICN_API_KEY = os.getenv('AQICN_API_KEY')
    OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
    HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
    
    HOPSWORKS_PROJECT_NAME = os.getenv('HOPSWORKS_PROJECT_NAME', 'AQI_PRED_10PEARLS')
    
    @staticmethod
    def load_cities():
        """Load cities from the YAML configuration file.
        
        Returns:
            list: List of city dictionaries with name, lat, lon.
        """
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cities.yaml')
        
        try:
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as file:
                    data = yaml.safe_load(file)
                    if data and 'cities' in data and isinstance(data['cities'], list):
                        return data['cities']
            return Config.default_cities  # Reference the default_cities as a class variable
        except Exception as e:
            print(f"Error loading cities from YAML: {e}")
            return Config.default_cities  # Reference the default_cities in case of error
    
    CITIES = load_cities.__func__()
    
    BACKFILL_DAYS = 365
    
    FEATURE_STORE_NAME = 'aqi_feature_store'
    
    MODEL_REGISTRY_NAME = 'aqi_model_registry'
    
    FORECAST_DAYS = 3
    
    DASHBOARD_TITLE = 'Pearls AQI Predictor'
