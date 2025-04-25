# config.py
import os
import yaml

class Config:
    """Configuration handler for the application."""
    
    # Default cities
    default_cities = [
        {'name': 'Karachi', 'lat': 24.8607, 'lon': 67.0011},
        {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
        {'name': 'London', 'lat': 51.5074, 'lon': -0.1278},
    ]
    
    # API keys pulled directly from environment variables (GitHub Secrets)
    AQICN_API_KEY = os.getenv('AQICN_API_KEY')
    OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
    HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
    
    HOPSWORKS_PROJECT_NAME = os.getenv('HOPSWORKS_PROJECT_NAME', 'aqi_pred_10pearls_featurestore')
    FEATURE_STORE_NAME = os.getenv('FEATURE_STORE_NAME', 'features')  # <-- add this

    @staticmethod
    def load_cities():
        """Load cities from YAML configuration file."""
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cities.yaml')
        
        try:
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as file:
                    data = yaml.safe_load(file)
                    if data and 'cities' in data and isinstance(data['cities'], list):
                        return data['cities']
            return Config.default_cities
        except Exception as e:
            print(f"Error loading cities from YAML: {e}")
            return Config.default_cities

Config.CITIES = Config.load_cities()
