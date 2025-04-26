import os
import yaml
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration handler for AQI Prediction Pipeline"""
    
    # Default cities
    default_cities = [
        {'name': 'Karachi', 'lat': 24.8607, 'lon': 67.0011},
        {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
        {'name': 'London', 'lat': 51.5074, 'lon': -0.1278}
    ]
    
    # API keys from environment variables
    AQICN_API_KEY = os.getenv('AQICN_API_KEY')
    OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
    HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
    
    # Hopsworks configuration
    HOPSWORKS_PROJECT_ID = "1219758"
    HOPSWORKS_PROJECT_NAME = "AQI_Pred_10Pearls"
    FEATURE_STORE_NAME = "air_quality_featurestore"
    
    # Data handling
    NULL_FILL_VALUE = -1  # For numeric nulls
    NULL_FILL_STRING = ""  # For string nulls

    @staticmethod
    def load_cities():
        """Load cities from YAML configuration file"""
        try:
            # Fixed path concatenation
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            yaml_path = os.path.join(base_dir, 'cities.yaml')
            
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if data and 'cities' in data:
                        return data['cities']
        except Exception as e:
            logger.error(f"Error loading cities: {str(e)}")
        return Config.default_cities

# Initialize cities
Config.CITIES = Config.load_cities()
