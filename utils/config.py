import os
import yaml
import logging

logger = logging.getLogger(__name__)

class Config:
    """Optimized configuration for AQI Prediction Pipeline"""
    
    # Default cities
    default_cities = [
        {'name': 'Karachi', 'lat': 24.8607, 'lon': 67.0011},
        {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
        {'name': 'London', 'lat': 51.5074, 'lon': -0.1278},
    ]
    
    # API keys (from environment variables)
    AQICN_API_KEY = os.getenv('AQICN_API_KEY')
    OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
    HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
    
    # Verified Hopsworks Configuration
    HOPSWORKS_PROJECT_ID = "1219758"  # From your URL
    HOPSWORKS_PROJECT_NAME = "AQI_Pred_10Pearls"  # Case-sensitive
    FEATURE_STORE_NAME = "air_quality_featurestore"  # From your logs
    
    # Connection Settings
    HOPSWORKS_HOST = "c.app.hopsworks.ai"  # Free tier endpoint
    
    # Data Handling
    NULL_FILL_VALUE = -1  # For numeric nulls
    NULL_FILL_STRING = ""  # For string nulls

    @staticmethod
    def load_cities():
        """Robust city data loading"""
        try:
            yaml_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)),
                'cities.yaml'
            )
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    return data.get('cities', Config.default_cities)
        except Exception as e:
            logger.error(f"Config load error: {str(e)}")
        return Config.default_cities

# Initialize
Config.CITIES = Config.load_cities()
