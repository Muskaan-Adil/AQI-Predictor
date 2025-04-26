import os
import yaml

class Config:
    """Optimized configuration for Hopsworks integration"""
    
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
    
    # Hopsworks Configuration (Verified from your project)
    HOPSWORKS_PROJECT_ID = "1219758"  # From URL: https://c.app.hopsworks.ai/p/1219758/view
    HOPSWORKS_PROJECT_NAME = "AQI_Pred_10Pearls"  # Must match EXACTLY (case-sensitive)
    FEATURE_STORE_NAME = "air_quality_featurestore"  # Confirmed from your logs
    
    # Connection Settings (Free Tier)
    HOPSWORKS_HOST = "c.app.hopsworks.ai"  # Free tier endpoint
    HOPSWORKS_DISABLE_KAFKA = True  # Critical for avoiding Kafka errors
    
    # Data Settings
    NULL_FILL_VALUE = -1  # Default for numeric nulls
    NULL_FILL_STRING = ""  # Default for string nulls

    @staticmethod
    def load_cities():
        """Load cities with improved error handling"""
        yaml_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'cities.yaml'
        )
        
        try:
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    return data.get('cities', Config.default_cities)
        except Exception as e:
            logger.error(f"YAML load error: {str(e)}")
        
        return Config.default_cities

# Initialize
Config.CITIES = Config.load_cities()
