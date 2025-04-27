import os
import yaml
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration handler for the application."""
    DASHBOARD_TITLE = "AQI Predictor Dashboard"

    # Define default cities first
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
    
    # Model registry name
    MODEL_REGISTRY_NAME = "air_quality_models"

    @staticmethod
    def load_cities():
        """Load cities from YAML configuration file"""
        try:
            # Get the base directory (3 levels up from config.py)
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            yaml_path = os.path.join(base_dir, 'cities.yaml')
            
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if data and 'cities' in data:
                        return data['cities']
        except Exception as e:
            logger.error(f"Error loading cities from YAML: {e}")
        
        # Return default cities if YAML loading fails
        return Config.default_cities

    @staticmethod
    def ensure_model_registry_exists(feature_store):
        """
        Ensure the model registry exists.
        If not, create it.
        """
        try:
            registry_name = Config.MODEL_REGISTRY_NAME
            model_registry = feature_store.get_model_registry()
            registries = model_registry.list_model_registries()

            existing_names = [r.name for r in registries]
            if registry_name not in existing_names:
                logger.info(f"Model Registry '{registry_name}' not found. Creating it...")
                model_registry.create_model_registry(name=registry_name)
            else:
                logger.info(f"Model Registry '{registry_name}' already exists.")

        except Exception as e:
            logger.error(f"Error ensuring model registry exists: {e}")
            raise e

# Initialize cities after class definition
Config.CITIES = Config.load_cities()
