import os
import yaml
import logging
from pathlib import Path  # Better path handling

logger = logging.getLogger(__name__)

class Config:
    """Configuration handler for the application."""
    DASHBOARD_TITLE = "AQI Predictor Dashboard"

    # Default cities (now a constant)
    _DEFAULT_CITIES = [{'name': 'Karachi', 'lat': 24.8607, 'lon': 67.0011}]
    
    # API keys (type hinted via comment)
    AQICN_API_KEY: str = os.getenv('AQICN_API_KEY')  # type: ignore
    OPENWEATHER_API_KEY: str = os.getenv('OPENWEATHER_API_KEY')  # type: ignore
    HOPSWORKS_API_KEY: str = os.getenv('HOPSWORKS_API_KEY')  # type: ignore
    
    # Hopsworks configuration (constants)
    HOPSWORKS_HOST = "c.app.hopsworks.ai"
    HOPSWORKS_PROJECT_ID = "1219758"
    HOPSWORKS_PROJECT_NAME = "AQI_Pred_10Pearls"
    FEATURE_STORE_NAME = "air_quality_featurestore"
    MODEL_REGISTRY_NAME = "air_quality_models"

    @classmethod
    def get_project_root(cls) -> Path:
        """Get absolute path to project root (more reliable)"""
        return Path(__file__).parent.parent.parent

    @classmethod
    def load_cities(cls) -> list[dict]:
        """Load cities from YAML with enhanced error handling"""
        try:
            yaml_path = cls.get_project_root() / 'cities.yaml'
            logger.info(f"Loading cities from: {yaml_path}")
            
            if not yaml_path.exists():
                logger.warning("cities.yaml not found, using defaults")
                return cls._DEFAULT_CITIES
                
            with open(yaml_path, 'r') as f:
                if (data := yaml.safe_load(f)) and 'cities' in data:
                    logger.info(f"Loaded {len(data['cities'])} cities")
                    return data['cities']
                
            logger.error("YAML missing 'cities' key")
            return cls._DEFAULT_CITIES
            
        except Exception as e:
            logger.error(f"YAML loading failed: {str(e)}")
            return cls._DEFAULT_CITIES

    @staticmethod
    def ensure_model_registry_exists(feature_store) -> bool:
        """Ensure model registry exists (now returns success status)"""
        try:
            registry_name = Config.MODEL_REGISTRY_NAME
            model_registry = feature_store.get_model_registry()
            
            if registry_name not in [r.name for r in model_registry.list_model_registries()]:
                logger.info(f"Creating registry: {registry_name}")
                model_registry.create_model_registry(name=registry_name)
                
            return True
            
        except Exception as e:
            logger.error(f"Registry setup failed: {str(e)}")
            raise RuntimeError(f"Could not ensure registry exists: {str(e)}")

# Initialize (now using classmethod)
CITIES = Config.load_cities()
