import logging
import os
import yaml

from data_collection.data_collector import DataCollector
from feature_engineering.feature_generator import FeatureGenerator
from feature_engineering.feature_store import FeatureStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_cities():
    """Load cities from YAML configuration file."""
    yaml_path = 'cities.yaml'
    
    default_cities = [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060},
        {"name": "London", "lat": 51.5074, "lon": -0.1278},
    ]
    
    try:
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)
                if data and 'cities' in data and isinstance(data['cities'], list):
                    logger.info(f"Loaded {len(data['cities'])} cities from YAML")
                    return data['cities']
        logger.warning("Cities YAML not found, using default cities")
        return default_cities
    except Exception as e:
        logger.error(f"Error loading cities from YAML: {e}")
        return default_cities


def run_feature_pipeline():
    """Run the full feature engineering pipeline."""
    logger.info("Starting feature pipeline...")

    try:
        # Step 1: Load Cities
        cities = load_cities()

        # Step 2: Collect Data
        logger.info("Collecting data from APIs...")
        
        # Instantiate the data collector with the necessary API keys
        api_key_aqicn = Config.AQICN_API_KEY  # Your AQICN API key
        api_key_openweather = Config.OPENWEATHER_API_KEY  # Your OpenWeather API key
        if not api_key_aqicn or not api_key_openweather:
            logger.error("API keys for AQICN or OpenWeather are missing")
            return
        
        data_collector = DataCollector(api_key_aqicn=api_key_aqicn, api_key_openweather=api_key_openweather)

        # Collect data for all cities
        all_data = []
        for city in cities:
            city_data = data_collector.collect_data(city)
            if city_data:
                all_data.append(city_data)
        
        if not all_data:
            logger.error("No data collected from APIs")
            return

        # Step 3: Feature Generation
        logger.info("Generating features...")
        feature_generator = FeatureGenerator()
        features = feature_generator.generate_all_features(all_data)

        if not features:
            logger.error("Feature generation failed")
            return

        logger.info(f"Generated features: {features[:5]}")  # Log only a few features for inspection

        # Step 4: Store Features in Hopsworks
        logger.info("Storing features in Hopsworks...")
        feature_store = FeatureStore()
        feature_store.store_features(features)

        logger.info("Feature pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Error in feature pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_feature_pipeline()
