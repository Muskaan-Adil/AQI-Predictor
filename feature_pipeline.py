import logging
import os
import yaml

from utils.config import Config  # ✅ Added to access API keys
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
    
    # ✅ Default to Karachi if YAML is missing
    default_cities = [
        {"name": "Karachi", "lat": 24.8607, "lon": 67.0011}
    ]
    
    try:
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)
                if data and 'cities' in data and isinstance(data['cities'], list):
                    logger.info(f"Loaded {len(data['cities'])} cities from YAML")
                    return data['cities']
        logger.warning("Cities YAML not found or invalid, using default cities")
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

        # ✅ Instantiate the data collector with API keys
        api_key_aqicn = Config.AQICN_API_KEY
        api_key_openweather = Config.OPENWEATHER_API_KEY
        if not api_key_aqicn or not api_key_openweather:
            logger.error("API keys for AQICN or OpenWeather are missing")
            return
        
        data_collector = DataCollector(
            api_key_aqicn=api_key_aqicn,
            api_key_openweather=api_key_openweather
        )

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

        # ✅ Improved logging
        logger.info(f"Generated {len(features)} feature records. Sample keys: {list(features[0].keys()) if features else []}")

        # Step 4: Store Features in Hopsworks
        logger.info("Storing features in Hopsworks...")
        feature_store = FeatureStore()
        feature_store.store_features(features)

        logger.info("✅ Feature pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Error in feature pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    run_feature_pipeline()
