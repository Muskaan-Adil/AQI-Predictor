import logging
import os
import yaml
from data_collection.data_collector import DataCollector
from feature_engineering.feature_generator import FeatureGenerator
from feature_engineering.feature_store import FeatureStore
from utils.config import Config

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

        # Step 2: Set up API keys
        api_key_aqicn = Config.AQICN_API_KEY
        api_key_openweather = Config.OPENWEATHER_API_KEY
        if not api_key_aqicn or not api_key_openweather:
            logger.error("API keys for AQICN or OpenWeather are missing")
            return

        data_collector = DataCollector(api_key_aqicn=api_key_aqicn, api_key_openweather=api_key_openweather)
        feature_generator = FeatureGenerator()

        all_features = []

        # Step 3: Backfill CSV Data
        logger.info("Backfilling from CSV...")
        backfilled_df = data_collector.backfill_with_csv()
        if not backfilled_df.empty:
            backfill_features = feature_generator.generate_from_backfill(backfilled_df)
            all_features.extend(backfill_features)

        # Step 4: Real-time data collection
        logger.info("Collecting real-time data...")
        for city in cities:
            city_data = data_collector.collect_data(city)
            if city_data:
                features = feature_generator.generate_features(city_data)
                if features:
                    all_features.append(features)

        if not all_features:
            logger.error("No features generated")
            return

        logger.info(f"Generated {len(all_features)} feature records. Sample keys: {list(all_features[0].keys())}")

        # Step 5: Store Features
        feature_store = FeatureStore()
        feature_store.store_features(all_features)

        logger.info("✅ Feature pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Error in feature pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_feature_pipeline()
