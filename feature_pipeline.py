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

        # Instantiate the data collector
        data_collector = DataCollector()

        # Load real-time data from APIs
        logger.info("Collecting real-time data from OpenAQ API...")
        real_time_data = data_collector.collect_all_cities_data(cities=cities)
        
        # Backfill data from CSV (using the backfill function in DataCollector)
        logger.info("Backfilling data from CSV file...")
        backfilled_data = data_collector.backfill_with_csv()

        # Combine real-time data and backfilled data
        logger.info("Merging real-time data with backfilled data...")
        combined_data = data_collector.merge_data(real_time_data, backfilled_data)

        if not combined_data:
            logger.error("No valid data available after merging real-time and backfilled data")
            return

        logger.info(f"Merged data contains {len(combined_data)} records.")

        # Step 3: Feature Generation
        logger.info("Generating features...")
        feature_generator = FeatureGenerator()
        features = feature_generator.generate_all_features(combined_data)

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
