import logging
import os
from datetime import datetime
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from data_collection.data_collector import DataCollector
from feature_engineering.feature_generator import FeatureGenerator
from feature_engineering.feature_store import FeatureStore

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
    """Run the feature engineering pipeline."""
    logger.info("Starting feature pipeline...")
    
    try:
        cities = load_cities()
        
        logger.info("Collecting data from APIs...")
        data_collector = DataCollector()
        data = data_collector.collect_all_cities_data(cities=cities)
        
        if not data:
            logger.error("No data collected from APIs")
            return
            
        logger.info("Generating features...")
        feature_generator = FeatureGenerator()
        features = feature_generator.generate_all_features(data)
        
        if not features:
            logger.error("Feature generation failed")
            return
            
        logger.info("Storing features...")
        feature_store = FeatureStore()
        feature_store.store_features(features)
        
        logger.info("Feature pipeline completed successfully!")
    
    except Exception as e:
        logger.error(f"Error in feature pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_feature_pipeline()
