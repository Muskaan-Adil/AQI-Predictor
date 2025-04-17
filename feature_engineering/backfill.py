import pandas as pd
from datetime import datetime, timedelta
import logging
from src.utils.config import Config
from src.data_collection.data_collector import DataCollector
from src.feature_engineering.feature_generator import FeatureGenerator
from src.feature_engineering.feature_store import FeatureStore

logger = logging.getLogger(__name__)

class HistoricalBackfill:
    """Class for backfilling historical data."""
    
    def __init__(self):
        """Initialize the historical backfill processor."""
        self.data_collector = DataCollector()
        self.feature_generator = FeatureGenerator()
        self.feature_store = FeatureStore()
    
    def run_backfill(self, days_back=None):
        """Run the backfill process.
        
        Args:
            days_back (int, optional): Days of historical data to backfill. Defaults to Config value.
            
        Returns:
            pd.DataFrame: DataFrame with processed historical data.
        """
        days_back = days_back or Config.BACKFILL_DAYS
        logger.info(f"Starting historical backfill for {days_back} days...")
        
        historical_df = self.data_collector.backfill_historical_data(days_back)
        
        if historical_df.empty:
            logger.warning("No historical data collected")
            return pd.DataFrame()
        
        logger.info("Generating features from historical data...")
        features_df = self.feature_generator.generate_all_features(historical_df)
        
        logger.info("Storing features in feature store...")
        success = self.feature_store.store_features(features_df)
        
        if success:
            logger.info("Historical backfill completed successfully")
        else:
            logger.warning("There were issues storing features in the feature store")
        
        return features_df
    
    def run_backfill_for_city(self, city, days_back=None):
        """Run the backfill process for a specific city.
        
        Args:
            city (dict): City dictionary with name, lat, lon.
            days_back (int, optional): Days of historical data to backfill. Defaults to Config value.
            
        Returns:
            pd.DataFrame: DataFrame with processed historical data for the city.
        """
        days_back = days_back or Config.BACKFILL_DAYS
        city_name = city['name']
        logger.info(f"Starting historical backfill for {city_name} for {days_back} days...")
        
        historical_df = self.data_collector.collect_historical_data(city, days_back)
        
        if historical_df.empty:
            logger.warning(f"No historical data collected for {city_name}")
            return pd.DataFrame()
        
        logger.info(f"Generating features from historical data for {city_name}...")
        features_df = self.feature_generator.generate_all_features(historical_df)
        
        logger.info(f"Storing features for {city_name} in feature store...")
        success = self.feature_store.store_features(features_df)
        
        if success:
            logger.info(f"Historical backfill for {city_name} completed successfully")
        else:
            logger.warning(f"There were issues storing features for {city_name} in the feature store")
        
        return features_df

def run_backfill_job():
    """Run the historical backfill job for all cities."""
    backfill = HistoricalBackfill()
    return backfill.run_backfill()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    run_backfill_job()
