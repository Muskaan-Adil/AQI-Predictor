import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging

from utils.config import Config
from data_collector import DataCollector
from src.feature_engineering.feature_generator import FeatureGenerator
from src.feature_engineering.feature_store import FeatureStore

logger = logging.getLogger(__name__)

class HistoricalBackfill:
    """Class for backfilling historical data from OpenAQ."""

    def __init__(self):
        self.data_collector = DataCollector()
        self.feature_generator = FeatureGenerator()
        self.feature_store = FeatureStore()

    def run_backfill_for_city(self, city):
        """Backfill historical AQI data for a specific city using OpenAQ."""
        end_date = datetime.utcnow().date()
        start_date = end_date - relativedelta(months=6)
        city_name = city['name']

        logger.info(f"Backfilling AQI data for {city_name} from {start_date} to {end_date}...")

        aqi_df = self.data_collector.get_openaq_aqi_between_dates(
            city=city,
            parameter='pm25',
            start_date=start_date,
            end_date=end_date
        )

        if aqi_df.empty:
            logger.warning(f"No AQI data collected for {city_name}")
            return pd.DataFrame()

        logger.info(f"Generating features for {city_name}...")
        features_df = self.feature_generator.generate_all_features(aqi_df)

        logger.info(f"Storing features for {city_name}...")
        success = self.feature_store.store_features(features_df)

        if success:
            logger.info(f"Backfill for {city_name} completed successfully.")
        else:
            logger.warning(f"Backfill for {city_name} failed to store features.")

        return features_df

    def run_backfill_for_all_cities(self):
        """Run backfill for all configured cities."""
        all_features = []
        for city in Config.CITIES:
            city_features = self.run_backfill_for_city(city)
            if not city_features.empty:
                all_features.append(city_features)
        if all_features:
            return pd.concat(all_features, ignore_index=True)
        return pd.DataFrame()

def run_backfill_job():
    backfill = HistoricalBackfill()
    return backfill.run_backfill_for_all_cities()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    run_backfill_job()
