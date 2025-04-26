import logging
import pandas as pd
from data_collector import DataCollector
from feature_store import FeatureStore  # Assuming you have a feature store to save processed data
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def backfill_data():
    """Backfill historical data for AQI from CSV file stored in a GitHub repository."""
    
    # Raw URL for your CSV file in the GitHub repository
    csv_url = "https://raw.githubusercontent.com/Muskaan-Adil/AQI-Predictor/main/aqi_data.csv"  # Raw GitHub URL

    # Load CSV directly from GitHub
    try:
        aqi_data = pd.read_csv(csv_url)
        logger.info(f"Successfully loaded CSV data from GitHub: {csv_url}")
    except Exception as e:
        logger.error(f"Failed to load CSV data from GitHub: {e}")
        return

    # Define the date range for backfilling (e.g., last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Example: last 30 days

    # Filter the data based on the date range
    aqi_data['date'] = pd.to_datetime(aqi_data['date'])
    filtered_data = aqi_data[(aqi_data['date'] >= start_date) & (aqi_data['date'] <= end_date)]

    if filtered_data.empty:
        logger.warning(f"No data found in the date range {start_date} to {end_date}")
        return

    logger.info(f"Backfilling data from {start_date} to {end_date}...")

    # Assuming the data has a 'city' column, and we want to process data for each city
    cities = filtered_data['city'].unique()

    # Loop through each city and process data
    for city in cities:
        city_data = filtered_data[filtered_data['city'] == city]
        
        # You would typically call your FeatureStore or other functions to process/store the data
        feature_store = FeatureStore()
        feature_store.save_city_data(city, city_data)

        logger.info(f"Data for {city} backfilled successfully.")

    logger.info("Backfilling completed.")

if __name__ == "__main__":
    backfill_data()
