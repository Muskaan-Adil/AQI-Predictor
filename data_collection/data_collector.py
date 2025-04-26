import logging
import pandas as pd
from datetime import datetime
import requests
import time

logger = logging.getLogger(__name__)

class DataCollector:
    """Class for collecting real-time AQI data and backfilling from CSV."""
    
    def __init__(self, api_key, csv_file_path="aqi_data.csv"):
        """Initialize the data collector."""
        self.api_key = api_key  # API Key for real-time data
        self.csv_file_path = csv_file_path
        self.default_parameter = "pm25"
    
    def get_real_time_aqi(self, city):
        """Get real-time AQI data from API (e.g., OpenWeather, etc.)."""
        base_url = "https://api.openweathermap.org/data/2.5/air_pollution"
        params = {
            "lat": city["lat"],
            "lon": city["lon"],
            "appid": self.api_key
        }

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            logger.warning(f"Failed to fetch real-time AQI for {city['name']} with status code {response.status_code}")
            return None

    def backfill_with_csv(self):
        """Load the AQI data from the CSV file."""
        try:
            df = pd.read_csv(self.csv_file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return pd.DataFrame()

    def collect_data(self, city):
        """Collect data for a specific city, real-time or backfill from CSV."""
        logger.info(f"Collecting data for {city['name']}...")

        # Step 1: Try to get real-time data
        real_time_data = self.get_real_time_aqi(city)

        if real_time_data:
            logger.info(f"Real-time data collected for {city['name']}")
            return real_time_data

        # Step 2: If real-time data is not available, backfill from CSV
        logger.info(f"Backfilling data for {city['name']} from CSV...")
        df = self.backfill_with_csv()

        # Filter data for the city
        city_data = df[df['city'] == city['name']]

        if city_data.empty:
            logger.warning(f"No data found for {city['name']} in CSV.")
            return None

        logger.info(f"Data backfilled from CSV for {city['name']}")
        return city_data

    def collect_all_data(self, cities):
        """Collect data for all cities."""
        all_data = []
        for city in cities:
            city_data = self.collect_data(city)
            if city_data is not None:
                all_data.append(city_data)
            time.sleep(1)  # Avoid rate limiting issues

        return all_data
