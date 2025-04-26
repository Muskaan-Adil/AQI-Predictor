import logging
import pandas as pd
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)

class DataCollector:
    """Class for collecting historical air quality data from a CSV file on GitHub."""

    def __init__(self):
        """Initialize the data collector."""
        self.csv_url = "https://raw.githubusercontent.com/Muskaan-Adil/AQI-Predictor/main/aqi_data.csv"  # Raw GitHub URL

    def get_data_from_csv(self, start_date, end_date):
        """Collect AQI data for a city from the CSV file."""
        try:
            # Load the CSV directly from GitHub
            aqi_data = pd.read_csv(self.csv_url)

            # Ensure the 'date' column is in datetime format
            aqi_data['date'] = pd.to_datetime(aqi_data['date'])

            # Filter the data based on the date range
            filtered_data = aqi_data[(aqi_data['date'] >= start_date) & (aqi_data['date'] <= end_date)]

            if filtered_data.empty:
                logger.warning(f"No AQI data found for the given date range: {start_date} to {end_date}")
                return pd.DataFrame()  # Return an empty dataframe

            return filtered_data
        except Exception as e:
            logger.error(f"Failed to load AQI data from CSV: {e}")
            return pd.DataFrame()  # Return an empty dataframe in case of error

    def collect_city_data(self, city):
        """Collect all relevant data for a city."""
        logger.info(f"Collecting data for {city['name']}...")

        # Define the date range for data collection (e.g., last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Example: last 30 days

        # Get the AQI data from CSV
        aqi_data = self.get_data_from_csv(start_date, end_date)

        if aqi_data.empty:
            logger.warning(f"No AQI data available for {city['name']}")
            return None

        # Return the collected data
        return {
            'city': city,
            'aqi': aqi_data,
            'timestamp': datetime.now().isoformat()
        }

    def collect_all_cities_data(self, cities=None):
        """Collect data for all cities."""
        if cities is None:
            # Default cities, you can replace this with your own config logic if necessary
            cities = [{"name": "New York", "lat": 40.7128, "lon": -74.0060}, {"name": "London", "lat": 51.5074, "lon": -0.1278}]

        logger.info(f"Collecting data for {len(cities)} cities...")

        all_data = []
        for city in cities:
            city_data = self.collect_city_data(city)
            if city_data:
                all_data.append(city_data)

        logger.info(f"Collected data for {len(all_data)} cities")
        return all_data
