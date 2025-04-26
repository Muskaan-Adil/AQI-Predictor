import requests
import pandas as pd
import logging
import time
from datetime import datetime
from utils.config import Config

logger = logging.getLogger(__name__)

class DataCollector:
    """Class for collecting historical and real-time air quality data from AQICN and OpenWeather."""

    def __init__(self, api_key_aqicn, api_key_openweather):
        """Initialize the data collector."""
        self.api_key_aqicn = api_key_aqicn
        self.api_key_openweather = api_key_openweather
        if not self.api_key_aqicn or not self.api_key_openweather:
            raise ValueError("Both AQICN and OpenWeather API keys are required.")
        self.default_parameter = "pm25"

    def get_aqicn_data(self, city):
        """Collect real-time AQI data for a city from AQICN API."""
        base_url = f"https://api.waqi.info/feed/{city['name']}/?token={self.api_key_aqicn}"
        try:
            response = requests.get(base_url)
            response.raise_for_status()
            data = response.json()
            if data["status"] == "ok":
                return data["data"]
            else:
                logger.warning(f"AQICN request failed for {city['name']}: {data.get('data', {}).get('error', {}).get('message', '')}")
        except requests.RequestException as e:
            logger.warning(f"AQICN request failed for {city['name']} with error: {e}")
        return None

    def get_openweather_data(self, city):
        """Collect real-time weather data for a city from OpenWeather API."""
        base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city['name']}&appid={self.api_key_openweather}&units=metric"
        try:
            response = requests.get(base_url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"OpenWeather request failed for {city['name']} with error: {e}")
        return None

    def backfill_with_csv(self, csv_path="aqi_data.csv"):
        """Backfill missing data from CSV file."""
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Backfilled {len(df)} rows from CSV")
            return df
        except Exception as e:
            logger.error(f"Error backfilling data from CSV: {e}")
            return pd.DataFrame()

    def collect_data(self, city):
        """Collect both real-time and backfilled data for a city."""
        logger.info(f"Collecting data for {city['name']}...")

        # Collect real-time data
        aqi_data = self.get_aqicn_data(city)
        weather_data = self.get_openweather_data(city)

        if not aqi_data or not weather_data:
            logger.warning(f"Real-time data collection failed for {city['name']}")
            return None

        # Backfill data from CSV
        backfilled_data = self.backfill_with_csv()

        # Standardize and combine the collected data
        combined_data = {
            'city': city,
            'aqi': aqi_data,
            'weather': weather_data,
            'backfilled': backfilled_data.to_dict(orient='records') if not backfilled_data.empty else [],
            'timestamp': datetime.now().isoformat()
        }

        return combined_data
