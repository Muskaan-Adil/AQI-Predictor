import requests
import logging
import pandas as pd
import time
from datetime import datetime, timedelta

from utils.config import Config

logger = logging.getLogger(__name__)

class DataCollector:
    """Class for collecting historical air quality data from OpenAQ."""

    def __init__(self):
        """Initialize the data collector."""
        self.default_parameter = "pm25"

    def get_openaq_aqi_between_dates(self, city, parameter, start_date, end_date, limit=1000):
        """Collect historical AQI data for a city from OpenAQ."""
        base_url = "https://api.openaq.org/v2/measurements"
        all_results = []
        page = 1

        while True:
            params = {
                "city": city['name'],
                "parameter": parameter,
                "limit": limit,
                "page": page,
                "date_from": start_date.isoformat(),
                "date_to": end_date.isoformat(),
                "sort": "asc"
            }

            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                logger.warning(f"OpenAQ request failed for {city['name']} on page {page} with status code {response.status_code}")
                break

            data = response.json()
            results = data.get("results", [])
            if not results:
                break

            all_results.extend(results)
            if len(results) < limit:
                break

            page += 1
            time.sleep(1)  # Be nice to the API

        if not all_results:
            logger.warning(f"No historical AQI data found for {city['name']}")
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        return df

    def collect_city_data(self, city):
        """Collect all relevant data for a city."""
        logger.info(f"Collecting data for {city['name']}...")

        # Example of collecting AQI and weather data (you may need additional methods here)
        # Assuming OpenAQ and OpenWeather APIs, you can combine them in a method

        # Collect AQI data
        start_date = datetime.now() - timedelta(days=30)  # last 30 days for example
        end_date = datetime.now()

        aqi_data = self.get_openaq_aqi_between_dates(city, self.default_parameter, start_date, end_date)

        if aqi_data.empty:
            logger.warning(f"No AQI data available for {city['name']}")
            return None

        # You can also collect weather data here if needed
        # weather_data = self.collect_weather_data(city)

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
            time.sleep(1)  # Sleep to avoid API rate limiting

        logger.info(f"Collected data for {len(all_data)} cities")
        return all_data
