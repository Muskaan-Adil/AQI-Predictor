import requests
import logging
import pandas as pd
import time
from datetime import datetime

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
