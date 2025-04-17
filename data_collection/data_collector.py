import requests
import logging
import json
import time
from datetime import datetime, timedelta

from utils.config import Config

logger = logging.getLogger(__name__)

class DataCollector:
    """Class for collecting air quality and weather data from APIs."""
    
    def __init__(self):
        """Initialize the data collector."""
        self.aqicn_api_key = Config.AQICN_API_KEY
        self.openweather_api_key = Config.OPENWEATHER_API_KEY
        
        if not self.aqicn_api_key or not self.openweather_api_key:
            logger.warning("API keys are not set. Set AQICN_API_KEY and OPENWEATHER_API_KEY environment variables.")
    
    def collect_aqi_data(self, city):
        """Collect air quality data for a specific city."""
        logger.info(f"Collecting AQI data for {city['name']}...")
        
        try:
            url = f"https://api.waqi.info/feed/geo:{city['lat']};{city['lon']}/?token={self.aqicn_api_key}"
            response = requests.get(url)
            data = response.json()
            
            if data['status'] == 'ok':
                return data['data']
            else:
                logger.error(f"Error fetching AQI data for {city['name']}: {data}")
                return None
        except Exception as e:
            logger.error(f"Exception while fetching AQI data for {city['name']}: {e}")
            return None
    
    def collect_weather_data(self, city):
        """Collect weather data for a specific city."""
        logger.info(f"Collecting weather data for {city['name']}...")
        
        try:
            url = f"https://api.openweathermap.org/data/2.5/onecall?lat={city['lat']}&lon={city['lon']}&appid={self.openweather_api_key}&units=metric"
            response = requests.get(url)
            data = response.json()
            
            if 'current' in data:
                return data
            else:
                logger.error(f"Error fetching weather data for {city['name']}: {data}")
                return None
        except Exception as e:
            logger.error(f"Exception while fetching weather data for {city['name']}: {e}")
            return None
    
    def collect_city_data(self, city):
        """Collect all data for a specific city."""
        logger.info(f"Collecting data for {city['name']}...")
        
        aqi_data = self.collect_aqi_data(city)
        weather_data = self.collect_weather_data(city)
        
        if aqi_data and weather_data:
            return {
                'city': city,
                'aqi': aqi_data,
                'weather': weather_data,
                'timestamp': datetime.now().isoformat()
            }
        else:
            logger.warning(f"Could not collect complete data for {city['name']}")
            return None
    
    def collect_all_cities_data(self, cities=None):
        """Collect data for all cities."""
        if cities is None:
            cities = Config.CITIES
        
        logger.info(f"Collecting data for {len(cities)} cities...")
        
        all_data = []
        for city in cities:
            city_data = self.collect_city_data(city)
            if city_data:
                all_data.append(city_data)
            time.sleep(1)
        
        logger.info(f"Collected data for {len(all_data)} cities")
        return all_data
