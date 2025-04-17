import pandas as pd
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
from src.utils.config import Config
from src.data_collection.aqicn_api import AQICNDataCollector
from src.data_collection.openweather_api import OpenWeatherDataCollector

logger = logging.getLogger(__name__)

class DataCollector:
    """Main class for collecting and combining data from multiple sources."""
    
    def __init__(self):
        """Initialize the data collector with API clients."""
        self.aqicn_collector = AQICNDataCollector()
        self.weather_collector = OpenWeatherDataCollector()
    
    def collect_city_data(self, city):
        """Collect current data for a specific city.
        
        Args:
            city (dict): City dictionary with name, lat, lon.
            
        Returns:
            dict: Combined AQI and weather data.
        """
        city_name = city['name']
        lat = city['lat']
        lon = city['lon']
        
        logger.info(f"Collecting data for {city_name}...")
        
        aqi_data = self.aqicn_collector.get_current_data(lat=lat, lon=lon)
        processed_aqi = self.aqicn_collector.process_data(aqi_data) if aqi_data else {}
        
        weather_data = self.weather_collector.get_current_weather(lat=lat, lon=lon)
        processed_weather = self.weather_collector.process_weather_data(weather_data) if weather_data else {}
        
        combined_data = {
            'timestamp': datetime.now().isoformat(),
            'city': city_name,
            'coordinates': {
                'latitude': lat,
                'longitude': lon
            }
        }
        
        if processed_aqi:
            combined_data['aqi'] = processed_aqi.get('aqi')
            combined_data['pm25'] = processed_aqi.get('pm25')
            combined_data['pm10'] = processed_aqi.get('pm10')
            combined_data['pollutants'] = processed_aqi.get('pollutants', {})
        
        if processed_weather:
            combined_data['temperature'] = processed_weather.get('temperature')
            combined_data['humidity'] = processed_weather.get('humidity')
            combined_data['pressure'] = processed_weather.get('pressure')
            combined_data['wind_speed'] = processed_weather.get('wind', {}).get('speed')
            combined_data['wind_direction'] = processed_weather.get('wind', {}).get('direction')
            combined_data['clouds'] = processed_weather.get('clouds')
            combined_data['rain'] = processed_weather.get('rain')
            combined_data['weather_main'] = processed_weather.get('weather', {}).get('main')
            combined_data['weather_description'] = processed_weather.get('weather', {}).get('description')
        
        return combined_data
    
    def collect_all_cities_data(self):
        """Collect data for all cities defined in the configuration.
        
        Returns:
            list: List of combined data dictionaries for each city.
        """
        all_data = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(self.collect_city_data, Config.CITIES)
            all_data = list(results)
        
        return all_data
    
    def collect_historical_data(self, city, days_back=7):
        """Collect historical data for a specific city.
        
        Args:
            city (dict): City dictionary with name, lat, lon.
            days_back (int, optional): Days of historical data to collect. Defaults to 7.
            
        Returns:
            pd.DataFrame: DataFrame with historical data.
        """
        city_name = city['name']
        lat = city['lat']
        lon = city['lon']
        
        logger.info(f"Collecting {days_back} days of historical data for {city_name}...")
        
        aqi_df = self.aqicn_collector.get_historical_data(
            lat=lat, lon=lon, days_back=days_back
        )
        
        weather_df = self.weather_collector.get_multiple_historical_days(
            lat=lat, lon=lon, days_back=days_back
        )
        
        if aqi_df.empty and weather_df.empty:
            logger.warning(f"No historical data available for {city_name}")
            return pd.DataFrame()
        elif aqi_df.empty:
            logger.warning(f"No historical AQI data for {city_name}")
            return weather_df
        elif weather_df.empty:
            logger.warning(f"No historical weather data for {city_name}")
            return aqi_df
        
        merged_df = pd.merge(
            aqi_df, weather_df, 
            left_on='date', right_on='date',
            how='outer', suffixes=('_aqi', '_weather')
        )
        
        merged_df['city'] = city_name
        merged_df['latitude'] = lat
        merged_df['longitude'] = lon
        
        return merged_df
    
    def backfill_historical_data(self, days_back=365):
        """Collect historical data for all cities.
        
        Args:
            days_back (int, optional): Days of historical data to collect. Defaults to 365.
            
        Returns:
            pd.DataFrame: DataFrame with historical data for all cities.
        """
        all_historical_data = []
        
        for city in Config.CITIES:
            city_data = self.collect_historical_data(city, days_back)
            all_historical_data.append(city_data)
        
        if all_historical_data:
            return pd.concat(all_historical_data, ignore_index=True)
        else:
            return pd.DataFrame()
