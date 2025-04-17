import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from src.utils.config import Config

logger = logging.getLogger(__name__)

class AQICNDataCollector:
    """Class to collect air quality data from AQICN API."""
    
    def __init__(self, api_key=None):
        """Initialize the AQICN data collector.
        
        Args:
            api_key (str, optional): API key for AQICN. Defaults to None, which will use the key from Config.
        """
        self.api_key = api_key or Config.AQICN_API_KEY
        if not self.api_key:
            logger.warning("AQICN API key not provided. API calls will fail.")
        
        self.base_url = "https://api.waqi.info/feed/"
    
    def get_current_data(self, city_name=None, lat=None, lon=None):
        """Get current air quality data for a location.
        
        Args:
            city_name (str, optional): Name of the city. Defaults to None.
            lat (float, optional): Latitude. Defaults to None.
            lon (float, optional): Longitude. Defaults to None.
            
        Returns:
            dict: Air quality data or None if request fails.
        """
        if city_name:
            endpoint = f"{self.base_url}{city_name}/"
        elif lat is not None and lon is not None:
            endpoint = f"{self.base_url}geo:{lat};{lon}/"
        else:
            logger.error("Either city_name or lat/lon must be provided.")
            return None
            
        params = {'token': self.api_key}
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'ok':
                return data['data']
            else:
                logger.error(f"API error: {data.get('data')}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
    
    def get_historical_data(self, city_name=None, lat=None, lon=None, days_back=7):
        """Get historical air quality data.
        
        Note: The free tier of AQICN API might have limitations on historical data.
        This method will collect what's available.
        
        Args:
            city_name (str, optional): Name of the city. Defaults to None.
            lat (float, optional): Latitude. Defaults to None.
            lon (float, optional): Longitude. Defaults to None.
            days_back (int, optional): Days of historical data to fetch. Defaults to 7.
            
        Returns:
            pd.DataFrame: DataFrame with historical data or empty DataFrame if request fails.
        """
        station_data = self.get_current_data(city_name, lat, lon)
        if not station_data:
            return pd.DataFrame()
            
        station_id = station_data.get('idx')
        if not station_id:
            logger.error("Could not find station ID in response.")
            return pd.DataFrame()
        
        historical_url = f"https://api.waqi.info/feed/@{station_id}/period=daily/history/"
        params = {'token': self.api_key}
        
        try:
            response = requests.get(historical_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'ok':
                historical_data = data['data']
                df_rows = []
                
                for entry in historical_data:
                    date = entry.get('date')
                    pm25 = entry.get('pm25', {}).get('value')
                    pm10 = entry.get('pm10', {}).get('value')
                    
                    if date and (pm25 is not None or pm10 is not None):
                        df_rows.append({
                            'date': date,
                            'pm25': pm25,
                            'pm10': pm10,
                            'city': city_name or f"{lat},{lon}"
                        })
                
                return pd.DataFrame(df_rows)
            else:
                logger.error(f"API error: {data.get('data')}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return pd.DataFrame()
    
    def process_data(self, data):
        """Process the raw API data into a structured format.
        
        Args:
            data (dict): Raw data from AQICN API.
            
        Returns:
            dict: Processed data with structured fields.
        """
        if not data:
            return {}
            
        result = {
            'timestamp': datetime.now().isoformat(),
            'aqi': data.get('aqi'),
            'city': data.get('city', {}).get('name'),
            'coordinates': {
                'latitude': data.get('city', {}).get('geo', [0, 0])[0],
                'longitude': data.get('city', {}).get('geo', [0, 0])[1]
            },
            'pollutants': {}
        }
        
        iaqi = data.get('iaqi', {})
        for pollutant, value in iaqi.items():
            result['pollutants'][pollutant] = value.get('v')
            
        if 'pm25' in iaqi:
            result['pm25'] = iaqi['pm25'].get('v')
        if 'pm10' in iaqi:
            result['pm10'] = iaqi['pm10'].get('v')
            
        return result
