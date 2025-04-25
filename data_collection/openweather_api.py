import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from src.utils.config import Config

logger = logging.getLogger(__name__)

class OpenWeatherDataCollector:
    """Class to collect weather data from OpenWeather API."""
    
    def __init__(self, api_key=None):
        """Initialize the OpenWeather data collector.
        
        Args:
            api_key (str, optional): API key for OpenWeather. Defaults to None, which will use the key from Config.
        """
        self.api_key = api_key or Config.OPENWEATHER_API_KEY
        if not self.api_key:
            logger.warning("OpenWeather API key not provided. API calls will fail.")
        
        self.current_weather_url = "https://api.openweathermap.org/data/2.5/weather"
        self.forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
        self.historical_url = "https://api.openweathermap.org/data/2.5/onecall/timemachine"
    
    def get_current_weather(self, city_name=None, lat=None, lon=None):
        """Get current weather data for a location.
        
        Args:
            city_name (str, optional): Name of the city. Defaults to None.
            lat (float, optional): Latitude. Defaults to None.
            lon (float, optional): Longitude. Defaults to None.
            
        Returns:
            dict: Weather data or None if request fails.
        """
        params = {'appid': self.api_key, 'units': 'metric'}
        
        if city_name:
            params['q'] = city_name
        elif lat is not None and lon is not None:
            params['lat'] = lat
            params['lon'] = lon
        else:
            logger.error("Either city_name or lat/lon must be provided.")
            return None
            
        try:
            response = requests.get(self.current_weather_url, params=params)
            response.raise_for_status()
            return response.json()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
    
    def get_forecast(self, city_name=None, lat=None, lon=None):
        """Get 5-day weather forecast for a location (3-hour intervals).
        
        Args:
            city_name (str, optional): Name of the city. Defaults to None.
            lat (float, optional): Latitude. Defaults to None.
            lon (float, optional): Longitude. Defaults to None.
            
        Returns:
            dict: Forecast data or None if request fails.
        """
        params = {'appid': self.api_key, 'units': 'metric'}
        
        if city_name:
            params['q'] = city_name
        elif lat is not None and lon is not None:
            params['lat'] = lat
            params['lon'] = lon
        else:
            logger.error("Either city_name or lat/lon must be provided.")
            return None
            
        try:
            response = requests.get(self.forecast_url, params=params)
            response.raise_for_status()
            return response.json()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
    
    def get_historical_weather(self, lat, lon, dt=None, days_back=1):
    """Get historical weather data using Open-Meteo (no API key required)."""
    date = datetime.now() - timedelta(days=days_back)
    date_str = date.strftime('%Y-%m-%d')
    
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&hourly=temperature_2m,relative_humidity_2m,pressure_msl,windspeed_10m"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Open-Meteo request error: {e}")
        return None

    
    def get_multiple_historical_days(self, lat, lon, days_back=7):
        """Get weather data for multiple past days.
        
        Args:
            lat (float): Latitude.
            lon (float): Longitude.
            days_back (int, optional): Number of days to go back. Defaults to 7.
            
        Returns:
            pd.DataFrame: DataFrame with historical weather data.
        """
        historical_data = []
        
        for i in range(days_back):
            dt = int((datetime.now() - timedelta(days=i+1)).timestamp())
            data = self.get_historical_weather(lat, lon, dt)
            
            if data:
                hourly_data = data.get('hourly', [])
                
                for hour in hourly_data:
                    hour_data = {
                        'date': datetime.fromtimestamp(hour.get('dt')).isoformat(),
                        'temp': hour.get('temp'),
                        'humidity': hour.get('humidity'),
                        'pressure': hour.get('pressure'),
                        'wind_speed': hour.get('wind_speed'),
                        'wind_direction': hour.get('wind_deg'),
                        'rain_1h': hour.get('rain', {}).get('1h', 0) if 'rain' in hour else 0,
                        'lat': lat,
                        'lon': lon
                    }
                    historical_data.append(hour_data)
        
        return pd.DataFrame(historical_data) if historical_data else pd.DataFrame()
    
    def process_weather_data(self, data):
        """Process raw weather data into structured format.
        
        Args:
            data (dict): Raw data from OpenWeather API.
            
        Returns:
            dict: Processed weather data.
        """
        if not data:
            return {}
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'city': data.get('name'),
            'coordinates': {
                'latitude': data.get('coord', {}).get('lat'),
                'longitude': data.get('coord', {}).get('lon')
            },
            'temperature': data.get('main', {}).get('temp'),
            'feels_like': data.get('main', {}).get('feels_like'),
            'humidity': data.get('main', {}).get('humidity'),
            'pressure': data.get('main', {}).get('pressure'),
            'wind': {
                'speed': data.get('wind', {}).get('speed'),
                'direction': data.get('wind', {}).get('deg')
            },
            'clouds': data.get('clouds', {}).get('all'),
            'rain': data.get('rain', {}).get('1h', 0) if 'rain' in data else 0,
            'snow': data.get('snow', {}).get('1h', 0) if 'snow' in data else 0,
            'weather': {
                'main': data.get('weather', [{}])[0].get('main'),
                'description': data.get('weather', [{}])[0].get('description')
            }
        }
        
        return result
