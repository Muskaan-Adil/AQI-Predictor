import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class FeatureGenerator:
    """Class for generating features from raw data."""

    def __init__(self):
        """Initialize the feature generator."""
        pass

    def extract_features_from_aqi(self, aqi_data):
        """Extract features from AQI data."""
        features = {}

        try:
            features['aqi'] = aqi_data.get('aqi')

            iaqi = aqi_data.get('iaqi', {})
            features['pm25'] = iaqi.get('pm25', {}).get('v')
            features['pm10'] = iaqi.get('pm10', {}).get('v')

            features['o3'] = iaqi.get('o3', {}).get('v')
            features['no2'] = iaqi.get('no2', {}).get('v')
            features['so2'] = iaqi.get('so2', {}).get('v')
            features['co'] = iaqi.get('co', {}).get('v')

            return features
        except Exception as e:
            logger.error(f"Error extracting AQI features: {e}")
            return {}

    def extract_features_from_weather(self, weather_data):
        """Extract features from weather data."""
        features = {}

        try:
            current = weather_data.get('current', {})
            features['temp'] = current.get('temp')
            features['feels_like'] = current.get('feels_like')
            features['pressure'] = current.get('pressure')
            features['humidity'] = current.get('humidity')
            features['wind_speed'] = current.get('wind_speed')
            features['wind_deg'] = current.get('wind_deg')
            features['clouds'] = current.get('clouds')

            weather = current.get('weather', [{}])[0]
            features['weather_id'] = weather.get('id')
            features['weather_main'] = weather.get('main')

            return features
        except Exception as e:
            logger.error(f"Error extracting weather features: {e}")
            return {}

    def generate_features(self, data):
        """Generate features from raw data for a single city."""
        try:
            city = data['city']
            city_name = city['name']

            timestamp = datetime.fromisoformat(data['timestamp'])

            aqi_features = self.extract_features_from_aqi(data['aqi'])
            weather_features = self.extract_features_from_weather(data['weather'])

            features = {
                'city': city_name,
                'lat': city['lat'],
                'lon': city['lon'],
                'timestamp': timestamp,
                **aqi_features,
                **weather_features
            }

            return features
        except Exception as e:
            logger.error(f"Error generating features: {e}")
            return None

    def generate_all_features(self, all_data):
        """Generate features for all cities data."""
        logger.info("Generating features for all cities...")

        all_features = []
        for data in all_data:
            features = self.generate_features(data)
            if features:
                all_features.append(features)

        logger.info(f"Generated features for {len(all_features)} cities")
        return all_features
