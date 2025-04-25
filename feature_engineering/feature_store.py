import hopsworks  # Use the new SDK instead of hsfs
import pandas as pd
from datetime import datetime
import os

class FeatureStore:
    def __init__(self):
        self.api_key = os.getenv('HOPSWORKS_API_KEY')
        if not self.api_key:
            raise ValueError("HOPSWORKS_API_KEY not set in environment variables")
        
        # Connect using the new SDK
        try:
            self.project = hopsworks.login(api_key_value=self.api_key)
            self.fs = self.project.get_feature_store()
            self.can_connect = True
        except Exception as e:
            print(f"Login failed: {str(e)}")
            self.can_connect = False

    def store_features(self, features):
        """Store using the new SDK"""
        try:
            df = pd.DataFrame(features)
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6
            
            # Get or create feature group
            fg = self.fs.get_or_create_feature_group(
                name="karachi_aqi_realtime",
                version=1,
                primary_key=['timestamp'],
                description="Karachi AQI Data",
                online_enabled=False
            )
            
            # Insert data
            fg.insert(df)
            print("✅ Successfully stored in Hopsworks!")
            
        except Exception as e:
            print(f"Storage failed: {str(e)}")
            self._store_locally(df)

    def _store_locally(self, df):
        """Fallback local storage"""
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/karachi_aqi.csv', mode='a', 
                 header=not os.path.exists('data/karachi_aqi.csv'))
        print("⚠️ Saved locally as backup")

# Usage example
if __name__ == "__main__":
    store = FeatureStore()
    
    test_data = [{
        'city': 'Karachi',
        'timestamp': datetime.now(),
        'aqi': 150,
        'pm25': 120
    }]
    
    if store.can_connect:
        store.store_features(test_data)
    else:
        print("Using local storage only")
        store._store_locally(pd.DataFrame(test_data))
