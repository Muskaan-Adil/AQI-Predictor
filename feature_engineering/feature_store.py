import os
import pandas as pd
import requests
from datetime import datetime

class FeatureStore:
    def __init__(self):
        self.project_numid = "1219758"  # From your URL
        self.api_key = os.getenv('HOPSWORKS_API_KEY')
        
        # Bypass HSFS and use direct REST API
        self.base_url = f"https://c.app.hopsworks.ai:443/hopsworks-api/api/project/{self.project_numid}"
        self.headers = {
            "Authorization": f"ApiKey {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Verify access
        self.can_connect = self._verify_access()

    def _verify_access(self):
        """Direct REST API verification"""
        try:
            response = requests.get(
                f"{self.base_url}/dataset",
                headers=self.headers,
                verify=False,
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False

    def store_features(self, features):
        """Foolproof storage with direct API calls"""
        df = pd.DataFrame(features)
        
        # 1. Always store locally first
        self._store_locally(df)
        
        # 2. Try Hopsworks if verified
        if self.can_connect:
            self._store_via_rest(df)

    def _store_via_rest(self, df):
        """Direct storage using REST API"""
        try:
            # Convert data
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6
            records = df.to_dict('records')
            
            # Create feature group if needed
            fg_name = "aqi_prediction"
            self._create_feature_group(fg_name)
            
            # Insert data
            response = requests.post(
                f"{self.base_url}/featurestores/{self.project_numid}/featuregroups/{fg_name}/insert",
                headers=self.headers,
                json=records,
                verify=False
            )
            
            if response.status_code == 200:
                print("Successfully stored via REST API")
            else:
                print(f"REST insert failed: {response.text}")
                
        except Exception as e:
            print(f"REST storage failed: {str(e)}")

    def _create_feature_group(self, name):
        """Ensure feature group exists"""
        try:
            requests.post(
                f"{self.base_url}/featurestores/{self.project_numid}/featuregroups",
                headers=self.headers,
                json={
                    "name": name,
                    "version": 1,
                    "primaryKey": ["timestamp"],
                    "eventTime": "timestamp"
                },
                verify=False
            )
        except Exception:
            pass

    def _store_locally(self, df):
        """Reliable local storage"""
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/karachi_aqi.csv', mode='a', header=not os.path.exists('data/karachi_aqi.csv'))
