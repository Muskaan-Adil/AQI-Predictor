import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_collection.data_collector import DataCollector
from src.feature_engineering.feature_generator import FeatureGenerator
from src.feature_engineering.feature_store import FeatureStore

def main():
    print("Collecting data...")
    dc = DataCollector()
    data = dc.collect_all_cities_data()
    
    print("Generating features...")
    fg = FeatureGenerator()
    features = fg.generate_all_features(data)
    
    print("Storing features...")
    fs = FeatureStore()
    fs.store_features(features)
    
    print("Feature pipeline completed successfully!")

if __name__ == "__main__":
    main()
