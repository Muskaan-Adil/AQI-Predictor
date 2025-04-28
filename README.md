# Pearls AQI Predictor
A serverless application for predicting PM2.5 and PM10 values for multiple cities over the next 3 days.

## Overview
Pearls AQI Predictor is a comprehensive air quality prediction system built with a 100% serverless architecture. It fetches current air quality and weather data, processes features, trains multiple ML models, and provides forecasts through an interactive Streamlit dashboard.

## Features
- Multi-city PM2.5 and PM10 prediction
- Historical data backfill for model training
- Multiple regression models with automatic model selection
- Feature importance analysis using SHAP
- Interactive dashboard for visualizing predictions
- Automated CI/CD pipeline for continuous data collection and model training

## Architecture
The application follows a modular architecture:
1. **Data Collection** - Fetches data from AQICN and OpenWeather APIs
2. **Feature Engineering** - Computes features from raw data and stores them in Hopsworks
3. **Model Training** - Trains and evaluates multiple models to predict PM2.5/PM10
4. **Prediction Display** - Streamlit dashboard showing forecasts and insights
5. **Automation** - GitHub Actions for scheduling pipeline runs

## Setup and Installation
1. Clone the repository: https://github.com/Muskaan-Adil/AQI-Predictor.git

3. Install dependencies: pip install -r requirements.txt

5. Set up environment variables:
  - AQICN_API_KEY=your_aqicn_api_key
  - OPENWEATHER_API_KEY=your_openweather_api_key
  - HOPSWORKS_API_KEY=your_hopsworks_api_key
  - HOPSWORKS_PROJECT_NAME=your_hopsworks_project

7. Run the backfill to collect historical data: python -m src.feature_engineering.backfill

8. Train the models: python -m src.models.model_trainer

9. Launch the dashboard: streamlit run src.dashboard.app

## Using the Dashboard
The dashboard allows you to:
- Select a city to view its air quality data
- Choose between PM2.5 and PM10 predictions
- View current conditions and 3-day forecasts
- Examine historical trends
- Understand which features most influence the predictions

## CI/CD Pipeline
The project includes GitHub Actions workflows:
- `feature_pipeline.yml` - Runs hourly to collect new data and update features
- `training_pipeline.yml` - Runs daily to retrain models with new data
