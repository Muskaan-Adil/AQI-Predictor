import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(
    page_title="Karachi AQI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

def generate_mock_data():
    current = {
        'pm25': 45.3,
        'pm10': 78.9,
        'temperature': 32.4,  # Corrected to lowercase
        'humidity': 65,
        'wind_speed': 12.5,
        'pressure': 1013
    }
    
    dates = pd.date_range(start="2024-01-01", periods=3).strftime("%Y-%m-%d").tolist()
    forecast = {
        'date': dates,
        'pm2.5': [current['pm25'] * (1 + i*0.15) for i in range(1,4)],
        'pm10': [current['pm10'] * (1 + i*0.12) for i in range(1,4)]
    }
    
    return current, forecast

# Feature importance visualization
def mock_shap_explanation():
    features = ['Temperature', 'Humidity', 'Wind Speed', 'Traffic', 'Industrial Activity']
    values = np.random.randn(5) * 10
    fig, ax = plt.subplots()
    ax.barh(features, values, color=['#FF6B6B' if x > 0 else '#4ECDC4' for x in values])
    ax.set_title("Factors Affecting Air Quality")
    ax.set_xlabel("Impact on AQI")
    plt.tight_layout()
    return fig

# Main dashboard
def main():
    st.sidebar.title("Settings")
    
    # City selection (only Karachi available)
    selected_city = st.sidebar.selectbox(
        "Select City",
        ["Karachi"],
        key='city_select'
    )
    
    # Pollutant selection
    pollutant = st.sidebar.selectbox(
        "Select Pollutant",
        ['PM2.5', 'PM10'],
        key='pollutant_select'
    )
    
    st.title(f"Karachi Air Quality Dashboard")
    
    # Generate mock data
    current, forecast = generate_mock_data()
    
    # Current metrics
    st.subheader(f"Current Air Quality in Karachi")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PM2.5", f"{current['pm25']:.1f} µg/m³", "Moderate")
        st.metric("Temperature", f"{current['temperature']:.1f} °C")
    with col2:
        st.metric("PM10", f"{current['pm10']:.1f} µg/m³", "Moderate")
        st.metric("Humidity", f"{current['humidity']:.0f}%")
    with col3:
        st.metric("Wind Speed", f"{current['wind_speed']:.1f} m/s")
        st.metric("Pressure", f"{current['pressure']:.1f} hPa")
    
    # Forecast visualization
    st.subheader("3-Day Air Quality Forecast")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast['date'],
        y=forecast[pollutant.lower()],
        mode='lines+markers',
        line=dict(color='#FFA15A', width=3),
        name=f"{pollutant} Forecast"
    ))
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title=f'{pollutant} (µg/m³)',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Factor analysis
    st.subheader("Key Contributing Factors")
    shap_fig = mock_shap_explanation()
    st.pyplot(shap_fig)

if __name__ == "__main__":
    main()
