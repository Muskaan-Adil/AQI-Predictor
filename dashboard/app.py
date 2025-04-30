import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="AQI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

def generate_mock_data():
    current = {
        'pm25': 45.3,
        'pm10': 78.9,
        'temperature': 32.4,
        'humidity': 65,
        'wind_speed': 12.5,
        'pressure': 1013
    }
    
    # Get today's date and next 3 days
    today = datetime.now()
    dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 4)]
    
    forecast = {
        'date': dates,
        'pm2.5': [current['pm25'] * (1 + i*0.15) for i in range(1,4)],
        'pm10': [current['pm10'] * (1 + i*0.12) for i in range(1,4)]
    }
    
    return current, forecast

# Enhanced feature importance visualization
def enhanced_shap_explanation():
    features = ['Temperature', 'Humidity', 'Wind Speed', 'Traffic', 'Industrial Activity']
    values = np.random.randn(5) * 10
    
    # Create a more aesthetic horizontal bar chart with Plotly
    fig = go.Figure()
    
    for feature, value in zip(features, values):
        color = '#FF6B6B' if value > 0 else '#4ECDC4'
        fig.add_trace(go.Bar(
            y=[feature],
            x=[abs(value)],
            orientation='h',
            marker_color=color,
            name=feature,
            hovertemplate=f"<b>{feature}</b>: %{{x:.1f}}<extra></extra>",
            width=0.6
        ))
    
    fig.update_layout(
        title="<b>Key Factors Affecting Air Quality</b>",
        title_font_size=18,
        xaxis_title="<b>Impact on AQI (Positive = Increases Pollution)</b>",
        yaxis_title="<b>Factors</b>",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=100, r=50, b=80, t=80, pad=10),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Add impact direction indicators
    fig.update_traces(
        texttemplate='%{x:.1f}',
        textposition='outside',
        textfont_size=12
    )
    
    # Add reference line at zero
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
    
    return fig

# Main dashboard
def main():
    st.sidebar.title("Settings")
    
    # City selection
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
    
    st.title(f"Air Quality Dashboard")
    
    # Generate mock data
    current, forecast = generate_mock_data()
    
    # Current metrics
    st.subheader(f"Current Air Quality")
    
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
    
    # Enhanced factor analysis
    st.subheader("Key Contributing Factors")
    shap_fig = enhanced_shap_explanation()
    st.plotly_chart(shap_fig, use_container_width=True)

if __name__ == "__main__":
    main()
