import joblib
import numpy as np
import requests
from geopy.geocoders import Nominatim
from config import OPENWEATHER_API_KEY
import math
from datetime import datetime
import random

def get_weather_data(lat, lon):
    """Get weather data from OpenWeather API"""
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def calculate_ndvi(nir_band, red_band):
    """Calculate NDVI from NIR and Red bands"""
    nir = float(nir_band)
    red = float(red_band)
    return (nir - red) / (nir + red)

def calculate_feels_like(temp, humidity, wind_speed):
    """Calculate feels like temperature"""
    temp_f = (temp - 273.15) * 9/5 + 32
    
    if temp_f <= 50 and wind_speed > 3:
        wind_speed_mph = wind_speed * 2.237
        wind_chill = 35.74 + (0.6215 * temp_f) - (35.75 * wind_speed_mph**0.16) + (0.4275 * temp_f * wind_speed_mph**0.16)
        return (wind_chill - 32) * 5/9 + 273.15
    elif temp_f >= 80:
        heat_index = -42.379 + (2.04901523 * temp_f) + (10.14333127 * humidity) - \
                    (0.22475541 * temp_f * humidity) - (6.83783e-3 * temp_f**2) - \
                    (5.481717e-2 * humidity**2) + (1.22874e-3 * temp_f**2 * humidity) + \
                    (8.5282e-4 * temp_f * humidity**2) - (1.99e-6 * temp_f**2 * humidity**2)
        return (heat_index - 32) * 5/9 + 273.15
    else:
        return temp

def calculate_rain_snow(temp, humidity, pressure):
    """Calculate rain and snow probability"""
    dew_point = temp - ((100 - humidity) / 5)
    rain_prob = max(0, min(1, (humidity / 100) * (1 - abs(temp - dew_point) / 20)))
    snow_prob = rain_prob if temp < 273.15 else 0
    return rain_prob, snow_prob

def calculate_wind_features(wind_speed, lat, lon):
    """Calculate wind features"""
    wind_direction = (math.atan2(lat, lon) * 180 / math.pi + 360) % 360
    wind_gust = wind_speed * (1 + 0.2 * random.random())
    return wind_direction, wind_gust

def get_combined_prediction(lat, lon, nir_value=0.5, red_value=0.3, burned_area=0.0):
    """Get combined prediction from both models"""
    try:
        # Load models
        temp_model = joblib.load('models/temp.joblib')
        veg_model = joblib.load('models/veg.joblib')
        
        # Get weather data
        weather_data = get_weather_data(lat, lon)
        if not weather_data:
            return "Error: Could not fetch weather data"
        
        # Extract temperature data
        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        pressure = weather_data['main']['pressure']
        wind_speed = weather_data['wind'].get('speed', 0)
        
        # Calculate additional features
        feels_like = calculate_feels_like(temp, humidity, wind_speed)
        rain_prob, snow_prob = calculate_rain_snow(temp, humidity, pressure)
        wind_direction, wind_gust = calculate_wind_features(wind_speed, lat, lon)
        
        # Current month for seasonal features
        current_month = datetime.now().month
        
        # Temperature model features
        temp_features = np.array([[
            temp,                    # Temperature
            humidity,                # Humidity
            pressure,                # Pressure
            wind_speed,              # Wind Speed
            feels_like,              # Feels Like Temperature
            rain_prob,               # Rain Probability
            snow_prob,               # Snow Probability
            wind_direction,          # Wind Direction
            wind_gust,               # Wind Gust
            current_month,           # Month
            lat,                     # Latitude
            lon,                     # Longitude
            abs(lat),                # Absolute Latitude
            pressure - 1013.25       # Pressure Difference from Standard
        ]])
        
        temp_pred_proba = temp_model.predict_proba(temp_features)[0][1]  # Probability of fire
        
        # Calculate NDVI
        ndvi = calculate_ndvi(nir_value, red_value)
        lst = temp  # Using current temperature as LST
        
        # Vegetation model prediction
        veg_features = np.array([[ndvi, lst, burned_area]])
        veg_pred_proba = veg_model.predict_proba(veg_features)[0][1]  # Probability of fire
        
        # Calculate average prediction
        avg_prediction = (temp_pred_proba + veg_pred_proba) / 2
        
        return {
            'temperature_prediction': float(temp_pred_proba),
            'vegetation_prediction': float(veg_pred_proba),
            'average_prediction': float(avg_prediction),
            'risk_level': get_risk_level(avg_prediction),
            'weather_data': {
                'temperature': temp - 273.15,  # Convert to Celsius
                'humidity': humidity,
                'pressure': pressure,
                'wind_speed': wind_speed,
                'ndvi': ndvi,
                'rain_probability': rain_prob * 100,
                'snow_probability': snow_prob * 100
            }
        }
        
    except Exception as e:
        return f"Error: {str(e)}"

def get_risk_level(probability):
    """Convert probability to risk level"""
    if probability <= 0.25:
        return "Low Risk"
    elif probability <= 0.50:
        return "Moderate Risk"
    elif probability <= 0.75:
        return "High Risk"
    else:
        return "Extreme Risk"

if __name__ == "__main__":
    # Example coordinates (New Delhi, India)
    lat, lon = 28.6139, 77.2090
    
    # Get prediction
    result = get_combined_prediction(lat, lon)
    
    # Print results
    if isinstance(result, dict):
        print("\nPrediction Results:")
        print("-" * 50)
        print(f"Temperature Model Prediction: {result['temperature_prediction']:.2%}")
        print(f"Vegetation Model Prediction: {result['vegetation_prediction']:.2%}")
        print(f"Average Prediction: {result['average_prediction']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print("\nWeather Data:")
        print("-" * 50)
        print(f"Temperature: {result['weather_data']['temperature']:.1f}°C")
        print(f"Humidity: {result['weather_data']['humidity']}%")
        print(f"Pressure: {result['weather_data']['pressure']} hPa")
        print(f"Wind Speed: {result['weather_data']['wind_speed']} m/s")
        print(f"NDVI: {result['weather_data']['ndvi']:.3f}")
        print(f"Rain Probability: {result['weather_data']['rain_probability']:.1f}%")
        print(f"Snow Probability: {result['weather_data']['snow_probability']:.1f}%")
    else:
        print(f"Error: {result}")
