from flask import Flask, render_template, request, jsonify
import os
import requests
from geopy.geocoders import Nominatim
import joblib
import numpy as np
from config import OPENWEATHER_API_KEY
from satellite_data import get_modis_data, get_burned_area

app = Flask(__name__)

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

def get_weather_data(lat, lon):
    """Get weather data from OpenWeather API"""
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric'  # Get temperature in Celsius
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant parameters
        weather_params = {
            'Temperature': data['main']['temp'],
            'RH': data['main']['humidity'],
            'Ws': data['wind']['speed'],
            'Rain': data['rain']['1h'] if 'rain' in data and '1h' in data['rain'] else 0.0,
        }
        return weather_params
    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def get_location_name(lat, lon):
    """Get location name from coordinates"""
    geolocator = Nominatim(user_agent="forest_fire_predictor")
    try:
        location = geolocator.reverse((lat, lon), language='en')
        return location.address if location else f"Location ({lat}, {lon})"
    except Exception as e:
        print(f"Error getting location name: {e}")
        return f"Location ({lat}, {lon})"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        lat = float(data['lat'])
        lon = float(data['lon'])
        
        # Get weather data for temperature model
        weather_params = get_weather_data(lat, lon)
        if not weather_params:
            return jsonify({'error': 'Could not fetch weather data'})
        
        # Get satellite data for vegetation model
        satellite_params = get_modis_data(lat, lon)
        if not satellite_params:
            return jsonify({'error': 'Could not fetch satellite data'})
            
        # Get burned area data
        burned_area = get_burned_area(lat, lon)
        satellite_params['BURNED_AREA'] = burned_area
        
        # Get location name
        location_name = get_location_name(lat, lon)
        
        # Load and run temperature model
        try:
            temp_model = joblib.load('models/temp.joblib')
            temp_features = [
                weather_params['Temperature'],
                weather_params['RH'],
                weather_params['Ws'],
                weather_params['Rain']
            ]
            temp_prediction = temp_model.predict([temp_features])[0]
        except Exception as e:
            print(f"Error with temperature model: {e}")
            temp_prediction = "Model error"
            
        # Load and run vegetation model
        try:
            veg_model = joblib.load('models/veg.joblib')
            veg_features = [
                satellite_params['NDVI'],
                satellite_params['LST'],
                satellite_params['BURNED_AREA']
            ]
            veg_prediction = veg_model.predict([veg_features])[0]
        except Exception as e:
            print(f"Error with vegetation model: {e}")
            veg_prediction = "Model error"
        
        return jsonify({
            'location': location_name,
            'parameters_used': {
                'Weather Parameters': {
                    'Temperature': f"{weather_params['Temperature']}°C",
                    'Relative Humidity': f"{weather_params['RH']}%",
                    'Wind Speed': f"{weather_params['Ws']} m/s",
                    'Rain': f"{weather_params['Rain']} mm"
                },
                'Vegetation Parameters': {
                    'NDVI': f"{satellite_params['NDVI']:.2f}",
                    'Land Surface Temperature': f"{satellite_params['LST']}°C",
                    'Burned Area Present': "Yes" if burned_area > 0 else "No"
                }
            },
            'predictions': {
                'temperature_based': temp_prediction,
                'vegetation_based': veg_prediction,
                'overall_risk': 'High' if (temp_prediction == 'fire' and veg_prediction == 'fire') else 'Medium' if (temp_prediction == 'fire' or veg_prediction == 'fire') else 'Low'
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
