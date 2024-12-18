from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import requests
import joblib
import math
from datetime import datetime, timedelta
import os
import json
from config import OPENWEATHER_API_KEY, MODEL_CONFIG, NASA_FIRMS_API_KEY
import time

app = Flask(__name__)

# Load temperature model
temp_model = joblib.load('models/temp.joblib')

# Print feature names for debugging
print("Temperature Model Features:", temp_model.feature_names_in_ if hasattr(temp_model, 'feature_names_in_') else "No feature names found")

def get_temperature(lat, lon):
    """Get temperature data"""
    try:
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
        weather_response = requests.get(weather_url)
        
        if weather_response.status_code == 200:
            weather_data = weather_response.json()
            return weather_data['main']['temp']
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting temperature data: {str(e)}")
        return None

def get_humidity(lat, lon):
    """Get humidity data"""
    try:
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
        weather_response = requests.get(weather_url)
        
        if weather_response.status_code == 200:
            weather_data = weather_response.json()
            return weather_data['main']['humidity']
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting humidity data: {str(e)}")
        return None

def get_wind_speed(lat, lon):
    """Get wind speed data"""
    try:
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
        weather_response = requests.get(weather_url)
        
        if weather_response.status_code == 200:
            weather_data = weather_response.json()
            return weather_data['wind'].get('speed', 0)
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting wind speed data: {str(e)}")
        return None

def get_precipitation(lat, lon):
    """Get precipitation data"""
    try:
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
        weather_response = requests.get(weather_url)
        
        if weather_response.status_code == 200:
            weather_data = weather_response.json()
            return weather_data.get('rain', {}).get('1h', 0)  # Rain in last hour
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting precipitation data: {str(e)}")
        return None

def calculate_ffmc(temp, humidity, wind, rain):
    """Calculate Fine Fuel Moisture Code"""
    # Convert temperature to Celsius
    temp_c = temp - 273.15
    
    # Initial FFMC moisture content
    mo = 147.2 * (101 - 85) / (59.5 + 85)  # Using average value
    
    # Rain effect
    if rain > 0.5:
        rf = rain - 0.5
        mr = mo + 42.5 * rf * np.exp(-100 / (251 - mo)) * (1 - np.exp(-6.93 / rf))
    else:
        mr = mo

    # Temperature and humidity effect
    ed = 0.942 * (humidity / 100) * 6.1078 * np.exp((17.27 * temp_c) / (237.3 + temp_c))
    ew = 0.618 * (humidity / 100) * 6.1078 * np.exp((17.27 * temp_c) / (237.3 + temp_c))
    
    # Drying and wetting factors
    ko = 0.424 * (1 - (humidity/100)**1.7) + 0.0694 * np.sqrt(wind) * (1 - (humidity/100)**8)
    kd = ko * 0.581 * np.exp(0.0365 * temp_c)
    
    # Final FFMC
    m = mr + (1000 * kd)
    ffmc = 59.5 * (250 - m) / (147.2 + m)
    
    return max(0, min(101, ffmc))

def calculate_dmc(temp, humidity, rain, prev_dmc=6):
    """Calculate Duff Moisture Code"""
    temp_c = temp - 273.15
    
    # Rain effect
    if rain > 1.5:
        re = 0.92 * rain - 1.27
        mo = 20 + np.exp(5.6348 - prev_dmc / 43.43)
        b = 100 / (0.5 + 0.3 * prev_dmc)
        mr = mo + 1000 * re / (48.77 + b * re)
        pr = 244.72 - 43.43 * np.log(mr - 20)
    else:
        pr = prev_dmc
        
    # Temperature and humidity effect
    k = 1.894 * (temp_c + 1.1) * (100 - humidity) * 1e-6
    
    return max(0, pr + 100 * k)

def calculate_dc(temp, rain, prev_dc=15):
    """Calculate Drought Code"""
    temp_c = temp - 273.15
    
    # Rain effect
    if rain > 2.8:
        rd = 0.83 * rain - 1.27
        Qo = 800 * np.exp(-prev_dc / 400)
        Qr = Qo + 3.937 * rd
        dr = 400 * np.log(800 / Qr)
    else:
        dr = prev_dc
        
    # Temperature effect
    V = 0.36 * (temp_c + 2.8) + 0.5
    
    return max(0, dr + 0.5 * V)

def calculate_isi(ffmc, wind):
    """Calculate Initial Spread Index"""
    f = np.exp(2.72 * (0.434 * np.log(101 - ffmc)) ** 0.647)
    return max(0, 0.208 * f * wind)

def calculate_bui(dmc, dc):
    """Calculate Buildup Index"""
    if dmc <= 0.4 * dc:
        return max(0, 0.8 * dmc * dc / (dmc + 0.4 * dc))
    else:
        return max(0, dmc - (1 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7))

def calculate_fwi(isi, bui):
    """Calculate Fire Weather Index"""
    if bui <= 80:
        fD = 0.626 * bui ** 0.809 + 2
    else:
        fD = 1000 / (25 + 108.64 * np.exp(-0.023 * bui))
    
    B = 0.1 * isi * fD
    
    if B > 1:
        return np.exp(2.72 * (0.434 * np.log(B)) ** 0.647)
    else:
        return B

@app.route('/')
def index():
    """Landing page route"""
    return render_template('landing.html')

@app.route('/predict')
def predict_page():
    """Main application route"""
    return render_template('index.html')

@app.route('/live')
def live_map():
    return render_template('live_map.html')

@app.route('/get_fire_data')
def get_fire_data():
    try:
        # Base URL for NASA FIRMS API
        base_url = "https://firms.modaps.eosdis.nasa.gov/api/area/json"
        
        # Parameters for India's bounding box
        params = {
            'key': NASA_FIRMS_API_KEY,
            'country': 'INDIA',
            'time': '24',  # Last 24 hours
        }
        
        all_fire_data = []
        
        # Fetch Landsat data without caching
        landsat_params = params.copy()
        landsat_params['source'] = 'LANDSAT_NRT'
        landsat_response = requests.get(base_url, params=landsat_params, timeout=30)
        if landsat_response.status_code == 200:
            landsat_data = landsat_response.json()
            for fire in landsat_data:
                fire['satellite'] = 'landsat'
            all_fire_data.extend(landsat_data)
        
        # Fetch VIIRS data without caching
        viirs_params = params.copy()
        viirs_params['source'] = 'VIIRS_SNPP_NRT'
        viirs_response = requests.get(base_url, params=viirs_params, timeout=30)
        if viirs_response.status_code == 200:
            viirs_data = viirs_response.json()
            for fire in viirs_data:
                fire['satellite'] = 'viirs'
            all_fire_data.extend(viirs_data)
        
        # Fetch MODIS data without caching
        modis_params = params.copy()
        modis_params['source'] = 'MODIS_NRT'
        modis_response = requests.get(base_url, params=modis_params, timeout=30)
        if modis_response.status_code == 200:
            modis_data = modis_response.json()
            for fire in modis_data:
                fire['satellite'] = 'modis'
            all_fire_data.extend(modis_data)
        
        return jsonify(all_fire_data)
        
    except Exception as e:
        print(f"Error fetching fire data: {str(e)}")
        return jsonify([])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        lat = float(data['lat'])
        lon = float(data['lon'])

        # Input validation
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({'error': 'Invalid coordinates'}), 400

        # Get weather data
        try:
            weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
            weather_response = requests.get(weather_url)
            
            if weather_response.status_code == 200:
                weather_data = weather_response.json()
            else:
                return jsonify({'error': f'Weather API error: {weather_response.status_code}'}), 503
        except requests.exceptions.RequestException as e:
            return jsonify({'error': f'Weather API request failed: {str(e)}'}), 503

        # Extract weather features
        try:
            temp = weather_data['main']['temp']
            humidity = weather_data['main']['humidity']
            pressure = weather_data['main']['pressure']
            wind_speed = weather_data['wind'].get('speed', 0)
            rain = weather_data.get('rain', {}).get('1h', 0)  # Rain in last hour
        except KeyError as e:
            return jsonify({'error': f'Missing weather data: {str(e)}'}), 503

        # Calculate Fire Weather Indices
        ffmc = calculate_ffmc(temp, humidity, wind_speed, rain)
        dmc = calculate_dmc(temp, humidity, rain)
        dc = calculate_dc(temp, rain)
        isi = calculate_isi(ffmc, wind_speed)
        bui = calculate_bui(dmc, dc)
        fwi = calculate_fwi(isi, bui)

        # Prepare model features with correct column names
        try:
            current_date = datetime.now()
            current_month = current_date.month
            current_day = current_date.day
            current_year = current_date.year
            
            # Temperature model features in exact order from training
            temp_features = pd.DataFrame([[
                current_day,    # day
                current_month,  # month
                current_year,   # year
                temp - 273.15,  # Temperature in Celsius
                humidity,       # RH
                wind_speed,     # Ws
                rain,          # Rain
                ffmc,          # FFMC
                dmc,           # DMC
                dc,            # DC
                isi,           # ISI
                bui,           # BUI
                fwi,           # FWI
                1              # Region (default to region 1)
            ]], columns=[
                'day',
                'month',
                'year',
                'Temperature',
                'RH',
                'Ws',
                'Rain',
                'FFMC',
                'DMC',
                'DC',
                'ISI',
                'BUI',
                'FWI',
                'Region'
            ])

            # Make prediction using temperature model only
            temp_prob = temp_model.predict_proba(temp_features)[0][1]

            # Determine risk level based on temperature model probability
            if temp_prob <= 0.3:
                risk_level = "Low Risk"
            elif temp_prob <= 0.6:
                risk_level = "Moderate Risk"
            elif temp_prob <= 0.8:
                risk_level = "High Risk"
            else:
                risk_level = "Extreme Risk"

            return jsonify({
                'risk_level': risk_level,
                'probability': round(temp_prob * 100, 2),
                'weather': {
                    'temperature': round(temp - 273.15, 2),
                    'humidity': humidity,
                    'wind_speed': wind_speed,
                    'pressure': pressure,
                    'rain': rain,
                    'ffmc': round(ffmc, 2),
                    'dmc': round(dmc, 2),
                    'dc': round(dc, 2),
                    'isi': round(isi, 2),
                    'bui': round(bui, 2),
                    'fwi': round(fwi, 2)
                }
            })

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return jsonify({'error': 'Error calculating risk level'}), 500

    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        print(f"Unexpected error in predict route: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/predict_new', methods=['POST'])
def make_prediction():
    try:
        # Get input values from the form
        data = request.get_json()
        lat = float(data['lat'])
        lon = float(data['lon'])

        # Input validation
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({'error': 'Invalid coordinates'}), 400

        # Get weather data
        try:
            weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
            weather_response = requests.get(weather_url)
            
            if weather_response.status_code == 200:
                weather_data = weather_response.json()
            else:
                return jsonify({'error': f'Weather API error: {weather_response.status_code}'}), 503
        except requests.exceptions.RequestException as e:
            return jsonify({'error': f'Weather API request failed: {str(e)}'}), 503

        # Extract weather features
        try:
            temp = weather_data['main']['temp']
            humidity = weather_data['main']['humidity']
            pressure = weather_data['main']['pressure']
            wind_speed = weather_data['wind'].get('speed', 0)
            rain = weather_data.get('rain', {}).get('1h', 0)  # Rain in last hour
        except KeyError as e:
            return jsonify({'error': f'Missing weather data: {str(e)}'}), 503

        # Calculate Fire Weather Indices
        ffmc = calculate_ffmc(temp, humidity, wind_speed, rain)
        dmc = calculate_dmc(temp, humidity, rain)
        dc = calculate_dc(temp, rain)
        isi = calculate_isi(ffmc, wind_speed)
        bui = calculate_bui(dmc, dc)
        fwi = calculate_fwi(isi, bui)

        # Prepare model features with correct column names
        try:
            current_date = datetime.now()
            current_month = current_date.month
            current_day = current_date.day
            current_year = current_date.year
            
            # Temperature model features in exact order from training
            temp_features = pd.DataFrame([[
                current_day,    # day
                current_month,  # month
                current_year,   # year
                temp - 273.15,  # Temperature in Celsius
                humidity,       # RH
                wind_speed,     # Ws
                rain,          # Rain
                ffmc,          # FFMC
                dmc,           # DMC
                dc,            # DC
                isi,           # ISI
                bui,           # BUI
                fwi,           # FWI
                1              # Region (default to region 1)
            ]], columns=[
                'day',
                'month',
                'year',
                'Temperature',
                'RH',
                'Ws',
                'Rain',
                'FFMC',
                'DMC',
                'DC',
                'ISI',
                'BUI',
                'FWI',
                'Region'
            ])

            # Make prediction using temperature model only
            temp_prob = temp_model.predict_proba(temp_features)[0][1]

            # Determine risk level based on temperature model probability
            if temp_prob <= 0.3:
                risk_level = "Low Risk"
            elif temp_prob <= 0.6:
                risk_level = "Moderate Risk"
            elif temp_prob <= 0.8:
                risk_level = "High Risk"
            else:
                risk_level = "Extreme Risk"

            return jsonify({
                'success': True,
                'risk_level': risk_level,
                'probability': round(temp_prob * 100, 2),
                'weather': {
                    'temperature': round(temp - 273.15, 2),
                    'humidity': humidity,
                    'wind_speed': wind_speed,
                    'pressure': pressure,
                    'rain': rain,
                    'ffmc': round(ffmc, 2),
                    'dmc': round(dmc, 2),
                    'dc': round(dc, 2),
                    'isi': round(isi, 2),
                    'bui': round(bui, 2),
                    'fwi': round(fwi, 2)
                }
            })

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return jsonify({'success': False, 'error': 'Error calculating risk level'}), 500

    except ValueError as e:
        return jsonify({'success': False, 'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        print(f"Unexpected error in predict route: {str(e)}")
        return jsonify({'success': False, 'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)
