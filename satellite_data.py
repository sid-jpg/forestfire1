import requests
import numpy as np
from datetime import datetime, timedelta
from config import NASA_API_KEY
import ee

# Correct API endpoint and parameters
MODIS_API_URL = "https://modis.ornl.gov/rst/api/v1/subset"

# Initialize the Earth Engine API
try:
    ee.Initialize()
except Exception as e:
    print("Error initializing Earth Engine API:", e)

def get_modis_data(lat, lon):
    """
    Get MODIS satellite data (NDVI and LST) for a given location
    """
    
    # Calculate date range for the last 8 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=8)
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'band': 'NDVI,LST_Day_1km',
        'startDate': start_date.strftime('%Y-%m-%d'),
        'endDate': end_date.strftime('%Y-%m-%d'),
        'apikey': NASA_API_KEY
    }
    
    try:
        response = requests.get(MODIS_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract NDVI and LST values
        ndvi = np.mean(data['NDVI']) if 'NDVI' in data else 0.0
        lst = np.mean(data['LST_Day_1km']) if 'LST_Day_1km' in data else 0.0
        
        # Normalize values
        ndvi = max(min(ndvi, 1.0), -1.0)  # NDVI ranges from -1 to 1
        lst = lst * 0.02 - 273.15  # Convert to Celsius
        
        return {
            'NDVI': ndvi,
            'LST': lst,
            'BURNED_AREA': 0.0  # Placeholder for burned area data
        }
        
    except requests.RequestException as e:
        print(f"Error fetching satellite data: {e}")
        return None

def get_burned_area(lat, lon):
    """
    Get burned area information from NASA FIRMS (Fire Information for Resource Management System)
    """
    base_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    
    # Calculate date range (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'radius': 1.0,  # 1km radius
        'startDate': start_date.strftime('%Y-%m-%d'),
        'endDate': end_date.strftime('%Y-%m-%d'),
        'apikey': NASA_API_KEY
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        # Parse CSV response to get burned area
        # This is a simplified calculation
        if len(response.text.split('\n')) > 1:
            return 1.0  # Area has recent fire activity
        return 0.0  # No recent fire activity
        
    except requests.RequestException as e:
        print(f"Error fetching burned area data: {e}")
        return 0.0

def get_ndvi_lst(lat, lon, start_date, end_date):
    """
    Fetch NDVI and LST data for a given location and date range using Google Earth Engine.
    """
    # Define the region of interest
    roi = ee.Geometry.Point([lon, lat])
    
    # Load MODIS NDVI data
    ndvi_collection = ee.ImageCollection('MODIS/006/MOD13A1') \
                        .filterDate(start_date, end_date) \
                        .select('NDVI')
    
    # Calculate the median NDVI
    ndvi = ndvi_collection.median().multiply(0.0001).clip(roi)
    
    # Load MODIS LST data
    lst_collection = ee.ImageCollection('MODIS/006/MOD11A1') \
                       .filterDate(start_date, end_date) \
                       .select('LST_Day_1km')
    
    # Calculate the median LST and convert to Celsius
    lst = lst_collection.median().multiply(0.02).subtract(273.15).clip(roi)
    
    # Get the mean NDVI and LST values for the region
    ndvi_value = ndvi.reduceRegion(ee.Reducer.mean(), roi, scale=500).get('NDVI').getInfo()
    lst_value = lst.reduceRegion(ee.Reducer.mean(), roi, scale=1000).get('LST_Day_1km').getInfo()
    
    return ndvi_value, lst_value
