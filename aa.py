import ee

# Initialize the Earth Engine API
ee.Initialize()

def get_ndvi(lat, lon, start_date, end_date):
    # Define the region of interest
    roi = ee.Geometry.Point([lon, lat])
    
    # Load Sentinel-2 data
    s2_collection = ee.ImageCollection('COPERNICUS/S2') \
                        .filterDate(start_date, end_date) \
                        .filterBounds(roi)
    
    # Calculate NDVI
    ndvi = s2_collection.median().normalizedDifference(['B8', 'B4']).clip(roi)
    
    # Get the mean NDVI value for the region
    ndvi_value = ndvi.reduceRegion(ee.Reducer.mean(), roi, scale=10).get('nd').getInfo()
    
    return ndvi_value

# Example usage
lat, lon = 20.5937, 78.9629
start_date, end_date = '2023-01-01', '2023-01-31'
ndvi = get_ndvi(lat, lon, start_date, end_date)
print(f'NDVI: {ndvi}')