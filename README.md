# Forest Fire Prediction System

A machine learning-based web application that predicts forest fire risks using both temperature and vegetation data.

## Features

- Interactive map interface using Leaflet.js
- Dual prediction models:
  - Temperature-based prediction using weather data
  - Vegetation-based prediction using satellite data
- Real-time weather data from OpenWeatherMap API
- Satellite data integration (NDVI and LST)
- Dark-themed modern UI
- Responsive design for all devices

## Prerequisites

- Python 3.10 or higher
- pip package manager
- API keys for:
  - OpenWeatherMap
  - NASA FIRMS (Fire Information for Resource Management System)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd major
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `config.py` file in the root directory with your API keys:
```python
OPENWEATHER_API_KEY = "your_openweather_api_key"
NASA_API_KEY = "your_nasa_api_key"
```

## Project Structure

```
major/
├── app.py              # Main Flask application
├── main.py            # Core prediction logic
├── satellite_data.py  # Satellite data processing
├── config.py         # API keys and configuration
├── requirements.txt  # Python dependencies
├── models/          # Trained ML models
│   ├── temp.joblib
│   └── veg.joblib
├── static/         # Static files
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── map.js
└── templates/     # HTML templates
    └── index.html
```

## Running the Application

1. Make sure your virtual environment is activated:
```bash
venv\Scripts\activate
```

2. Start the Flask application:
```bash
python main.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. The application will display a map interface
2. Click anywhere on the map to:
   - Get current weather data for that location
   - Retrieve satellite data (NDVI and LST)
   - Generate fire risk predictions
3. View the results showing:
   - Temperature-based prediction
   - Vegetation-based prediction
   - Overall risk assessment
   - All parameters used in the prediction

## Models

### Temperature Model
- Uses Random Forest Classifier
- Features:
  - Temperature (°C)
  - Relative Humidity (%)
  - Wind Speed (m/s)
  - Rainfall (mm)

### Vegetation Model
- Uses Random Forest Classifier
- Features:
  - NDVI (Normalized Difference Vegetation Index)
  - LST (Land Surface Temperature)
  - Burned Area

## Error Handling

The application includes comprehensive error handling for:
- API failures
- Model prediction errors
- Invalid coordinates
- Missing data

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details
