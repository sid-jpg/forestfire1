// Initialize map
let map = L.map('map').setView([20.5937, 78.9629], 5);  // Center on India
let marker = null;

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: ' OpenStreetMap contributors'
}).addTo(map);

// Function to update coordinate inputs
function updateCoordinates(lat, lng) {
    document.getElementById('latitude').value = lat.toFixed(6);
    document.getElementById('longitude').value = lng.toFixed(6);
}

// Function to update the results display
function updateResults(data) {
    // Update risk level and probabilities
    document.getElementById('riskLevel').textContent = data.risk_level;
    document.getElementById('probability').textContent = data.probability;
    document.getElementById('tempProbability').textContent = data.temp_probability;
    document.getElementById('vegProbability').textContent = data.veg_probability;
    
    // Update weather information
    document.getElementById('temperature').textContent = data.weather.temperature;
    document.getElementById('humidity').textContent = data.weather.humidity;
    document.getElementById('windSpeed').textContent = data.weather.wind_speed;
    document.getElementById('pressure').textContent = data.weather.pressure;
    document.getElementById('ndvi').textContent = data.weather.ndvi;
    document.getElementById('rainProb').textContent = data.weather.rain_probability;
    document.getElementById('snowProb').textContent = data.weather.snow_probability;
    
    // Show results container and prediction panel
    document.getElementById('results').style.display = 'block';
    document.querySelector('.prediction-panel').classList.add('active');
}

// Function to handle prediction request
async function getPrediction(lat, lon) {
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ lat, lon })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        updateResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error getting prediction: ' + error.message);
    }
}

// Handle map clicks
map.on('click', function(e) {
    const lat = e.latlng.lat;
    const lng = e.latlng.lng;
    
    // Update marker position
    if (marker) {
        marker.setLatLng([lat, lng]);
    } else {
        marker = L.marker([lat, lng]).addTo(map);
    }
    
    // Update coordinate inputs
    updateCoordinates(lat, lng);
});

// Handle predict button click
document.getElementById('predictBtn').addEventListener('click', function() {
    const lat = document.getElementById('latitude').value;
    const lng = document.getElementById('longitude').value;
    
    if (!lat || !lng) {
        alert('Please select a location on the map first');
        return;
    }
    
    getPrediction(parseFloat(lat), parseFloat(lng));
});
