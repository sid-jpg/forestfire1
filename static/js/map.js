// Initialize map with dark theme
const map = L.map('map').setView([20.5937, 78.9629], 5);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: ' OpenStreetMap contributors'
}).addTo(map);

let marker;
let selectedLat, selectedLon;

// Add click event to map
map.on('click', function(e) {
    if (marker) {
        map.removeLayer(marker);
    }
    
    selectedLat = e.latlng.lat;
    selectedLon = e.latlng.lng;
    
    // Update input fields
    document.getElementById('latitude').value = selectedLat.toFixed(6);
    document.getElementById('longitude').value = selectedLon.toFixed(6);
    
    marker = L.marker([selectedLat, selectedLon]).addTo(map);
    
    // Update location info
    document.querySelector('.coordinates').textContent = 
        `${selectedLat.toFixed(4)}°N, ${selectedLon.toFixed(4)}°E`;
});

// Add click event to the predict button
document.getElementById('predictBtn').addEventListener('click', async function() {
    const lat = parseFloat(document.getElementById('latitude').value);
    const lon = parseFloat(document.getElementById('longitude').value);
    
    if (!lat || !lon || isNaN(lat) || isNaN(lon)) {
        showError('Please select a valid location on the map');
        return;
    }
    
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
        showError('Invalid coordinates range');
        return;
    }
    
    try {
        showLoading();
        const data = await predict(lat, lon);
        if (data) {
            updatePredictionPanel(data);
        }
    } catch (error) {
        handleError(error);
    } finally {
        hideLoading();
    }
});

async function predict(lat, lon) {
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
        return data;
    } catch (error) {
        console.error('Prediction error:', error);
        throw error;
    }
}

function updatePredictionPanel(data) {
    try {
        // Update risk level and probability
        const riskLevel = document.querySelector('.risk-level');
        riskLevel.textContent = data.risk_level;
        riskLevel.className = 'risk-level ' + data.risk_level.toLowerCase().replace(' ', '-');

        // Update progress bar
        const progress = document.querySelector('.progress');
        progress.style.width = `${data.probability}%`;
        progress.className = 'progress ' + data.risk_level.toLowerCase().replace(' ', '-');

        // Update weather parameters
        document.querySelector('.temp-value').textContent = `${data.weather.temperature.toFixed(1)}°C`;
        document.querySelector('.humidity-value').textContent = `${data.weather.humidity}%`;
        document.querySelector('.wind-value').textContent = `${data.weather.wind_speed.toFixed(1)} m/s`;
        document.querySelector('.rain-value').textContent = `${data.weather.rain} mm`;
        document.querySelector('.ffmc-value').textContent = data.weather.ffmc.toFixed(1);
        document.querySelector('.dmc-value').textContent = data.weather.dmc.toFixed(1);
        document.querySelector('.dc-value').textContent = data.weather.dc.toFixed(1);
        document.querySelector('.isi-value').textContent = data.weather.isi.toFixed(1);
        document.querySelector('.bui-value').textContent = data.weather.bui.toFixed(1);
        document.querySelector('.fwi-value').textContent = data.weather.fwi.toFixed(1);

        // Update risk probability
        document.querySelector('.risk-prob-value').textContent = `${data.probability}%`;

        // Show prediction panel
        const panel = document.querySelector('.prediction-panel');
        panel.classList.add('active');
    } catch (error) {
        console.error('Error updating prediction panel:', error);
        handleError(error);
    }
}

function showPredictionPanel() {
    const panel = document.querySelector('.prediction-panel');
    panel.classList.add('active');
}

function hidePredictionPanel() {
    const panel = document.querySelector('.prediction-panel');
    panel.classList.remove('active');
}

function animateValue(selector, value) {
    const element = document.querySelector(selector);
    if (element) {
        element.style.opacity = '0';
        setTimeout(() => {
            element.textContent = value;
            element.style.opacity = '1';
        }, 200);
    }
}

function handleError(error) {
    console.error('Error:', error);
    const riskLevel = document.querySelector('.risk-level');
    if (riskLevel) {
        riskLevel.textContent = 'Error: Could not calculate risk';
        riskLevel.style.color = 'var(--error-color)';
    }
    hideLoading();
}

// Show loading animation with fire icon
function showLoading() {
    const loadingOverlay = document.querySelector('.fire-loading');
    if (loadingOverlay) {
        loadingOverlay.classList.add('active');
    }
}

// Hide loading animation
function hideLoading() {
    const loadingOverlay = document.querySelector('.fire-loading');
    if (loadingOverlay) {
        loadingOverlay.classList.remove('active');
    }
}

// Initialize tooltips
function initTooltips() {
    const cards = document.querySelectorAll('.parameter-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', (e) => {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip-custom';
            tooltip.textContent = card.dataset.tooltip;
            
            const rect = card.getBoundingClientRect();
            tooltip.style.left = rect.right + 10 + 'px';
            tooltip.style.top = rect.top + (rect.height / 2) - 10 + 'px';
            
            document.body.appendChild(tooltip);
        });

        card.addEventListener('mouseleave', () => {
            const tooltip = document.querySelector('.tooltip-custom');
            if (tooltip) {
                tooltip.remove();
            }
        });
    });
}

// Show error message
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message fade-in';
    errorDiv.textContent = message;
    document.body.appendChild(errorDiv);
    
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

// Initialize tooltips when document is ready
document.addEventListener('DOMContentLoaded', initTooltips);