// Initialize the map
const map = L.map('map').setView([20.5937, 78.9629], 5);  // Centered on India

// Add dark theme map tiles
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 19
}).addTo(map);

// Store fire markers and layers
let fireMarkers = {
    landsat: [],
    viirs: [],
    modis: []
};
let lastUpdate = new Date();

// Custom fire icon based on confidence and satellite
function getFireIcon(confidence, satellite) {
    let size, color;
    
    // Base size on satellite resolution
    switch(satellite) {
        case 'landsat':
            size = confidence === 'high' ? 14 : confidence === 'nominal' ? 12 : 10;
            color = '#ff4444';
            break;
        case 'viirs':
            size = confidence === 'high' ? 12 : confidence === 'nominal' ? 10 : 8;
            color = '#ffaa33';
            break;
        case 'modis':
            size = confidence === 'high' ? 10 : confidence === 'nominal' ? 8 : 6;
            color = '#ffdd33';
            break;
        default:
            size = 10;
            color = '#ff4444';
    }
    
    if (confidence === 'low') {
        color = satellite === 'landsat' ? '#ff8888' : 
                satellite === 'viirs' ? '#ffcc88' : 
                '#ffee88';
    }
    
    return L.divIcon({
        className: 'fire-marker',
        html: `<div style="
            width: ${size}px;
            height: ${size}px;
            background-color: ${color};
            border-radius: 50%;
            box-shadow: 0 0 ${size}px ${color};
            animation: pulse 1.5s infinite;
        "></div>`,
        iconSize: [size, size]
    });
}

// Fetch fire data from NASA FIRMS
async function fetchFireData() {
    try {
        const response = await fetch('/get_fire_data');
        const data = await response.json();
        
        // Clear existing markers
        Object.values(fireMarkers).forEach(markers => {
            markers.forEach(marker => marker.remove());
        });
        fireMarkers = { landsat: [], viirs: [], modis: [] };
        
        // Update statistics
        updateStats(data);
        
        // Add new markers based on satellite type
        data.forEach(fire => {
            const satellite = fire.satellite.toLowerCase();
            const marker = L.marker([fire.latitude, fire.longitude], {
                icon: getFireIcon(fire.confidence.toLowerCase(), satellite)
            });
            
            marker.confidence = fire.confidence.toLowerCase();
            marker.satellite = satellite;
            
            marker.bindPopup(`
                <div class="fire-popup">
                    <h3>Fire Detection</h3>
                    <p><strong>Satellite:</strong> ${fire.satellite}</p>
                    <p><strong>Confidence:</strong> ${fire.confidence}</p>
                    <p><strong>Detected:</strong> ${new Date(fire.acq_date).toLocaleString()}</p>
                    <p><strong>Brightness:</strong> ${fire.brightness.toFixed(2)}K</p>
                    <p><strong>FRP:</strong> ${fire.frp.toFixed(2)} MW</p>
                </div>
            `);
            
            if (shouldShowMarker(marker)) {
                marker.addTo(map);
            }
            
            fireMarkers[satellite].push(marker);
        });
        
        // Update last update time
        lastUpdate = new Date();
        document.getElementById('updateTime').textContent = lastUpdate.toLocaleString();
        
    } catch (error) {
        console.error('Error fetching fire data:', error);
    }
}

// Helper function to determine if a marker should be shown
function shouldShowMarker(marker) {
    const confidenceChecked = document.getElementById(marker.confidence + 'Confidence').checked;
    const satelliteChecked = document.getElementById(marker.satellite + 'Layer').checked;
    return confidenceChecked && satelliteChecked;
}

// Update statistics
function updateStats(data) {
    const totalFires = data.length;
    const highConfFires = data.filter(fire => fire.confidence.toLowerCase() === 'high').length;
    const last24h = data.filter(fire => {
        const fireTime = new Date(fire.acq_date).getTime();
        const dayAgo = Date.now() - (24 * 60 * 60 * 1000);
        return fireTime > dayAgo;
    }).length;
    
    document.getElementById('totalFires').textContent = totalFires;
    document.getElementById('highConfFires').textContent = highConfFires;
    document.getElementById('last24h').textContent = last24h;
}

// Filter handlers for confidence levels
['high', 'nominal', 'low'].forEach(confidence => {
    document.getElementById(confidence + 'Confidence').addEventListener('change', function() {
        Object.values(fireMarkers).flat().forEach(marker => {
            if (marker.confidence === confidence) {
                if (shouldShowMarker(marker)) marker.addTo(map);
                else marker.remove();
            }
        });
    });
});

// Filter handlers for satellite layers
['landsat', 'viirs', 'modis'].forEach(satellite => {
    document.getElementById(satellite + 'Layer').addEventListener('change', function() {
        fireMarkers[satellite].forEach(marker => {
            if (shouldShowMarker(marker)) marker.addTo(map);
            else marker.remove();
        });
    });
});

// Fetch data initially and update frequently
fetchFireData();
setInterval(fetchFireData, 10 * 1000); // Update every 10 seconds
