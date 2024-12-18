"""
Forest Fire Prediction - Vegetation Features Analysis

This script shows the features used in training the vegetation model for forest fire prediction.

Features used:
1. NDVI (Normalized Difference Vegetation Index)
   - Measures vegetation density and health
   - Range: -1 to 1 (higher values indicate denser vegetation)

2. LST (Land Surface Temperature)
   - Surface temperature of the land
   - Measured in Kelvin

3. BURNED_AREA
   - Area affected by fire
   - Measured in square units

Target Variable:
- CLASS: Binary classification (fire, no_fire)

Model Performance:
- Accuracy: 84%
- Precision for fire class: 68%
- Recall for fire class: 38%
- F1-score for fire class: 49%

Feature Importance:
1. LST: 39.15%
2. NDVI: 33.71%
3. BURNED_AREA: 27.14%

Dataset Distribution:
- Total samples: 1,713
- No fire cases: 1,327
- Fire cases: 386
"""

# Print the information
if __name__ == "__main__":
    with open(__file__, 'r') as file:
        print(file.read())
