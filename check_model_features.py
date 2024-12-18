import joblib
import pandas as pd

# Load the trained model
model_path = 'models/temp.joblib'
model = joblib.load(model_path)

# If the model is a sklearn model, we can try to get feature names
if hasattr(model, 'feature_names_in_'):
    print("\nFeature names from model:")
    for i, feature in enumerate(model.feature_names_in_, 1):
        print(f"{i}. {feature}")
    print(f"\nTotal number of features: {len(model.feature_names_in_)}")
else:
    print("\nModel input features:")
    print(f"Number of features expected by model: {model.n_features_in_}")

# Load the training dataset to verify features
try:
    # Try different possible dataset files
    dataset_files = [
        'Data/Temp/Fire_dataset_cleaned.csv',
        'Data/Temp/Algerian_forest_fires_dataset_CLEANED.csv',
        'Data/Temp/dataset.csv'
    ]
    
    for file in dataset_files:
        try:
            df = pd.read_csv(file)
            print(f"\nFeatures found in {file}:")
            features = df.columns.tolist()
            for i, feature in enumerate(features, 1):
                print(f"{i}. {feature}")
            print(f"\nTotal number of features in dataset: {len(features)}")
            break
        except FileNotFoundError:
            continue
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
