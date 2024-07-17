import joblib
import sklearn
import json

# Load the version information
with open('version_info.json', 'r') as f:
    saved_version_info = json.load(f)

# Check the current versions
current_version_info = {
    'scikit-learn': sklearn.__version__,
    'joblib': joblib.__version__
}

# Print version info for debugging
print("Saved version info:", saved_version_info)
print("Current version info:", current_version_info)

# Compare the versions
if saved_version_info != current_version_info:
    print("Warning: Version mismatch detected. There may be compatibility issues.")
else:
    print("Version check passed. No compatibility issues detected.")

# Load the model
model_filename = 'random_forest_churn_model.joblib'
loaded_model = joblib.load(model_filename)




exit()

# Save the model
model_filename = 'random_forest_churn_model.joblib'
joblib.dump('test', model_filename)

# Save the version information
version_info = {
    'scikit-learn': sklearn.__version__,
    'joblib': joblib.__version__
}
with open('version_info.json', 'w') as f:
    json.dump(version_info, f)
