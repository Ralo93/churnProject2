import pandas as pd
import joblib

from data_preparation_pipeline import prepare_data

# Load the trained model
model_path = 'random_forest_churn_model.joblib'
loaded_model = joblib.load(model_path)

# Assume you have a DataFrame `prepared_data` for prediction
prepared_data = pd.DataFrame([{
'Customer ID': '0012-IGKFF',
    'Gender': 'Male',
    'Age': 78,
    'Married': 'Yes',
    'Number of Dependents': 0,
    'City': 'Martinez',
    'Zip Code': 94553,
    'Latitude': 38.014457,
    'Longitude': -122.115432,
    'Number of Referrals': 1,
    'Tenure in Months': 7,
    'Offer': 'None',
    'Phone Service': 'Yes',
    'Internet Service': 'Yes',
    'Contract': 'Month-to-Month',
    'Paperless Billing': 'Yes',
    'Payment Method': 'Bank Withdrawal',
    'Monthly Charge': 79.30,
    'Total Charges': 523.15,
    'Total Refunds': 0.00,
    'Total Extra Data Charges': 10.00,
    'Total Long Distance Charges': 134.47,
    'Total Revenue': 667.62
}])

# Check the feature names used during training
trained_feature_names = loaded_model.feature_names_in_

# Ensure the input data has the same feature names
missing_features = set(trained_feature_names) - set(prepared_data.columns)
extra_features = set(prepared_data.columns) - set(trained_feature_names)
extra_features = set(prepare_data(prepared_data).columns) - set(trained_feature_names)

if missing_features:
    print(f"Missing features: {missing_features}")

if extra_features:
    print(f"Extra features: {extra_features}")


# Align the columns of prepared_data with the trained model
prepared_data = prepared_data[trained_feature_names]

# Now make predictions
prediction = loaded_model.predict(prepared_data)
print(prediction)
