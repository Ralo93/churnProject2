import pandas as pd
import joblib
import json
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# Function to prepare the data (same as in the training script)
def prepare_data(input_data):
    columns_to_drop = [
        'Churn Category', 'Churn Reason', 'Internet Type', 'Avg Monthly GB Download',
        'Online Security', 'Online Backup', 'Device Protection Plan',
        'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music',
        'Unlimited Data', 'Avg Monthly Long Distance Charges', 'Multiple Lines'
    ]
    input_data = input_data.drop(columns=columns_to_drop, errors='ignore')
    imputer = SimpleImputer(strategy='mean')
    numerical_columns = ['Total Charges', 'Total Refunds', 'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue']
    input_data[numerical_columns] = imputer.fit_transform(input_data[numerical_columns])
    label_encoder = LabelEncoder()
    categorical_columns = ['Gender', 'Married', 'Payment Method']
    for column in categorical_columns:
        input_data[column] = label_encoder.fit_transform(input_data[column])
    categorical_columns = ['Offer', 'Phone Service', 'Internet Service', 'Contract', 'Paperless Billing']
    input_data = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)
    input_data = input_data.drop(columns=['Customer ID', 'City', 'Zip Code', 'Latitude', 'Longitude'], errors='ignore')
    return input_data

# Function to align input data with the feature names used during training
def align_features(input_data, feature_names):
    # Prepare the input data
    input_data = prepare_data(input_data)
    # Align the columns
    aligned_data = pd.DataFrame(columns=feature_names)
    for column in feature_names:
        if column in input_data.columns:
            aligned_data[column] = input_data[column]
        else:
            aligned_data[column] = 0
    return aligned_data

# Load the saved model
model_filename = 'random_forest_churn_model.joblib'
loaded_model = joblib.load(model_filename)

# Load the feature names
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

# Load new data for inference
new_data_file = 'new_customer_data.csv'
new_data = pd.read_csv(new_data_file)

# Align the input data with the saved feature names
aligned_data = align_features(new_data, feature_names)

# Make predictions
predictions = loaded_model.predict(aligned_data)

# Print predictions
print(predictions)

# Make predictions
probabilities = loaded_model.predict_proba(aligned_data)

# Print class labels and probabilities
for prob in probabilities:
    print({class_label: p for class_label, p in zip(loaded_model.classes_, prob)})


# Optionally, you can add the predictions back to the original data for better interpretation
new_data['Predictions'] = predictions
print(new_data.head())
