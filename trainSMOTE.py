import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from scipy.stats import randint
import joblib
import json

# Function to prepare the data
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
    categorical_columns = ['Gender', 'Married', 'Payment Method', 'Customer Status']
    for column in categorical_columns:
        input_data[column] = label_encoder.fit_transform(input_data[column])
    categorical_columns = ['Offer', 'Phone Service', 'Internet Service', 'Contract', 'Paperless Billing']
    input_data = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)
    input_data = input_data.drop(columns=['Customer ID', 'City', 'Zip Code', 'Latitude', 'Longitude'], errors='ignore')
    return input_data

# Load the dataset
file_path = 'telecom_customer_churn.csv'
data = pd.read_csv(file_path)

# Prepare the data
data_cleaned = prepare_data(data)

# Split the data
X = data_cleaned.drop(columns=['Customer Status'])
y = data_cleaned['Customer Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the classes in the training set
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 300),
    'max_features': [None, 'sqrt', 'log2'],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4],
    'bootstrap': [True, False]
}

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced')

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist,
                                   n_iter=100, cv=5, n_jobs=-1, random_state=42, verbose=2)

# Fit the random search to the data
random_search.fit(X_train_balanced, y_train_balanced)

# Get the best parameters
best_params = random_search.best_params_

# Train the Random Forest classifier with the best parameters
best_rf_classifier = RandomForestClassifier(**best_params, random_state=42, class_weight='balanced')
best_rf_classifier.fit(X_train_balanced, y_train_balanced)

# Save the model using joblib
model_filename = 'random_forest_churn_model.joblib'
joblib.dump(best_rf_classifier, model_filename)

# Save the feature names
feature_names = X_train.columns.tolist()
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)

# Make predictions on the test set
y_pred = best_rf_classifier.predict(X_test)
y_pred_prob = best_rf_classifier.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (Churned)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Calculate the lift for "Churned" outcome
df_lift = pd.DataFrame({'true_labels': y_test, 'predicted_prob': y_pred_prob})
df_lift = df_lift.sort_values(by='predicted_prob', ascending=False)

# Lift for top 25%
top_25_percent = df_lift.head(int(len(df_lift) * 0.25))
churn_rate_top_25 = top_25_percent['true_labels'].mean()  # Proportion of "Churned" in top 25%
churn_rate_overall = df_lift['true_labels'].mean()  # Overall proportion of "Churned"
lift_25 = churn_rate_top_25 / churn_rate_overall

# Lift for top 10%
top_10_percent = df_lift.head(int(len(df_lift) * 0.10))
churn_rate_top_10 = top_10_percent['true_labels'].mean()  # Proportion of "Churned" in top 10%
lift_10 = churn_rate_top_10 / churn_rate_overall

# Lift for top 5%
top_5_percent = df_lift.head(int(len(df_lift) * 0.05))
churn_rate_top_5 = top_5_percent['true_labels'].mean()  # Proportion of "Churned" in top 5%
lift_5 = churn_rate_top_5 / churn_rate_overall

print("Test Set Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Lift (top 25% targeted):", lift_25)
print("Lift (top 10% targeted):", lift_10)
print("Lift (top 5% targeted):", lift_5)
