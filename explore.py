import pandas as pd

# Load the data
data = pd.read_csv('telecom_customer_churn.csv')

# Explore the data
print(data.head())
print(data.info())
print(data.describe())