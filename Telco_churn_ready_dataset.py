# Detailed Sample Code for Data Collection, Cleaning, and Preprocessing

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# 1. Data Collection
# Load dataset 
data = pd.read_csv('C:\\Users\\DELL\\Downloads\\archive (2)\\Telco_customer_churn.csv')

# Overview of the dataset
print("Overview of the dataset:")
print(data.describe())  # Summary statistics
print(data.info())      # DataFrame info including column types and non-null counts

# Inspect the data after the overview
print("Inspecting the first few rows of the dataset:")
print(data.head())      # Display the first few rows of the DataFrame

# 2. Data Integration
# Combine multiple datasets
data_2 = pd.read_csv('C:\\Users\\DELL\\Downloads\\archive (2)\\Telco_customer_churn_demographics.csv')

# Print the columns of both DataFrames before merging
print("Columns in data DataFrame:", data.columns)
print("Columns in data_2 DataFrame:", data_2.columns)

# Check if 'CustomerID' exists in both DataFrames before merging
if 'CustomerID' in data.columns and 'CustomerID' in data_2.columns:
    final_data = pd.merge(data, data_2, on='CustomerID', how='inner')
else:
    print("Column 'CustomerID' does not exist in one of the DataFrames.")

# 3. Data Cleaning
# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Handle missing values (using mean for numerical and most frequent for categorical)
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

imputer_num = SimpleImputer(strategy='mean')
imputer_cat = SimpleImputer(strategy='most_frequent')

data[numerical_cols] = imputer_num.fit_transform(data[numerical_cols])
data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])

# Check for duplicates
data = data.drop_duplicates()

# Remove outliers using IQR method
for col in numerical_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    data = data[(data[col] >= Q1 - 1.5 * IQR) & (data[col] <= Q3 + 1.5 * IQR)]

# 4. Data Preprocessing
# Encode categorical variables
encoder = OneHotEncoder(sparse_output=False, drop='first')
categorical_encoded = pd.DataFrame(encoder.fit_transform(data[categorical_cols]),
                                    columns=encoder.get_feature_names_out(categorical_cols))

# Merge encoded columns back into the dataset
data = pd.concat([data[numerical_cols], categorical_encoded], axis=1)

# Normalize numerical variables
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# 5. Feature Engineering
# Create new features (example: tenure in years)
data['tenure_years'] = data['Tenure Months'] / 12

# Drop original column if transformed
data = data.drop(columns=['Tenure Months'])

# 6. Data Quality Assessment
# Check for data integrity issues
print("Final Dataset Info:\n")
print(final_data.info())

# Visualization to identify potential bias
sns.histplot(final_data['Age'], bins=10, kde=True)
plt.title('Age Distribution')
plt.show()

# Save cleaned and processed data
final_data.to_csv('cleaned_customer_data.csv', index=False)

# Exploratory Data Analysis (EDA)
# Print the columns and their data types in final_data for debugging
print("Data types in final_data DataFrame:\n", final_data.dtypes)

# Additional EDA visualizations
# Plotting the distribution of tenure in years
sns.histplot(final_data['Tenure Months'], bins=10, kde=True)
plt.title('Tenure in Months Distribution')
plt.show()

# Additional visualizations
data['churn'] = np.random.choice([0, 1], size=len(data))  # Example column for churn
sns.countplot(x='Churn Value', data=final_data)
plt.title('Churn Distribution')
plt.show()

print("Data DataFrame head:\n", data.head())
print("Data_2 DataFrame head:\n", data_2.head())
