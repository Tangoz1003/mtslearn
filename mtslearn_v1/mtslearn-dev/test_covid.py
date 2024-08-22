import pandas as pd  # Import pandas library for data manipulation
import os  # Import os library for interacting with the operating system
import mtslearn.feature_extraction as fe  # Import feature_extraction module from mtslearn package and alias it as 'fe'

# Load the Excel file into a pandas DataFrame
file_path = '/hot_data/tangoz/code/mtslearn/mtslearn-dev/375_patients_example.xlsx'  # Replace with your actual file path
df = pd.read_excel(file_path)  # Read the Excel file located at file_path

# Fill missing values in the PATIENT_ID column using the forward fill method
df['PATIENT_ID'].fillna(method='ffill', inplace=True)

# Optionally print the first 50 rows of the DataFrame to check the data
# print(df.head(50))

# Ensure the data is sorted by PATIENT_ID and RE_DATE to maintain chronological order
df.sort_values(by=['PATIENT_ID', 'RE_DATE'], inplace=True)

# Save the sorted DataFrame to a CSV file
df.to_csv('/hot_data/tangoz/code/mtslearn/mtslearn-dev/375prep.csv')

# Initialize FeModEvaluator class for feature extraction and model evaluation
fe = fe.FeModEvaluator(df, 'PATIENT_ID', 'RE_DATE', 'outcome', ['eGFR', 'creatinine'], ['mean', 'max'], include_duration=True)

# Run the feature extraction and model evaluation using XGBoost as the model
fe.run(model_type='xgboost', fill=True, fill_method='mean', test_size=0.3, balance_data=True, plot_importance=True)

# Example calls to the describe_data method to visualize the data
fe.describe_data(plot_type='boxplot', value_col='eGFR')  # Generate a boxplot for the eGFR column
fe.describe_data(plot_type='correlation_matrix', feature1='eGFR', feature2='creatinine')  # Generate a correlation matrix between eGFR and creatinine
