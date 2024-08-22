# Quickstart
### **Step 1: Installation**
First, ensure you have all the necessary libraries installed in your Python environment. You can install them using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lifelines imbalanced-learn
```
### **Step 2: Prepare Your Data**
Load your time-series medical data into a pandas DataFrame. Your dataset should include columns for patient IDs, timestamps, clinical measurements, and outcome variables.
```python
import pandas as pd

# Load your patient data into a DataFrame
file_path = 'path/to/your/375_patients_example.xlsx'  # Replace with the actual file path
df = pd.read_excel(file_path)

# Ensure your data is sorted by patient ID and record time
df.sort_values(by=['PATIENT_ID', 'RE_DATE'], inplace=True)
```
### **Step 3: Initialize the Toolkit**
Instantiate the `FeModEvaluator` class with your data, specifying the key columns for analysis:
```python
import mtslearn.feature_extraction as fe

# Initialize the feature extraction and evaluation tool
fe = fe.FeModEvaluator(
    df=df,
    group_col='PATIENT_ID',
    time_col='RE_DATE',
    outcome_col='outcome',
    value_cols=['eGFR', 'creatinine'],
    selected_features=['mean', 'max'],
    include_duration=True
)
```
### **Step 4: Run a Full Analysis Pipeline**
Use the `run` method to automatically extract features, prepare the data, and evaluate a predictive model. In this example, we’ll use the XGBoost model:
```python
# Run the pipeline with XGBoost, filling missing values with the mean
fe.run(
    model_type='xgboost',
    fill=True,
    fill_method='mean',
    test_size=0.3,
    balance_data=True,
    cross_val=False
)
```
### **Step 5: Visualize the Results**
After running the analysis, you can generate visualizations to help doctors interpret the results. For instance, you can create a boxplot or a correlation matrix:
```python
# Visualize data with a boxplot for a specific clinical measurement
fe.describe_data(plot_type='boxplot', value_col='eGFR')

# Generate a correlation matrix between two clinical measurements
fe.describe_data(plot_type='correlation_matrix', feature1='eGFR', feature2='creatinine')
```
### **Step 6: Interpret the Results**
Review the output, including model performance metrics, feature importance plots (if applicable), and visualizations. These results can guide doctors in making informed clinical decisions, such as adjusting treatment plans or identifying patients at higher risk.

---

### **Summary**
In just a few steps, you can use the Medical Time-Series Data Analysis Toolkit to analyze patient data and generate insightful visualizations. This quickstart guide is designed to get you up and running quickly, enabling you to provide valuable data-driven insights in a clinical setting. 
For more detailed customizations and advanced usage, refer to the full user guide.
