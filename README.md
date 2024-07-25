# mtslearn

# FeatureExtractorAndModelEvaluator

The `FeatureExtractorAndModelEvaluator` class is a comprehensive tool for extracting features from time-series data, preparing data for machine learning models, and evaluating various models. It simplifies the process of handling complex datasets, particularly in medical or clinical research.

## Features

- **Feature Extraction**: Extracts basic statistical features from time-series data.
- **Data Preparation**: Handles missing values, balances data using SMOTE, and splits data into training and testing sets.
- **Model Evaluation**: Evaluates logistic regression, Cox proportional hazards, XGBoost, and Lasso regression models.
- **Cross-Validation**: Supports cross-validation for robust model evaluation.

## Installation

To use this class, ensure you have the following dependencies installed:

```bash
pip install pandas scikit-learn matplotlib seaborn lifelines xgboost imbalanced-learn
```

## Usage

### Initialization

Initialize the `FeatureExtractorAndModelEvaluator` with your dataset and column information:

```python
from feature_extractor_and_model_evaluator import FeatureExtractorAndModelEvaluator

fe = FeatureExtractorAndModelEvaluator(
    df=dataframe,
    group_col='patient_id',
    time_col='date',
    outcome_col='outcome',
    value_cols=['feature1', 'feature2']
)
```

### Running the Pipeline

Run the entire pipeline with a single method:

```python
fe.run(model_type='logit', fill=True, fill_method='mean', test_size=0.2, balance_data=True, cross_val=False)
```

## Example

```python
import pandas as pd

# Load your dataset
df = pd.read_csv('data.csv')

# Initialize the feature extractor and model evaluator
fe = FeatureExtractorAndModelEvaluator(
    df=df,
    group_col='patient_id',
    time_col='date',
    outcome_col='outcome',
    value_cols=['feature1', 'feature2']
)

# Run the logistic regression model
fe.run(model_type='logit', fill=True, fill_method='mean', test_size=0.2, balance_data=True, cross_val=False)
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## Acknowledgements

This class was developed to simplify the process of feature extraction and model evaluation for time-series data, particularly in the context of medical research.
```
