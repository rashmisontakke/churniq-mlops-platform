import pandas as pd

# Load predictions
preds = pd.read_csv('models/predictions.csv')

# Calculate churn rate in predictions
churn_rate = preds['prediction'].mean()
print(f'Predicted churn rate: {churn_rate:.2%}')

# If ground truth is available, print accuracy
if 'Churn' in preds.columns:
    accuracy = (preds['Churn'] == preds['prediction']).mean()
    print(f'Prediction accuracy: {accuracy:.2%}') 