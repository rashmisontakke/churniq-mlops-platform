import pandas as pd
import pickle

# Load model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data
df = pd.read_csv('data/Customer-Churn-Records.csv')
# Define clean features (available before churn)
clean_features = [
    'Geography', 'Gender', 'Age',
    'CreditScore', 'Tenure', 'Balance', 'EstimatedSalary',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember'
]

df = df.dropna()
X = df[clean_features]

df_encoded = pd.get_dummies(X)

# Align columns with training (handle missing columns)
with open('models/model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)
for col in model_columns:
    if col not in df_encoded:
        df_encoded[col] = 0

df_encoded = df_encoded[model_columns]

# Predict
preds = model.predict(df_encoded)

# Output
output = df.copy()
output['prediction'] = preds
output.to_csv('models/predictions.csv', index=False)
print('Predictions saved to predictions.csv') 