import json
import pandas as pd
import pickle
import os
from typing import Dict, Any
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global variables for model caching
model = None
model_columns = None

def load_model():
    """Load model and model columns if not already loaded"""
    global model, model_columns
    
    if model is None or model_columns is None:
        try:
            # Check if running in Lambda environment
            if os.path.exists('/var/task'):
                # Lambda environment
                model_path = os.getenv('MODEL_PATH', '/var/task/models/model.pkl')
                columns_path = os.getenv('MODEL_COLUMNS_PATH', '/var/task/models/model_columns.pkl')
            else:
                # Local environment
                model_path = os.getenv('MODEL_PATH', 'models/model.pkl')
                columns_path = os.getenv('MODEL_COLUMNS_PATH', 'models/model_columns.pkl')
            
            logger.info(f"Loading model from {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"Loading model columns from {columns_path}")
            with open(columns_path, 'rb') as f:
                model_columns = pickle.load(f)
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for churn prediction
    
    Args:
        event: Lambda event object containing the input data
        context: Lambda context object
        
    Returns:
        Dictionary with prediction result
    """
    try:
        # Load model if not already loaded
        load_model()
        
        # Parse the input data
        if 'body' in event:
            # Handle API Gateway event
            if isinstance(event['body'], str):
                input_data = json.loads(event['body'])
            else:
                input_data = event['body']
        else:
            # Handle direct Lambda invocation
            input_data = event
        
        logger.info(f"Processing input: {input_data}")
        
        # Validate required fields
        required_fields = [
            'Geography', 'Gender', 'Age', 'CreditScore', 
            'Tenure', 'Balance', 'EstimatedSalary', 
            'NumOfProducts', 'HasCrCard', 'IsActiveMember'
        ]
        
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': f'Missing required fields: {missing_fields}'
                })
            }
        
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # Apply same preprocessing as training
        df_encoded = pd.get_dummies(df)
        
        # Ensure all model columns are present
        for col in model_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Reorder columns to match training data
        df_encoded = df_encoded[model_columns]
        
        # Make prediction
        prediction = model.predict(df_encoded)[0]
        prediction_proba = model.predict_proba(df_encoded)[0]
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Will Churn' if prediction == 1 else 'Will Stay',
            'confidence': {
                'stay_probability': float(prediction_proba[0]),
                'churn_probability': float(prediction_proba[1])
            },
            'input_data': input_data,
            'model_version': '1.0'
        }
        
        logger.info(f"Prediction result: {result}")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result)
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f'Internal server error: {str(e)}'
            })
        }

# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "CreditScore": 600,
        "Tenure": 3,
        "Balance": 0.0,
        "EstimatedSalary": 50000,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1
    }
    
    result = handler(test_event, None)
    print(json.dumps(result, indent=2)) 