#!/usr/bin/env python3
"""
Local Lambda Function Tester
Test the Lambda handler locally before deployment
"""

import json
import sys
import os

# Add project root directory to path to import lambda_handler
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.lambda_handler import handler
except ImportError as e:
    print(f"❌ Error importing lambda_handler: {e}")
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Python path: {sys.path[:3]}...")  # Show first 3 entries
    print("💡 Make sure you're running this script from the project root directory")
    sys.exit(1)

def test_lambda_function():
    """Test the Lambda function with various scenarios"""
    
    print("🧪 Testing Lambda Handler Locally")
    print("=" * 50)
    
    # Test case 1: Valid input
    print("\n📝 Test 1: Valid customer data")
    test_event_1 = {
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
    
    result_1 = handler(test_event_1, None)
    print(f"Status Code: {result_1['statusCode']}")
    print(f"Response: {json.dumps(json.loads(result_1['body']), indent=2)}")
    
    # Test case 2: API Gateway format
    print("\n📝 Test 2: API Gateway event format")
    test_event_2 = {
        "body": json.dumps({
            "Geography": "Germany",
            "Gender": "Male",
            "Age": 35,
            "CreditScore": 750,
            "Tenure": 5,
            "Balance": 100000.0,
            "EstimatedSalary": 75000,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1
        })
    }
    
    result_2 = handler(test_event_2, None)
    print(f"Status Code: {result_2['statusCode']}")
    print(f"Response: {json.dumps(json.loads(result_2['body']), indent=2)}")
    
    # Test case 3: Missing fields
    print("\n📝 Test 3: Missing required fields")
    test_event_3 = {
        "Geography": "Spain",
        "Gender": "Female",
        "Age": 30
        # Missing other required fields
    }
    
    result_3 = handler(test_event_3, None)
    print(f"Status Code: {result_3['statusCode']}")
    print(f"Response: {json.dumps(json.loads(result_3['body']), indent=2)}")
    
    # Test case 4: High churn probability
    print("\n📝 Test 4: High churn probability customer")
    test_event_4 = {
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 55,
        "CreditScore": 400,
        "Tenure": 1,
        "Balance": 0.0,
        "EstimatedSalary": 30000,
        "NumOfProducts": 4,
        "HasCrCard": 0,
        "IsActiveMember": 0
    }
    
    result_4 = handler(test_event_4, None)
    print(f"Status Code: {result_4['statusCode']}")
    print(f"Response: {json.dumps(json.loads(result_4['body']), indent=2)}")
    
    print("\n✅ Lambda testing completed!")
    print("\n💡 Tips:")
    print("- All tests should return status 200 except the missing fields test (400)")
    print("- Check that predictions and probabilities are reasonable")
    print("- Verify that the model is loading correctly")

def check_model_files():
    """Check if model files exist"""
    model_files = [
        'models/model.pkl',
        'models/model_columns.pkl'
    ]
    
    print("\n🔍 Checking model files...")
    for file in model_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing - run 'python src/train.py' first")
            return False
    return True

if __name__ == "__main__":
    # Check if model files exist
    if not check_model_files():
        print("\n🚨 Model files missing. Training model first...")
        os.system("python src/train.py")
    
    # Run tests
    test_lambda_function() 