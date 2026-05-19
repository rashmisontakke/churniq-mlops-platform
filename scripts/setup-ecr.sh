#!/bin/bash

# AWS ECR Setup Script for Lambda Deployment
# This script creates the necessary ECR repository for the Lambda deployment

set -e

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
ECR_REPOSITORY=${ECR_REPOSITORY:-churn-model-lambda}

echo "🚀 Setting up ECR repository for Lambda deployment..."
echo "📍 Region: $AWS_REGION"
echo "📦 Repository: $ECR_REPOSITORY"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if user is authenticated
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

echo "✅ AWS CLI is configured and authenticated"

# Create ECR repository if it doesn't exist
echo "🔍 Checking if ECR repository exists..."
if aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $AWS_REGION &> /dev/null; then
    echo "✅ ECR repository '$ECR_REPOSITORY' already exists"
else
    echo "🆕 Creating ECR repository '$ECR_REPOSITORY'..."
    aws ecr create-repository \
        --repository-name $ECR_REPOSITORY \
        --region $AWS_REGION \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256
    
    echo "✅ ECR repository created successfully"
fi

# Get repository URI
REPOSITORY_URI=$(aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $AWS_REGION --query 'repositories[0].repositoryUri' --output text)
echo "📋 Repository URI: $REPOSITORY_URI"

# Set lifecycle policy to manage image retention
echo "🔄 Setting up lifecycle policy..."
cat > /tmp/lifecycle-policy.json << EOF
{
    "rules": [
        {
            "rulePriority": 1,
            "description": "Keep last 10 images",
            "selection": {
                "tagStatus": "any",
                "countType": "imageCountMoreThan",
                "countNumber": 10
            },
            "action": {
                "type": "expire"
            }
        }
    ]
}
EOF

aws ecr put-lifecycle-policy \
    --repository-name $ECR_REPOSITORY \
    --region $AWS_REGION \
    --lifecycle-policy-text file:///tmp/lifecycle-policy.json

echo "✅ Lifecycle policy set successfully"

# Clean up
rm -f /tmp/lifecycle-policy.json

echo ""
echo "🎉 ECR setup completed successfully!"
echo ""
echo "📝 Next steps:"
echo "1. Set up GitHub secrets in your repository:"
echo "   - AWS_ACCESS_KEY_ID"
echo "   - AWS_SECRET_ACCESS_KEY"
echo ""
echo "2. Push to main branch to trigger the deployment workflow"
echo ""
echo "3. The workflow will:"
echo "   - Build Docker image"
echo "   - Push to ECR: $REPOSITORY_URI"
echo "   - Deploy to Lambda function: churn-model-predictor"
echo "" 