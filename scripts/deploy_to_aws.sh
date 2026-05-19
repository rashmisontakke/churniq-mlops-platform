#!/bin/bash

# AWS ECR Deployment Script for Churn Model API
# Usage: ./deploy_to_aws.sh [AWS_ACCOUNT_ID] [AWS_REGION]

set -e

# Configuration
AWS_ACCOUNT_ID=${1:-$(aws sts get-caller-identity --query Account --output text)}
AWS_REGION=${2:-us-east-1}
REPOSITORY_NAME="churn-model-api"
IMAGE_TAG="latest"

echo "🚀 Starting deployment to AWS ECR..."
echo "Account ID: $AWS_ACCOUNT_ID"
echo "Region: $AWS_REGION"
echo "Repository: $REPOSITORY_NAME"

# Build Docker image
echo "📦 Building Docker image..."
docker build -t $REPOSITORY_NAME:$IMAGE_TAG .

# Create ECR repository if it doesn't exist
echo "🏗️  Creating/checking ECR repository..."
aws ecr describe-repositories --repository-names $REPOSITORY_NAME --region $AWS_REGION 2>/dev/null || \
aws ecr create-repository --repository-name $REPOSITORY_NAME --region $AWS_REGION

# Get ECR login
echo "🔐 Authenticating with ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Tag image for ECR
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:$IMAGE_TAG"
echo "🏷️  Tagging image: $ECR_URI"
docker tag $REPOSITORY_NAME:$IMAGE_TAG $ECR_URI

# Push to ECR
echo "⬆️  Pushing to ECR..."
docker push $ECR_URI

echo "✅ Successfully deployed to ECR!"
echo "Image URI: $ECR_URI"
echo ""
echo "Next steps:"
echo "1. Deploy to ECS: aws ecs create-service..."
echo "2. Deploy to EKS: kubectl apply -f k8s-deployment.yaml"
echo "3. Deploy to Lambda: aws lambda create-function..."
echo "4. Deploy to App Runner: aws apprunner create-service..." 