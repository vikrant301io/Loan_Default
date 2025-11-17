#!/bin/bash
# Deployment script for AWS ECS

set -e

REGION="us-east-1"
ECR_REPOSITORY="loan-default"
CLUSTER_NAME="loan-default-cluster"
SERVICE_NAME="loan-default-service"
TASK_DEFINITION="loan-default-task"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Login to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Build and push Docker image
docker build -f docker/Dockerfile.api -t $ECR_REPOSITORY:latest .
docker tag $ECR_REPOSITORY:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPOSITORY:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPOSITORY:latest

# Update ECS service
aws ecs update-service \
  --cluster $CLUSTER_NAME \
  --service $SERVICE_NAME \
  --force-new-deployment \
  --region $REGION

echo "Deployment completed successfully!"

