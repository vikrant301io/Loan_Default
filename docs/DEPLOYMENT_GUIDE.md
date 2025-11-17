# Deployment Guide

## Prerequisites

1. AWS Account with appropriate permissions
2. Docker installed locally
3. Python 3.9+
4. DVC installed
5. MLflow installed

## Local Development Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd Loan_Default
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Initialize DVC
```bash
dvc init
dvc remote add -d storage s3://your-bucket/dvc-storage
```

### 5. Setup MLflow
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

## Training Pipeline

### 1. Prepare Data
Place your training data in `data/raw/loan_data.csv`

### 2. Run Training
```bash
python scripts/train.py --config configs/config.yaml
```

### 3. View Results
- MLflow UI: http://localhost:5000
- Model saved to: `models/best_model.pkl`
- Preprocessor saved to: `models/preprocessor.pkl`

## Local API Testing

### 1. Build Docker Image
```bash
docker build -f docker/Dockerfile.api -t loan-default-api .
```

### 2. Run Container
```bash
docker run -p 8080:8080 loan-default-api
```

### 3. Test API
```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"Client_Income": 50000, "Credit_Amount": 100000, ...}'
```

## AWS Deployment

### 1. Setup AWS Resources

#### Create ECR Repository
```bash
aws ecr create-repository --repository-name loan-default --region us-east-1
```

#### Create ECS Cluster
```bash
aws ecs create-cluster --cluster-name loan-default-cluster --region us-east-1
```

#### Create S3 Bucket for DVC
```bash
aws s3 mb s3://your-bucket-name --region us-east-1
```

### 2. Configure AWS Credentials

#### GitHub Secrets
Add the following secrets to your GitHub repository:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `DOCKER_USERNAME` (if using Docker Hub)
- `DOCKER_PASSWORD` (if using Docker Hub)

### 3. Update Configuration Files

#### Update ECS Task Definition
Edit `aws/ecs-task-definition.json`:
- Replace `<account-id>` with your AWS account ID
- Adjust CPU/memory as needed
- Update environment variables

#### Update DVC Config
Edit `.dvc/config`:
- Update S3 bucket name
- Configure AWS credentials

### 4. Deploy via CI/CD

#### Automatic Deployment
1. Push code to `main` branch
2. GitHub Actions will:
   - Run tests
   - Build Docker image
   - Push to ECR
   - Deploy to ECS

#### Manual Deployment
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and tag
docker build -f docker/Dockerfile.api -t loan-default-api .
docker tag loan-default-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/loan-default:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/loan-default:latest

# Update ECS service
aws ecs update-service \
  --cluster loan-default-cluster \
  --service loan-default-service \
  --force-new-deployment \
  --region us-east-1
```

## Monitoring Setup

### 1. CloudWatch Logs
Logs are automatically sent to CloudWatch:
- Log group: `/ecs/loan-default`
- Stream prefix: `ecs`

### 2. View Logs
```bash
aws logs tail /ecs/loan-default --follow --region us-east-1
```

### 3. Setup Alarms
Create CloudWatch alarms for:
- High error rate
- High latency
- Container failures

## Monitoring & Drift Detection

### Run Monitoring Script
```bash
python scripts/monitor.py \
  --reference-data data/train.csv \
  --current-data data/production.csv
```

### Setup Scheduled Monitoring
Use AWS EventBridge to schedule regular monitoring runs.

## Troubleshooting

### Common Issues

1. **Docker build fails**
   - Check Dockerfile syntax
   - Verify all dependencies in requirements.txt

2. **ECS deployment fails**
   - Check task definition JSON syntax
   - Verify IAM permissions
   - Check CloudWatch logs

3. **API not responding**
   - Check health endpoint
   - Verify model files are loaded
   - Check container logs

4. **DVC push fails**
   - Verify AWS credentials
   - Check S3 bucket permissions
   - Verify bucket exists

## Rollback Procedure

### Rollback ECS Service
```bash
# List previous task definitions
aws ecs list-task-definitions --family-prefix loan-default-task

# Update service to previous revision
aws ecs update-service \
  --cluster loan-default-cluster \
  --service loan-default-service \
  --task-definition loan-default-task:<previous-revision> \
  --region us-east-1
```

## Production Checklist

- [ ] All tests passing
- [ ] Code reviewed and approved
- [ ] Configuration files updated
- [ ] AWS resources created
- [ ] Secrets configured
- [ ] Monitoring setup
- [ ] Alerts configured
- [ ] Documentation updated
- [ ] Backup strategy in place
- [ ] Rollback procedure tested

