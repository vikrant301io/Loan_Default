# Loan Default Prediction - System Architecture Design

## Document Information
- **Project**: Loan Default Prediction System
- **Version**: 1.0
- **Date**: 2025-11-17
- **Author**: ML Engineering Team
- **Status**: Production-Ready

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Architecture Diagrams](#architecture-diagrams)
4. [Component Details](#component-details)
5. [Data Flow](#data-flow)
6. [Technology Stack](#technology-stack)
7. [Deployment Architecture](#deployment-architecture)
8. [Security Architecture](#security-architecture)
9. [Monitoring & Observability](#monitoring--observability)
10. [Scalability & Performance](#scalability--performance)
11. [Disaster Recovery](#disaster-recovery)
12. [Cost Analysis](#cost-analysis)

---

## 1. Executive Summary

### 1.1 Purpose
This document describes the system architecture for a production-ready Machine Learning pipeline that predicts loan default risk. The system implements MLOps best practices including data versioning, experiment tracking, model monitoring, and automated deployment.

### 1.2 Key Objectives
- **Accuracy**: Achieve high prediction accuracy (Target: >90%)
- **Scalability**: Handle high-volume prediction requests
- **Reliability**: 99.9% uptime target
- **Maintainability**: Modular, well-documented codebase
- **Reproducibility**: Version-controlled data and models
- **Observability**: Comprehensive monitoring and alerting

### 1.3 Architecture Principles
- **Modularity**: Separation of concerns, reusable components
- **Scalability**: Horizontal scaling capability
- **Reliability**: Fault tolerance and error handling
- **Security**: Data encryption, access controls
- **Observability**: Logging, monitoring, tracing
- **Automation**: CI/CD, automated testing, deployment

---

## 2. System Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │ Web App      │  │ Mobile App   │  │ API Clients  │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              FastAPI REST API Service                        │  │
│  │  - /predict (single)                                         │  │
│  │  - /predict/batch (batch)                                    │  │
│  │  - /health (health check)                                    │  │
│  │  - /model/info (model metadata)                             │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │ Prediction   │  │ Preprocessing│  │ Model        │            │
│  │ Service     │  │ Pipeline    │  │ Registry     │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA & ML LAYER                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │ Data        │  │ MLflow       │  │ DVC          │            │
│  │ Storage     │  │ Tracking     │  │ Versioning   │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MONITORING & OBSERVABILITY                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │ Drift        │  │ Performance  │  │ CloudWatch   │            │
│  │ Detection    │  │ Monitoring   │  │ Logs         │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 System Components

#### 2.2.1 Data Management
- **Data Ingestion**: CSV file loading with validation
- **Data Versioning**: DVC with S3 backend
- **Data Storage**: S3 for raw and processed data
- **Data Pipeline**: Automated preprocessing pipeline

#### 2.2.2 Model Development
- **Training Pipeline**: Automated model training
- **Experiment Tracking**: MLflow for experiment management
- **Model Registry**: Versioned model storage
- **Feature Engineering**: Domain-specific feature creation

#### 2.2.3 Model Serving
- **API Service**: FastAPI REST API
- **Containerization**: Docker containers
- **Orchestration**: AWS ECS
- **Load Balancing**: Application Load Balancer

#### 2.2.4 Monitoring
- **Drift Detection**: Statistical tests (KS, PSI)
- **Performance Monitoring**: Real-time metrics
- **Logging**: Structured logging to CloudWatch
- **Alerting**: Automated alerts on anomalies

---

## 3. Architecture Diagrams

### 3.1 Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│  │   Data   │───▶│  Preproc │───▶│ Feature  │            │
│  │ Ingestion│    │          │    │ Engineer │            │
│  └──────────┘    └──────────┘    └──────────┘            │
│       │                │                │                  │
│       └────────────────┴────────────────┘                │
│                          │                                 │
│                          ▼                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│  │  Model   │───▶│  MLflow  │───▶│   Model  │            │
│  │ Training │    │ Tracking │    │ Registry │            │
│  └──────────┘    └──────────┘    └──────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   SERVING PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│  │   API    │───▶│ Preproc  │───▶│  Model   │            │
│  │ Request  │    │ Pipeline │    │ Inference│            │
│  └──────────┘    └──────────┘    └──────────┘            │
│       │                │                │                  │
│       └────────────────┴────────────────┘                │
│                          │                                 │
│                          ▼                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│  │ Response │    │ Logging  │    │ Monitor  │            │
│  │          │    │          │    │          │            │
│  └──────────┘    └──────────┘    └──────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AWS CLOUD                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              INTERNET                                │  │
│  └─────────────────────────────────────────────────────┘  │
│                          │                                 │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────────┐  │
│  │         APPLICATION LOAD BALANCER (ALB)              │  │
│  │         - Health Checks                              │  │
│  │         - SSL Termination                            │  │
│  │         - Request Routing                            │  │
│  └─────────────────────────────────────────────────────┘  │
│                          │                                 │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              ECS CLUSTER                             │  │
│  │  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │ ECS Service  │  │ ECS Service  │  (Auto-scaled)  │  │
│  │  │  (Task 1)    │  │  (Task 2)    │                │  │
│  │  └──────────────┘  └──────────────┘                │  │
│  │         │                  │                        │  │
│  │         └──────────┬───────┘                        │  │
│  │                    ▼                                 │  │
│  │         ┌──────────────────┐                         │  │
│  │         │  FastAPI Container│                        │  │
│  │         │  - Model Loading │                        │  │
│  │         │  - Prediction    │                        │  │
│  │         └──────────────────┘                         │  │
│  └─────────────────────────────────────────────────────┘  │
│                          │                                 │
│         ┌─────────────────┴─────────────────┐             │
│         │                                   │             │
│         ▼                                   ▼             │
│  ┌──────────────┐                  ┌──────────────┐     │
│  │  ECR         │                  │  S3          │     │
│  │  (Images)    │                  │  (Data/Models)│     │
│  └──────────────┘                  └──────────────┘     │
│                                                           │
│  ┌──────────────┐                  ┌──────────────┐     │
│  │  CloudWatch  │                  │  Secrets      │     │
│  │  (Logs/Metrics)│                │  Manager     │     │
│  └──────────────┘                  └──────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Data Flow Diagram

```
┌─────────────┐
│  Raw Data   │
│  (CSV)      │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Data Ingestion │
│  & Validation   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐      ┌──────────┐
│  Data Cleaning  │─────▶│   DVC    │
│  & Preprocessing│      │ Version  │
└──────┬──────────┘      └──────────┘
       │
       ▼
┌─────────────────┐
│   Feature       │
│   Engineering   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐      ┌──────────┐
│  Train/Test     │─────▶│  MLflow  │
│  Split          │      │ Tracking │
└──────┬──────────┘      └──────────┘
       │
       ▼
┌─────────────────┐
│  Model Training │
│  (5 Algorithms) │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐      ┌──────────┐
│  Model          │─────▶│  Model   │
│  Evaluation     │      │ Registry │
└──────┬──────────┘      └──────────┘
       │
       ▼
┌─────────────────┐
│  Best Model     │
│  Selection      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Model          │
│  Deployment     │
└─────────────────┘
```

---

## 4. Component Details

### 4.1 Data Ingestion Module
**Location**: `src/data/ingestion.py`

**Responsibilities**:
- Load CSV data files
- Validate data structure
- Handle directory or file paths
- Provide data statistics

**Key Methods**:
- `load_data()`: Load CSV file
- `load_data_with_validation()`: Load with validation
- `get_data_info()`: Extract dataset information

### 4.2 Data Validation Module
**Location**: `src/data/validation.py`

**Responsibilities**:
- Validate data quality
- Check for missing values
- Verify target distribution
- Schema validation

**Key Methods**:
- `validate_data()`: Comprehensive validation
- `validate_schema()`: Schema checking

### 4.3 Preprocessing Pipeline
**Location**: `src/preprocessing/`

**Components**:
- **cleaner.py**: Missing values, outliers, duplicates
- **encoder.py**: Categorical encoding
- **feature_engineering.py**: Domain features
- **scaler.py**: Feature scaling

**Pipeline Flow**:
1. Remove ID columns
2. Handle missing values
3. Handle outliers
4. Encode categorical variables
5. Create engineered features
6. Scale numerical features

### 4.4 Model Training Module
**Location**: `src/models/trainer.py`

**Responsibilities**:
- Train multiple models
- Handle class imbalance
- Evaluate model performance
- Select best model

**Supported Models**:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

### 4.5 Model Evaluation Module
**Location**: `src/evaluation/metrics.py`

**Responsibilities**:
- Calculate comprehensive metrics
- Generate evaluation reports
- Feature importance analysis
- Threshold analysis

**Metrics Calculated**:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix
- Specificity, Sensitivity

### 4.6 API Service
**Location**: `app.py`

**Technology**: FastAPI

**Endpoints**:
- `GET /`: Health check
- `GET /health`: Detailed health status
- `POST /predict`: Single prediction
- `POST /predict/batch`: Batch predictions
- `GET /model/info`: Model information

**Features**:
- Request validation (Pydantic)
- Error handling
- Logging
- CORS support

### 4.7 Monitoring Module
**Location**: `src/monitoring/`

**Components**:
- **drift_detection.py**: Data drift detection
- **performance_monitor.py**: Performance tracking

**Drift Detection Methods**:
- Kolmogorov-Smirnov test
- Population Stability Index (PSI)

**Performance Metrics**:
- Accuracy, Precision, Recall, F1, ROC-AUC
- Threshold-based alerting
- Performance degradation detection

---

## 5. Data Flow

### 5.1 Training Data Flow

```
1. Raw CSV Data
   ↓
2. Data Ingestion (load, validate)
   ↓
3. Data Cleaning (missing values, outliers)
   ↓
4. Categorical Encoding
   ↓
5. Feature Engineering (20+ features)
   ↓
6. Feature Scaling
   ↓
7. Train/Test Split
   ↓
8. Class Imbalance Handling (SMOTE)
   ↓
9. Model Training (5 algorithms)
   ↓
10. Model Evaluation
   ↓
11. Best Model Selection
   ↓
12. Model & Preprocessor Saving
   ↓
13. MLflow Logging
```

### 5.2 Inference Data Flow

```
1. API Request (JSON)
   ↓
2. Request Validation
   ↓
3. Load Preprocessor
   ↓
4. Data Preprocessing
   ↓
5. Feature Engineering
   ↓
6. Feature Scaling
   ↓
7. Load Model
   ↓
8. Prediction
   ↓
9. Response Formatting
   ↓
10. Logging & Monitoring
   ↓
11. API Response (JSON)
```

---

## 6. Technology Stack

### 6.1 Core Technologies

**Programming Language**: Python 3.9+

**ML Frameworks**:
- scikit-learn 1.3.2
- XGBoost 2.0.3
- LightGBM 4.1.0
- imbalanced-learn 0.11.0

**Data Processing**:
- pandas 2.1.4
- numpy 1.24.3
- scipy 1.11.4

**MLOps Tools**:
- MLflow 2.9.2 (Experiment tracking)
- DVC 3.38.1 (Data versioning)

**API Framework**:
- FastAPI 0.104.1
- Uvicorn 0.24.0
- Pydantic 1.10.13

**Visualization**:
- matplotlib 3.8.2
- seaborn 0.13.0

### 6.2 Infrastructure

**Containerization**:
- Docker
- Docker Compose

**Cloud Services (AWS)**:
- ECR (Elastic Container Registry)
- ECS (Elastic Container Service)
- S3 (Simple Storage Service)
- CloudWatch (Logging & Monitoring)
- Secrets Manager (Credential management)

**CI/CD**:
- GitHub Actions
- Docker Build & Push
- Automated Deployment

### 6.3 Development Tools

**Testing**:
- pytest 7.4.3
- pytest-cov 4.1.0
- pytest-mock 3.12.0

**Code Quality**:
- black 23.11.0 (Code formatting)
- flake8 6.1.0 (Linting)
- mypy 1.7.1 (Type checking)

**Notebooks**:
- Jupyter 1.0.0
- ipykernel 6.26.0

---

## 7. Deployment Architecture

### 7.1 Container Architecture

**Docker Images**:
1. **Training Image** (`Dockerfile.train`)
   - For model training jobs
   - Includes full ML stack
   - Can run training pipelines

2. **API Image** (`Dockerfile.api`)
   - For serving predictions
   - Optimized for inference
   - Includes FastAPI server

### 7.2 AWS ECS Deployment

**Task Definition**:
- **CPU**: 1024 (1 vCPU)
- **Memory**: 2048 MB (2 GB)
- **Network Mode**: awsvpc
- **Platform**: Fargate

**Service Configuration**:
- **Desired Count**: 2 (for high availability)
- **Auto Scaling**: Enabled
- **Health Checks**: Configured
- **Load Balancer**: Application Load Balancer

**Container Configuration**:
- **Port**: 8080
- **Health Check**: HTTP GET /health
- **Logging**: CloudWatch Logs
- **Environment Variables**: From Secrets Manager

### 7.3 Deployment Process

```
1. Code Push to GitHub
   ↓
2. GitHub Actions Triggered
   ↓
3. Run Tests & Linting
   ↓
4. Build Docker Image
   ↓
5. Push to ECR
   ↓
6. Update ECS Task Definition
   ↓
7. Deploy to ECS Service
   ↓
8. Health Check Verification
   ↓
9. Traffic Routing (if successful)
```

---

## 8. Security Architecture

### 8.1 Security Layers

**Network Security**:
- VPC with private subnets
- Security groups for access control
- Application Load Balancer with SSL/TLS

**Data Security**:
- Encryption at rest (S3, EBS)
- Encryption in transit (TLS 1.2+)
- Secrets management (AWS Secrets Manager)

**Access Control**:
- IAM roles and policies
- Least privilege principle
- API authentication (recommended: JWT/OAuth)

**Container Security**:
- Minimal base images
- No root user execution
- Security scanning (ECR image scanning)
- Regular updates

### 8.2 Compliance

**Data Privacy**:
- PII handling procedures
- Data retention policies
- Audit logging

**Regulatory**:
- GDPR considerations
- Financial regulations compliance
- Audit trail maintenance

---

## 9. Monitoring & Observability

### 9.1 Logging Strategy

**Application Logs**:
- Structured logging (JSON format)
- Log levels: DEBUG, INFO, WARNING, ERROR
- Log aggregation: CloudWatch Logs

**Access Logs**:
- API request/response logging
- Performance metrics
- Error tracking

### 9.2 Metrics & Monitoring

**Application Metrics**:
- Request rate
- Response time (p50, p95, p99)
- Error rate
- Prediction latency

**Model Metrics**:
- Prediction accuracy
- Model performance (precision, recall, F1)
- Feature distribution
- Prediction distribution

**Infrastructure Metrics**:
- CPU utilization
- Memory usage
- Network I/O
- Container health

### 9.3 Alerting

**Alert Conditions**:
- Model performance degradation (>5% drop)
- Data drift detected
- High error rate (>1%)
- High latency (>500ms p95)
- Container failures
- Low prediction confidence

**Alert Channels**:
- CloudWatch Alarms
- SNS notifications
- Email alerts
- Slack integration (optional)

### 9.4 Dashboards

**Operational Dashboard**:
- System health overview
- Request metrics
- Error rates
- Resource utilization

**ML Dashboard**:
- Model performance trends
- Prediction distributions
- Drift detection results
- Feature importance changes

---

## 10. Scalability & Performance

### 10.1 Horizontal Scaling

**Auto-Scaling Configuration**:
- **Min Tasks**: 2
- **Max Tasks**: 10
- **Target CPU**: 70%
- **Target Memory**: 80%
- **Scale-up Cooldown**: 60s
- **Scale-down Cooldown**: 300s

**Load Distribution**:
- Application Load Balancer
- Round-robin routing
- Health check based routing

### 10.2 Performance Optimization

**Model Optimization**:
- Model caching in memory
- Batch prediction support
- Async processing for large batches

**API Optimization**:
- Request batching
- Response caching (for similar requests)
- Connection pooling

**Data Optimization**:
- Efficient data loading
- Feature caching
- Precomputed features (future)

### 10.3 Capacity Planning

**Expected Load**:
- **Peak Requests**: 1000 requests/minute
- **Average Latency**: <100ms
- **Batch Size**: Up to 1000 records

**Resource Requirements**:
- **Per Container**: 1 vCPU, 2GB RAM
- **Total Capacity**: 10 containers (max)
- **Storage**: 50GB (models + data)

---

## 11. Disaster Recovery

### 11.1 Backup Strategy

**Data Backups**:
- S3 versioning enabled
- DVC remote storage (S3)
- Daily automated backups

**Model Backups**:
- MLflow artifact storage
- ECR image versioning
- Model registry backups

**Configuration Backups**:
- Infrastructure as Code (Terraform/CloudFormation)
- Configuration files in version control

### 11.2 Recovery Procedures

**Data Recovery**:
- Restore from S3 backups
- DVC data pull
- Point-in-time recovery (if enabled)

**Service Recovery**:
- ECS service auto-recovery
- Multi-AZ deployment
- Previous task definition rollback

**Disaster Recovery Plan**:
1. Identify failure type
2. Assess impact
3. Execute recovery procedure
4. Verify system health
5. Document incident

---

## 12. Cost Analysis

### 12.1 Infrastructure Costs (Monthly)

**Compute (ECS Fargate)**:
- 2 tasks × 1 vCPU × 2GB × $0.04/vCPU-hour × 730 hours = ~$58
- 2 tasks × 2GB × $0.004/GB-hour × 730 hours = ~$12
- **Subtotal**: ~$70/month

**Storage (S3)**:
- Data storage: 10GB × $0.023/GB = ~$0.23
- Model artifacts: 5GB × $0.023/GB = ~$0.12
- **Subtotal**: ~$0.35/month

**Container Registry (ECR)**:
- Storage: 5GB × $0.10/GB = ~$0.50
- **Subtotal**: ~$0.50/month

**Monitoring (CloudWatch)**:
- Logs: 10GB × $0.50/GB = ~$5
- Metrics: ~$1
- **Subtotal**: ~$6/month

**Load Balancer (ALB)**:
- Base: $16.20/month
- LCU charges: ~$5
- **Subtotal**: ~$21/month

**Data Transfer**:
- Outbound: ~$5/month
- **Subtotal**: ~$5/month

**Total Estimated Monthly Cost**: ~$103/month

### 12.2 Cost Optimization Strategies

1. **Right-sizing**: Monitor and adjust CPU/memory
2. **Reserved Capacity**: Consider for predictable workloads
3. **S3 Lifecycle Policies**: Archive old data
4. **Auto-scaling**: Scale down during low traffic
5. **Spot Instances**: For non-production workloads

---

## 13. Appendix

### 13.1 Component Interaction Sequence

**Training Sequence**:
```
DataIngestion → DataValidator → DataCleaner → 
CategoricalEncoder → FeatureEngineer → FeatureScaler → 
ModelTrainer → ModelEvaluator → MLflowTracker
```

**Inference Sequence**:
```
API Request → Request Validator → Preprocessor → 
FeatureEngineer → FeatureScaler → Model → 
Response Formatter → Logger → API Response
```

### 13.2 API Specifications

**Request Format** (Single Prediction):
```json
{
  "Client_Income": 50000,
  "Credit_Amount": 100000,
  "Loan_Annuity": 3000,
  "Age_Days": -10950,
  ...
}
```

**Response Format**:
```json
{
  "prediction": 0,
  "probability": 0.15,
  "risk_level": "LOW"
}
```

### 13.3 Configuration Management

**Configuration Files**:
- `configs/config.yaml`: Main configuration
- `configs/model_config.yaml`: Model-specific config
- Environment variables for secrets

**Configuration Hierarchy**:
1. Environment variables (highest priority)
2. Config file values
3. Default values (lowest priority)

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-17 | ML Engineering Team | Initial architecture design |

---

## References

- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- DVC Documentation: https://dvc.org/doc
- AWS ECS Documentation: https://docs.aws.amazon.com/ecs/
- FastAPI Documentation: https://fastapi.tiangolo.com/

