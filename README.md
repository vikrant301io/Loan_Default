# Loan Default Prediction - Production ML Pipeline

A production-ready, modular machine learning pipeline for loan default prediction with comprehensive MLOps practices.

## Project Structure

```
Loan_Default/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   └── validation.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   ├── encoder.py
│   │   └── feature_engineering.py
│   ├── eda/
│   │   ├── __init__.py
│   │   └── analyzer.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── registry.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── drift_detection.py
│   │   └── performance_monitor.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logger.py
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
├── configs/
│   ├── config.yaml
│   └── model_config.yaml
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.train
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── monitor.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── requirements.txt
├── setup.py
├── .dvcignore
├── .gitignore
└── README.md
```

## Features

- **Modular Architecture**: Clean separation of concerns
- **Data Versioning**: DVC for data and model versioning
- **Experiment Tracking**: MLflow integration
- **Model Monitoring**: Drift detection and performance monitoring
- **Containerization**: Docker support
- **CI/CD**: GitHub Actions workflows
- **Cloud Deployment**: AWS ECR/ECS configurations
- **Unit Testing**: Comprehensive test coverage

## Setup

1. **Clone and setup environment:**
```bash
git clone <repo-url>
cd Loan_Default
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Initialize DVC:**
```bash
dvc init
dvc remote add -d storage s3://your-bucket/dvc-storage
```

3. **Setup MLflow:**
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

## Usage

### Training
```bash
python scripts/train.py --config configs/config.yaml
```

### Prediction
```bash
python scripts/predict.py --model-path models/best_model.pkl --data data/test.csv
```

### Monitoring
```bash
python scripts/monitor.py --reference-data data/train.csv --current-data data/production.csv
```

## Deployment

### Docker
```bash
docker build -f docker/Dockerfile -t loan-default-predictor .
docker run -p 8080:8080 loan-default-predictor
```

### AWS ECS
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag loan-default-predictor:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/loan-default:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/loan-default:latest
```

## Testing

```bash
pytest tests/ -v
```

## Project Structure Details

### Source Code (`src/`)

- **`data/`**: Data ingestion and validation
  - `ingestion.py`: CSV data loading
  - `validation.py`: Data quality checks

- **`preprocessing/`**: Data preprocessing pipeline
  - `cleaner.py`: Missing values, outliers, duplicates
  - `encoder.py`: Categorical encoding
  - `feature_engineering.py`: Domain-specific features
  - `scaler.py`: Feature scaling (Robust/Standard)

- **`eda/`**: Exploratory Data Analysis
  - `analyzer.py`: Comprehensive EDA with visualizations

- **`models/`**: Model training and management
  - `trainer.py`: Multi-model training with hyperparameters
  - `registry.py`: Model versioning and storage

- **`evaluation/`**: Model evaluation
  - `metrics.py`: Comprehensive metrics calculation

- **`monitoring/`**: Production monitoring
  - `drift_detection.py`: Data drift detection (KS test, PSI)
  - `performance_monitor.py`: Model performance tracking

- **`utils/`**: Utilities
  - `config.py`: Configuration management
  - `logger.py`: Logging setup
  - `mlflow_tracker.py`: MLflow integration

### Testing (`tests/`)

- `test_data.py`: Data ingestion and validation tests
- `test_preprocessing.py`: Preprocessing pipeline tests
- `test_models.py`: Model training tests
- `test_evaluation.py`: Evaluation metrics tests
- `test_pipeline.py`: End-to-end pipeline tests

### Configuration (`configs/`)

- `config.yaml`: Main configuration file
- `model_config.yaml`: Model-specific configurations

### Deployment (`docker/`, `aws/`)

- `Dockerfile.api`: Production API container
- `Dockerfile.train`: Training container
- `docker-compose.yml`: Local development
- `ecs-task-definition.json`: AWS ECS configuration
- `deploy.sh`: Deployment script

### CI/CD (`.github/workflows/`)

- `ci.yml`: Continuous Integration (testing, linting)
- `cd.yml`: Continuous Deployment (AWS ECR/ECS)

## Key Features Implemented

✅ **Modular Architecture**: Clean separation of concerns
✅ **Data Versioning**: DVC with S3 backend
✅ **Experiment Tracking**: MLflow integration
✅ **Model Monitoring**: Drift detection and performance tracking
✅ **Containerization**: Docker support for training and serving
✅ **CI/CD**: GitHub Actions workflows
✅ **Cloud Deployment**: AWS ECR/ECS configurations
✅ **Unit Testing**: Comprehensive pytest test suite
✅ **API Serving**: FastAPI REST API
✅ **Documentation**: System architecture and deployment guides

## Technology Stack

- **ML Frameworks**: scikit-learn, XGBoost, LightGBM
- **MLOps**: MLflow, DVC
- **API**: FastAPI, Uvicorn
- **Containerization**: Docker
- **Cloud**: AWS (ECR, ECS, S3, CloudWatch)
- **CI/CD**: GitHub Actions
- **Testing**: pytest, pytest-cov
- **Monitoring**: Evidently (drift detection)

## Quick Start

### 1. Setup Environment
```bash
git clone <repo-url>
cd Loan_Default
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Initialize DVC
```bash
dvc init
dvc remote add -d storage s3://your-bucket/dvc-storage
```

### 3. Train Model
```bash
python scripts/train.py --config configs/config.yaml
```

### 4. Run API Locally
```bash
docker build -f docker/Dockerfile.api -t loan-default-api .
docker run -p 8080:8080 loan-default-api
```

### 5. Run Tests
```bash
pytest tests/ -v --cov=src
```

## Documentation

- [System Architecture](docs/SYSTEM_ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)

## License

MIT

