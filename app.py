"""FastAPI application for loan default prediction service."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

from src.pipeline import MLPipeline
from src.utils.config import Config
from src.utils.logger import setup_logger

# Setup logging
setup_logger(level='INFO', log_file='logs/api.log')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Loan Default Prediction API",
    description="Production API for predicting loan default risk",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
config = Config()

# Global variables for model and preprocessors
model = None
preprocessors = None
pipeline = None


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    Client_Income: float = Field(..., description="Client income")
    Credit_Amount: float = Field(..., description="Credit amount")
    Loan_Annuity: float = Field(..., description="Loan annuity")
    Age_Days: int = Field(..., description="Age in days")
    # Add other required fields as needed
    # For brevity, showing key fields only


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    data: List[dict] = Field(..., description="List of loan applications")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: int = Field(..., description="Prediction (0=No Default, 1=Default)")
    probability: float = Field(..., description="Probability of default")
    risk_level: str = Field(..., description="Risk level (LOW, MEDIUM, HIGH)")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_loaded: bool


@app.on_event("startup")
async def load_model():
    """Load model and preprocessors on startup."""
    global model, preprocessors, pipeline
    
    try:
        logger.info("Loading model and preprocessors...")
        
        model_path = Path(config.get('model_registry.registry_path', 'models')) / \
                    config.get('model_registry.best_model_name', 'best_model.pkl')
        preprocessor_path = Path(config.get('model_registry.registry_path', 'models')) / \
                          config.get('model_registry.preprocessor_name', 'preprocessor.pkl')
        
        if model_path.exists() and preprocessor_path.exists():
            model = joblib.load(model_path)
            preprocessors = joblib.load(preprocessor_path)
            pipeline = MLPipeline()
            logger.info("✓ Model and preprocessors loaded successfully")
        else:
            logger.warning("Model files not found. Please train the model first.")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction.
    
    Args:
        request: Prediction request with loan application data
        
    Returns:
        Prediction response with prediction, probability, and risk level
    """
    if model is None or preprocessors is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        df = pd.DataFrame([request.dict()])
        
        # Preprocess
        df_processed, _ = pipeline.preprocess_data(df, fit=False)
        
        # Scale features
        if config.get('preprocessing.scale_features', True):
            X_scaled = preprocessors['scaler'].transform(df_processed)
        else:
            X_scaled = df_processed.values
        
        # Predict
        probability = model.predict_proba(X_scaled)[0, 1]
        prediction = int(probability >= 0.5)
        
        # Determine risk level
        if probability < 0.4:
            risk_level = "LOW"
        elif probability < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        return PredictionResponse(
            prediction=prediction,
            probability=float(probability),
            risk_level=risk_level
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions.
    
    Args:
        request: Batch prediction request with list of loan applications
        
    Returns:
        List of prediction responses
    """
    if model is None or preprocessors is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        # Preprocess
        df_processed, _ = pipeline.preprocess_data(df, fit=False)
        
        # Scale features
        if config.get('preprocessing.scale_features', True):
            X_scaled = preprocessors['scaler'].transform(df_processed)
        else:
            X_scaled = df_processed.values
        
        # Predict
        probabilities = model.predict_proba(X_scaled)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        
        # Create responses
        responses = []
        for pred, prob in zip(predictions, probabilities):
            if prob < 0.4:
                risk_level = "LOW"
            elif prob < 0.7:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            responses.append(PredictionResponse(
                prediction=int(pred),
                probability=float(prob),
                risk_level=risk_level
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_type": type(model).__name__,
        "model_loaded": True,
        "features_count": getattr(model, 'n_features_in_', 'Unknown')
    }
    
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

