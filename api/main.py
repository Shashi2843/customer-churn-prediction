"""
FastAPI application for Customer Churn prediction
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from typing import Optional
import os

from config.config import MODELS_DIR, PROCESSED_DATA_DIR
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    MetricsResponse
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using ML models",
    version="1.0.0"
)

# Global variables for model caching
MODEL = None
PREPROCESSOR = None
BEST_MODEL_NAME = "xgboost"


def load_artifacts():
    """Load model and preprocessor on startup"""
    global MODEL, PREPROCESSOR
    
    try:
        preprocessor_path = MODELS_DIR / "preprocessor.pkl"

        # Preferred and fallback model paths
        preferred = MODELS_DIR / f"{BEST_MODEL_NAME}.pkl"
        fallback = MODELS_DIR / "logistic_regression.pkl"

        if preferred.exists():
            model_path = preferred
            loaded_name = BEST_MODEL_NAME
        elif fallback.exists():
            model_path = fallback
            loaded_name = "logistic_regression"
            logger.warning(f"Preferred model not found; falling back to {model_path}")
        else:
            logger.error(f"No model found at {preferred} or {fallback}")
            raise FileNotFoundError(f"Model file not found: {preferred} or {fallback}")

        if not preprocessor_path.exists():
            logger.error(f"Preprocessor not found at {preprocessor_path}")
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

        MODEL = joblib.load(str(model_path))
        PREPROCESSOR = joblib.load(str(preprocessor_path))

        logger.info(f"Model '{loaded_name}' loaded from {model_path}")
        logger.info("Preprocessor loaded successfully")
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_artifacts()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Customer Churn Prediction API is running",
        "model": BEST_MODEL_NAME
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict churn for a single customer
    
    Args:
        request: Customer data
        
    Returns:
        PredictionResponse with prediction and probability
    """
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        data_dict = request.dict()
        X = pd.DataFrame([data_dict])
        
        # Preprocess
        X_processed = PREPROCESSOR.transform(X)
        
        # Predict
        prediction = MODEL.predict(X_processed)[0]
        probability = MODEL.predict_proba(X_processed)[0][1]
        
        # Determine churn risk
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            message=f"Churn probability: {probability:.2%}"
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Predict churn for multiple customers from CSV file
    
    Args:
        file: CSV file with customer data
        
    Returns:
        JSON response with predictions
    """
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(contents.decode('utf8')))
        
        # Preprocess
        X_processed = PREPROCESSOR.transform(df)
        
        # Predict
        predictions = MODEL.predict(X_processed)
        probabilities = MODEL.predict_proba(X_processed)[:, 1]
        
        # Create response
        results = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities,
            'risk_level': ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' for p in probabilities]
        })
        
        return BatchPredictionResponse(
            total_customers=len(df),
            predictions=results.to_dict('records'),
            churn_count=int(predictions.sum()),
            churn_rate=float(predictions.mean())
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get model performance metrics
    
    Returns:
        MetricsResponse with model metrics
    """
    metrics_path = MODELS_DIR / "metrics.json"
    
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found")
    
    try:
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        return MetricsResponse(
            model_name=BEST_MODEL_NAME,
            metrics=metrics
        )
    
    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_name": BEST_MODEL_NAME,
        "model_type": type(MODEL).__name__,
        "preprocessor_type": type(PREPROCESSOR).__name__,
        "status": "loaded"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
