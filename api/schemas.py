"""
Request and response schemas for FastAPI
"""
from pydantic import BaseModel
from typing import List, Dict, Optional


class PredictionRequest(BaseModel):
    """Request schema for single prediction"""
    age: int = 45
    tenure: int = 24
    monthly_charges: float = 65.1
    total_charges: float = 1560.24
    internet_service: str = "Fiber optic"
    contract: str = "Two year"
    online_security: str = "Yes"
    tech_support: str = "Yes"
    
    class Config:
        schema_extra = {
            "example": {
                "age": 45,
                "tenure": 24,
                "monthly_charges": 65.1,
                "total_charges": 1560.24,
                "internet_service": "Fiber optic",
                "contract": "Two year",
                "online_security": "Yes",
                "tech_support": "Yes"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    prediction: int
    probability: float
    risk_level: str
    message: str


class BatchPredictionItem(BaseModel):
    """Item in batch prediction response"""
    prediction: int
    probability: float
    risk_level: str


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction"""
    total_customers: int
    predictions: List[Dict]
    churn_count: int
    churn_rate: float


class MetricsResponse(BaseModel):
    """Response schema for metrics"""
    model_name: str
    metrics: Dict[str, float]


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    model: str
