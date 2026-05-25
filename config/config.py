"""
Configuration module for Customer Churn Prediction system
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Data configuration
DATASET_NAME = "WA_Fn-UseC_-_Telco_Customer_Churn.csv"
DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

TARGET_COLUMN = "Churn"
RANDOM_STATE = 42

# Train-Test Split
TRAIN_SIZE = 0.7
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# Model configuration
MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]
BEST_MODEL = "xgboost"

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Streamlit configuration
STREAMLIT_PAGE_CONFIG = {
    "page_title": "Customer Churn Prediction",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
MLFLOW_EXPERIMENT_NAME = "customer-churn-prediction"

# Evidently AI configuration
MONITORING_CONFIG = {
    "reference_data_split": 0.3,
    "drift_threshold": 0.05,
}

# Feature engineering
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
