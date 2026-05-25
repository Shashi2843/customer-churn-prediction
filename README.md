# Customer Churn Prediction System 📊

A complete end-to-end Machine Learning system that predicts customer churn using the IBM Telco Customer Churn dataset.

## 🎯 Project Overview

This project implements a production-level ML pipeline for predicting whether a customer will churn (leave the service). It includes:

- **EDA & Visualization**: Exploratory data analysis with Matplotlib and Plotly
- **Data Preprocessing**: Cleaning, handling missing values, and feature engineering
- **Model Development**: Logistic Regression, Random Forest, and XGBoost with comparison metrics
- **Explainability**: SHAP analysis for model interpretability
- **API**: FastAPI deployment with prediction endpoints
- **Dashboard**: Interactive Streamlit dashboard for end-users
- **Monitoring**: Data drift detection with Evidently AI
- **MLOps**: Experiment tracking with MLflow
- **Containerization**: Docker and Docker Compose support

## 📁 Project Structure

```
customer-churn-prediction/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned and preprocessed data
├── src/                        # Reusable Python modules
│   ├── data_loader.py         # Data loading utilities
│   ├── data_cleaner.py        # Data cleaning functions
│   ├── feature_engineering.py # Feature preprocessing
│   ├── model_trainer.py       # Model training pipeline
│   ├── evaluator.py           # Model evaluation metrics
│   └── explainability.py      # SHAP analysis
├── notebooks/                  # Jupyter notebooks for exploration
├── api/                        # FastAPI application
│   ├── main.py                # API endpoints
│   └── schemas.py             # Request/response schemas
├── dashboard/                  # Streamlit dashboard
│   └── app.py                 # Dashboard application
├── models/                     # Saved trained models
├── tests/                      # Unit tests
├── config/
│   └── config.py              # Configuration settings
├── logs/                       # Application logs
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Docker compose orchestration
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Clone Repository & Setup Environment

```bash
cd "c:\Users\sharm\OneDrive\Desktop\Capstone Project\customer-churn-prediction"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Environment Variables

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Download Dataset

```bash
python -c "from src.data_loader import download_dataset; download_dataset()"
```

### 4. Run Pipeline

```bash
# Exploratory Data Analysis
python notebooks/01_eda.py

# Data Preprocessing
python notebooks/02_preprocessing.py

# Model Training
python notebooks/03_model_training.py

# Model Evaluation & Explainability
python notebooks/04_evaluation_shap.py
```

### 5. Start API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## 🐳 Docker Deployment

### Build and Run with Docker Compose

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

The system will be available at:
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- MLflow: http://localhost:5000

## 📊 Dataset

**IBM Telco Customer Churn Dataset**
- **Records**: 7,043 customers
- **Features**: 20 features (demographic, account, and service usage)
- **Target**: Binary classification (Churn: Yes/No)

## 🔧 Technologies Used

| Category | Technology |
|----------|-----------|
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Modeling** | Scikit-learn, XGBoost, LightGBM |
| **Explainability** | SHAP |
| **API** | FastAPI, Uvicorn |
| **Dashboard** | Streamlit |
| **Monitoring** | Evidently AI |
| **MLOps** | MLflow |
| **Containerization** | Docker, Docker Compose |
| **Testing** | Pytest |
| **Visualization** | Matplotlib, Seaborn, Plotly |

## 📈 Model Performance

Models are compared using:
- **ROC-AUC**: Overall discrimination ability
- **F1-Score**: Balance between precision and recall
- **PR-AUC**: Precision-Recall curve for imbalanced data
- **Confusion Matrix**: TP, TN, FP, FN analysis

## 💡 Features Implemented

- ✅ Automated data pipeline
- ✅ Multiple model comparison
- ✅ SHAP explainability
- ✅ REST API with FastAPI
- ✅ Interactive Streamlit dashboard
- ✅ Model versioning with MLflow
- ✅ Data drift monitoring
- ✅ Docker containerization
- ✅ Unit tests
- ✅ Comprehensive logging

## 📝 API Endpoints

### GET /
Health check endpoint

### POST /predict
Predict churn for a single customer

**Request Body:**
```json
{
  "age": 65,
  "tenure": 2,
  "monthly_charges": 65.1,
  "total_charges": 130.2,
  "internet_service": "Fiber optic",
  "contract": "Month-to-month"
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.85,
  "message": "High churn risk"
}
```

### POST /predict-batch
Predict churn for multiple customers from a CSV file

### GET /metrics
Retrieve model performance metrics

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov=api

# Run specific test file
pytest tests/test_data_loader.py
```

## 📚 Documentation

Detailed documentation for each module is available in the respective files. Key documentation:

- [Data Loader](src/data_loader.py) - Loading and saving datasets
- [Feature Engineering](src/feature_engineering.py) - Feature preprocessing
- [Model Training](src/model_trainer.py) - Training pipeline
- [API Documentation](api/main.py) - API endpoints
- [Dashboard Guide](dashboard/app.py) - Dashboard usage

## 🔍 Monitoring with Evidently AI

The system includes automated monitoring for:
- **Data Drift**: Distribution changes in features
- **Target Drift**: Changes in churn proportion
- **Data Quality**: Missing values, outliers
- **Model Performance**: Drift in model predictions

Generate monitoring reports:
```bash
python src/monitoring.py
```

## 🛠️ Development Workflow

1. **Data Exploration**: Use notebooks for EDA
2. **Preprocessing**: Implement in `src/data_cleaner.py`
3. **Feature Engineering**: Add features in `src/feature_engineering.py`
4. **Model Development**: Train models in `src/model_trainer.py`
5. **Evaluation**: Evaluate using `src/evaluator.py`
6. **Deployment**: Serve via `api/main.py` and `dashboard/app.py`
7. **Monitoring**: Track with `src/monitoring.py`

## 📞 Support

For issues or questions, please refer to the documentation or create an issue.

## 📄 License

This project is open source and available under the MIT License.

---

**Author**: Data Science Team  
**Last Updated**: 2026-05-25  
**Version**: 1.0.0
