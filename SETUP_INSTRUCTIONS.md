"""
SETUP_INSTRUCTIONS.md - Detailed setup guide
"""

# Customer Churn Prediction System - Setup Instructions

## 🚀 Prerequisites

- Python 3.9 or higher
- Git (for version control)
- Docker (optional, for containerization)
- pip (Python package manager)

## 📋 Step-by-Step Setup

### 1. Create Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:
- Data Processing: pandas, numpy, scikit-learn
- Modeling: xgboost, lightgbm
- Explainability: shap
- API: fastapi, uvicorn
- Dashboard: streamlit
- Monitoring: evidently, mlflow
- And more...

### 3. Setup Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your configuration if needed.

### 4. Download Dataset

```bash
python -c "from src.data_loader import download_dataset; download_dataset()"
```

This will download the IBM Telco Customer Churn dataset to `data/raw/`

### 5. Run the Complete Pipeline

**Quick Start (Automated):**
```bash
python quickstart.py
```

**Manual Step-by-Step:**

#### 5.1 Exploratory Data Analysis
```bash
python notebooks/01_eda.py
```
This generates:
- churn_distribution.png
- demographics_analysis.png
- categorical_analysis.png
- correlation_analysis.png

#### 5.2 Train Models
```bash
python train_pipeline.py
```
This will:
- Clean and preprocess data
- Create train/validation/test splits
- Train Logistic Regression, Random Forest, and XGBoost models
- Evaluate models and save the best one
- Generate SHAP explanations

#### 5.3 Model Evaluation & Analysis
```bash
python notebooks/04_evaluation_shap.py
```
This generates:
- confusion_matrix.png
- roc_curve.png
- pr_curve.png
- feature_importance_tree.png

### 6. Start the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 7. Start the Dashboard

In a new terminal:
```bash
streamlit run dashboard/app.py
```

Dashboard will be available at: http://localhost:8501

### 8. Run Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov=api
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

Services will be available at:
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- MLflow: http://localhost:5000

## 📁 Project Structure

```
customer-churn-prediction/
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned data
├── src/                     # Reusable modules
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── feature_engineering.py
│   ├── model_trainer.py
│   ├── evaluator.py
│   ├── explainability.py
│   └── monitoring.py
├── api/                     # FastAPI application
│   ├── main.py
│   └── schemas.py
├── dashboard/              # Streamlit app
│   └── app.py
├── notebooks/              # Analysis scripts
│   ├── 01_eda.py
│   └── 04_evaluation_shap.py
├── models/                 # Saved models
├── config/                 # Configuration
│   └── config.py
├── tests/                  # Unit tests
│   └── test_pipeline.py
├── requirements.txt        # Dependencies
├── Dockerfile              # Docker image
└── docker-compose.yml      # Docker compose config
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest --cov=src --cov=api --cov-report=html
```

## 📊 API Endpoints

### Health Check
```bash
curl http://localhost:8000/
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 24,
    "monthly_charges": 65.1,
    "total_charges": 1560.24,
    "internet_service": "Fiber optic",
    "contract": "Two year"
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict-batch \
  -F "file=@data/sample_data.csv"
```

### Get Metrics
```bash
curl http://localhost:8000/metrics
```

## 🔧 Troubleshooting

### Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Dataset Download Issues
- Check internet connection
- Verify URL is accessible: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- Try manual download and place in `data/raw/`

### Port Already in Use
```bash
# Change port for API
uvicorn api.main:app --port 8001

# Change port for Dashboard
streamlit run dashboard/app.py --server.port 8502
```

### SHAP Issues
SHAP may take time on large datasets. If it times out:
- Reduce sample size in explainability.py
- Use only a subset of test data

## 📝 Next Steps

1. **Customize Models**: Modify hyperparameters in `src/model_trainer.py`
2. **Add Features**: Create new features in `src/feature_engineering.py`
3. **Integrate MLflow**: Log experiments and model versions
4. **Setup Monitoring**: Use `src/monitoring.py` for production monitoring
5. **Deploy**: Use Docker or cloud platforms (AWS, GCP, Azure)

## 📚 Documentation

- [README.md](README.md) - Project overview
- [config/config.py](config/config.py) - Configuration settings
- [src/](src/) - Module documentation
- [API Documentation](http://localhost:8000/docs) - Interactive API docs

## 🤝 Contributing

Guidelines for contributing to this project:
1. Create a new branch for your feature
2. Make changes and test thoroughly
3. Update documentation
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

---

**Need Help?** Check the README.md or consult the documentation in each module.
