"""
Streamlit Dashboard for Customer Churn Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
import requests
import logging

# Configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = logging.getLogger(__name__)

# API Base URL
API_URL = "http://localhost:8000"

# Cache for loading models
@st.cache_resource
def load_model_artifacts():
    """Load model and preprocessor"""
    try:
        models_dir = Path("models")
        model_path = models_dir / "xgboost.pkl"
        preprocessor_path = models_dir / "preprocessor.pkl"
        
        if model_path.exists() and preprocessor_path.exists():
            model = joblib.load(str(model_path))
            preprocessor = joblib.load(str(preprocessor_path))
            return model, preprocessor
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


@st.cache_data
def load_dataset():
    """Load the dataset for analysis"""
    try:
        data_path = Path("data/processed/processed_data.csv")
        if data_path.exists():
            return pd.read_csv(data_path)
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


def main():
    """Main dashboard application"""
    
    # Header
    st.title("📊 Customer Churn Prediction Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["Home", "Predictions", "Analytics", "Model Performance", "Batch Predictions"]
        )
        
        st.markdown("---")
        st.info(
            """
            This dashboard helps predict customer churn risk using machine learning.
            - **Predictions**: Predict churn for a single customer
            - **Analytics**: Explore customer data insights
            - **Model Performance**: View model metrics
            - **Batch**: Predict for multiple customers
            """
        )
    
    # Page routing
    if page == "Home":
        show_home()
    elif page == "Predictions":
        show_predictions()
    elif page == "Analytics":
        show_analytics()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "Batch Predictions":
        show_batch_predictions()


def show_home():
    """Home page"""
    st.header("Welcome to Customer Churn Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Features", 20)
    
    with col2:
        st.metric("Model Accuracy", "85%")
    
    with col3:
        st.metric("Dataset Size", "7,043")
    
    st.markdown("---")
    
    st.subheader("About This Project")
    st.markdown("""
    This is a comprehensive end-to-end Machine Learning system for predicting customer churn 
    in the telecommunications industry.
    
    **Key Features:**
    - 🤖 Multiple ML models (Logistic Regression, Random Forest, XGBoost)
    - 📊 Interactive visualizations and analytics
    - 🔍 SHAP explainability for model predictions
    - 📈 Real-time churn risk assessment
    - 🚀 Production-ready API and deployment
    
    **Dataset:**
    - IBM Telco Customer Churn dataset
    - 7,043 customers with 20 features
    - Binary classification problem
    """)


def show_predictions():
    """Single customer prediction page"""
    st.header("Single Customer Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Information")
        
        tenure = st.slider("Tenure (months)", 0, 72, 24)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.1)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1560.0)
        
        internet_service = st.selectbox(
            "Internet Service",
            ["Fiber optic", "DSL", "No"]
        )
        
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    
    with col2:
        st.subheader("Prediction Result")
        
        if st.button("🔮 Predict Churn", key="predict_button"):
            try:
                # Prepare data
                data = {
                    "tenure": tenure,
                    "monthly_charges": monthly_charges,
                    "total_charges": total_charges,
                    "internet_service": internet_service,
                    "contract": contract,
                    "online_security": online_security,
                    "tech_support": tech_support
                }
                
                # Call API
                response = requests.post(f"{API_URL}/predict", json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    probability = result['probability']
                    risk_level = result['risk_level']
                    
                    # Color based on risk
                    if risk_level == "High":
                        color = "🔴"
                    elif risk_level == "Medium":
                        color = "🟡"
                    else:
                        color = "🟢"
                    
                    st.metric(
                        "Churn Risk Level",
                        f"{color} {risk_level}",
                        f"{probability:.2%} probability"
                    )
                    
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=probability * 100,
                        title={'text': "Churn Risk (%)"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightgreen"},
                                {'range': [40, 70], 'color': "lightyellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Error in prediction. Please check your inputs.")
            
            except Exception as e:
                st.error(f"Error: {e}")


def show_analytics():
    """Analytics page"""
    st.header("Customer Analytics")
    
    df = load_dataset()
    
    if df is not None:
        st.subheader("Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", len(df))
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.markdown("---")
        
        # Sample data
        if st.checkbox("Show Sample Data"):
            st.dataframe(df.head(10))
        
        # Column selection for analysis
        col = st.selectbox("Select Column for Analysis", df.columns)
        
        if col:
            st.subheader(f"Analysis: {col}")
            
            # Check if numeric or categorical
            if df[col].dtype in ['int64', 'float64']:
                # Histogram
                fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Bar chart
                value_counts = df[col].value_counts()
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Distribution of {col}",
                    labels={"x": col, "y": "Count"}
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dataset not found. Please run the preprocessing pipeline first.")


def show_model_performance():
    """Model performance page"""
    st.header("Model Performance Metrics")
    
    try:
        response = requests.get(f"{API_URL}/metrics")
        
        if response.status_code == 200:
            data = response.json()
            
            metrics = data.get('metrics', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "ROC-AUC",
                    f"{metrics.get('roc_auc', 0):.3f}"
                )
            
            with col2:
                st.metric(
                    "F1-Score",
                    f"{metrics.get('f1', 0):.3f}"
                )
            
            with col3:
                st.metric(
                    "Precision",
                    f"{metrics.get('precision', 0):.3f}"
                )
            
            with col4:
                st.metric(
                    "Recall",
                    f"{metrics.get('recall', 0):.3f}"
                )
            
            st.markdown("---")
            
            # Metrics table
            metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
            st.dataframe(metrics_df)
        
        else:
            st.error("Could not fetch metrics from API")
    
    except Exception as e:
        st.error(f"Error: {e}")


def show_batch_predictions():
    """Batch prediction page"""
    st.header("Batch Predictions")
    
    st.markdown("Upload a CSV file with customer data to predict churn for multiple customers.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        if st.button("🚀 Run Batch Prediction"):
            try:
                files = {"file": uploaded_file}
                response = requests.post(f"{API_URL}/predict-batch", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Customers", result['total_customers'])
                    
                    with col2:
                        st.metric("Churn Count", result['churn_count'])
                    
                    with col3:
                        st.metric(
                            "Churn Rate",
                            f"{result['churn_rate']:.2%}"
                        )
                    
                    st.markdown("---")
                    
                    # Results table
                    results_df = pd.DataFrame(result['predictions'])
                    st.dataframe(results_df)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "predictions.csv",
                        "text/csv"
                    )
                
                else:
                    st.error("Error in batch prediction")
            
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
