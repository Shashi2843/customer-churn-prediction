"""
Unit tests for data processing modules
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_cleaner import clean_data, split_data, handle_missing_values
from src.feature_engineering import identify_feature_types
from src.model_trainer import ModelTrainer


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'churn': [0, 1, 0, 1, 0],
        'age': [45, 35, 50, 40, 55],
        'tenure': [12, 24, 36, 48, 60],
        'monthly_charges': [65.1, 75.2, 55.3, 85.4, 45.5],
        'total_charges': [781.2, 1806.0, 1989.0, 4096.0, 2730.0],
        'internet_service': ['Fiber optic', 'DSL', 'Fiber optic', 'DSL', 'No'],
        'contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'Two year']
    })


def test_handle_missing_values(sample_data):
    """Test missing value handling"""
    data_with_missing = sample_data.copy()
    data_with_missing.loc[0, 'tenure'] = np.nan
    
    result = handle_missing_values(data_with_missing)
    
    assert result['tenure'].isnull().sum() == 0
    assert len(result) == len(data_with_missing)


def test_clean_data(sample_data):
    """Test data cleaning"""
    result = clean_data(sample_data)
    
    assert 'churn' in result.columns
    assert result['churn'].dtype in ['int64', 'int32']
    assert result.isnull().sum().sum() == 0


def test_split_data(sample_data):
    """Test data splitting"""
    df_clean = clean_data(sample_data)
    train, val, test = split_data(df_clean, train_size=0.5, val_size=0.25, test_size=0.25)
    
    assert len(train) + len(val) + len(test) == len(df_clean)
    assert len(train) > len(val)


def test_identify_feature_types(sample_data):
    """Test feature type identification"""
    X = sample_data.drop('churn', axis=1)
    
    cat_features, num_features = identify_feature_types(X)
    
    assert len(cat_features) > 0
    assert len(num_features) > 0
    assert 'internet_service' in cat_features
    assert 'tenure' in num_features


def test_model_trainer(sample_data):
    """Test model training"""
    X = sample_data.drop('churn', axis=1).values
    y = sample_data['churn'].values
    
    trainer = ModelTrainer()
    trainer.train_logistic_regression(X, y)
    
    assert 'logistic_regression' in trainer.models
    predictions = trainer.predict('logistic_regression', X)
    assert len(predictions) == len(y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
