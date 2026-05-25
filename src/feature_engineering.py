"""
Feature engineering and preprocessing module
"""
import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

logger = logging.getLogger(__name__)


def identify_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify categorical and numerical features
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (categorical_features, numerical_features)
    """
    categorical = df.select_dtypes(include=['object']).columns.tolist()
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    
    logger.info(f"Categorical features: {categorical}")
    logger.info(f"Numerical features: {numerical}")
    
    return categorical, numerical


def create_preprocessing_pipeline(
    categorical_features: List[str],
    numerical_features: List[str]
) -> ColumnTransformer:
    """
    Create preprocessing pipeline for feature scaling and encoding
    
    Args:
        categorical_features: List of categorical column names
        numerical_features: List of numerical column names
        
    Returns:
        ColumnTransformer: Fitted preprocessing pipeline
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
             categorical_features)
        ]
    )
    
    logger.info("Preprocessing pipeline created")
    return preprocessor


def preprocess_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_features: List[str],
    numerical_features: List[str],
    fit_preprocessor: bool = True,
    preprocessor: ColumnTransformer = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Preprocess features using standard scaling and one-hot encoding
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        categorical_features: List of categorical columns
        numerical_features: List of numerical columns
        fit_preprocessor: Whether to fit the preprocessor
        preprocessor: Existing preprocessor object
        
    Returns:
        Tuple of (X_train_processed, X_val_processed, X_test_processed, preprocessor)
    """
    if fit_preprocessor:
        preprocessor = create_preprocessing_pipeline(categorical_features, numerical_features)
        X_train_processed = preprocessor.fit_transform(X_train)
        logger.info("Preprocessor fitted on training data")
    else:
        if preprocessor is None:
            raise ValueError("Preprocessor must be provided if fit_preprocessor=False")
        X_train_processed = preprocessor.transform(X_train)
    
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    logger.info(f"Features preprocessed - Shape: {X_train_processed.shape}")
    
    return X_train_processed, X_val_processed, X_test_processed, preprocessor


def save_preprocessor(preprocessor: ColumnTransformer, file_path: str):
    """
    Save preprocessing pipeline
    
    Args:
        preprocessor: ColumnTransformer object
        file_path: Path to save the preprocessor
    """
    joblib.dump(preprocessor, file_path)
    logger.info(f"Preprocessor saved to {file_path}")


def load_preprocessor(file_path: str) -> ColumnTransformer:
    """
    Load preprocessing pipeline
    
    Args:
        file_path: Path to load the preprocessor
        
    Returns:
        ColumnTransformer: Loaded preprocessor
    """
    preprocessor = joblib.load(file_path)
    logger.info(f"Preprocessor loaded from {file_path}")
    return preprocessor


def get_feature_names(
    preprocessor: ColumnTransformer,
    categorical_features: List[str],
    numerical_features: List[str]
) -> List[str]:
    """
    Get feature names after preprocessing
    
    Args:
        preprocessor: Fitted preprocessor
        categorical_features: Original categorical features
        numerical_features: Original numerical features
        
    Returns:
        List of feature names after preprocessing
    """
    feature_names = (
        numerical_features +
        preprocessor.named_transformers_['cat'].get_feature_names_out(
            categorical_features
        ).tolist()
    )
    return feature_names


def handle_class_imbalance(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle class imbalance using SMOTE
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    try:
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        logger.info(f"Class balance after SMOTE - {np.bincount(y_resampled)}")
        return X_resampled, y_resampled
    except ImportError:
        logger.warning("imbalanced-learn not installed. Skipping SMOTE.")
        return X_train, y_train


def create_interaction_features(df: pd.DataFrame, features: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Create interaction features between specified pairs
    
    Args:
        df: Input DataFrame
        features: List of (feature1, feature2) tuples for interaction
        
    Returns:
        pd.DataFrame: DataFrame with interaction features
    """
    df = df.copy()
    
    for feat1, feat2 in features:
        if feat1 in df.columns and feat2 in df.columns:
            interaction_name = f"{feat1}_x_{feat2}"
            df[interaction_name] = df[feat1] * df[feat2]
            logger.info(f"Created interaction feature: {interaction_name}")
    
    return df


if __name__ == "__main__":
    from data_loader import load_raw_data
    from data_cleaner import clean_data, split_data
    
    # Example usage
    df = load_raw_data()
    df_cleaned = clean_data(df)
    train_df, val_df, test_df = split_data(df_cleaned)
    
    # Prepare features and target
    target = 'churn'
    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]
    
    X_val = val_df.drop(target, axis=1)
    X_test = test_df.drop(target, axis=1)
    
    # Identify feature types
    cat_features, num_features = identify_feature_types(X_train)
    
    # Preprocess
    X_train_proc, X_val_proc, X_test_proc, preprocessor = preprocess_features(
        X_train, X_val, X_test, cat_features, num_features
    )
    
    print(f"Preprocessed shapes: Train {X_train_proc.shape}, Val {X_val_proc.shape}, Test {X_test_proc.shape}")
