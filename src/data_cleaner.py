"""
Data cleaning and preprocessing module
"""
import logging
import pandas as pd
import numpy as np
from typing import Tuple, List

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the dataset
    
    Args:
        df: Raw input DataFrame
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df = df.copy()
    
    logger.info("Starting data cleaning process")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Convert churn to binary
    if 'churn' in df.columns:
        df['churn'] = (df['churn'] == 'Yes').astype(int)
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Fix data types
    df = fix_data_types(df)
    
    logger.info("Data cleaning completed")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    logger.info(f"Missing values before handling:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    
    # Handle TotalCharges - convert to numeric and fill missing
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Fill any other numeric missing values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical missing values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
    
    logger.info(f"Missing values after handling:\n{df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().sum().sum() > 0 else 'None'}")
    
    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix and standardize data types
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with correct data types
    """
    # Convert numeric string columns
    for col in df.columns:
        if col in ['totalcharges']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert tenure and charges to numeric
        if col in ['tenure', 'monthlycharges']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def split_data(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify_col: str = 'churn'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets with stratification
    
    Args:
        df: Input DataFrame
        train_size: Proportion for training
        val_size: Proportion for validation
        test_size: Proportion for testing
        random_state: Random seed
        stratify_col: Column to stratify on
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    # First split: train + temp (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        random_state=random_state,
        stratify=df[stratify_col] if stratify_col in df.columns else None
    )
    
    # Second split: val and test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        random_state=random_state,
        stratify=temp_df[stratify_col] if stratify_col in temp_df.columns else None
    )
    
    logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def remove_outliers(
    df: pd.DataFrame,
    columns: List[str] = None,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers using IQR or Z-score method
    
    Args:
        df: Input DataFrame
        columns: Columns to check for outliers
        method: 'iqr' or 'zscore'
        threshold: IQR multiplier or Z-score threshold
        
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    initial_rows = len(df)
    
    if method == 'iqr':
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[columns].select_dtypes(include=[np.number])))
        df = df[(z_scores < threshold).all(axis=1)]
    
    logger.info(f"Removed {initial_rows - len(df)} outliers using {method} method")
    
    return df


if __name__ == "__main__":
    from data_loader import load_raw_data, save_processed_data
    
    # Example usage
    df = load_raw_data()
    df_cleaned = clean_data(df)
    train_df, val_df, test_df = split_data(df_cleaned)
    
    print(f"Cleaned data shape: {df_cleaned.shape}")
    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
