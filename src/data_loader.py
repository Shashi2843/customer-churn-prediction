"""
Data loading utilities for Customer Churn dataset
"""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_URL, DATASET_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_dataset() -> str:
    """
    Download IBM Telco Customer Churn dataset if not already present
    
    Returns:
        str: Path to the dataset file
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = RAW_DATA_DIR / DATASET_NAME
    
    if dataset_path.exists():
        logger.info(f"Dataset already exists at {dataset_path}")
        return str(dataset_path)
    
    try:
        logger.info(f"Downloading dataset from {DATASET_URL}")
        df = pd.read_csv(DATASET_URL)
        df.to_csv(dataset_path, index=False)
        logger.info(f"Dataset downloaded successfully to {dataset_path}")
        return str(dataset_path)
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise


def load_raw_data(file_path: str = None) -> pd.DataFrame:
    """
    Load raw dataset from CSV
    
    Args:
        file_path: Path to CSV file. If None, uses default dataset path
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if file_path is None:
        file_path = RAW_DATA_DIR / DATASET_NAME
    
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df


def load_processed_data(file_name: str = "processed_data.csv") -> pd.DataFrame:
    """
    Load processed dataset
    
    Args:
        file_name: Name of processed data file
        
    Returns:
        pd.DataFrame: Processed dataset
    """
    file_path = PROCESSED_DATA_DIR / file_name
    logger.info(f"Loading processed data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Processed data loaded. Shape: {df.shape}")
    return df


def save_processed_data(df: pd.DataFrame, file_name: str = "processed_data.csv"):
    """
    Save processed dataset
    
    Args:
        df: DataFrame to save
        file_name: Name of output file
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = PROCESSED_DATA_DIR / file_name
    df.to_csv(file_path, index=False)
    logger.info(f"Processed data saved to {file_path}")


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset
    
    Args:
        df: Input DataFrame
        
    Returns:
        dict: Dataset information
    """
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
    }


if __name__ == "__main__":
    # Example usage
    dataset_path = download_dataset()
    df = load_raw_data(dataset_path)
    print(df.head())
    print(f"\nDataset Info:")
    print(get_data_info(df))
