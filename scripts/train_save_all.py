"""Train all models and save them to the `models/` directory.

Usage: run with the project's venv python:
    venv\Scripts\python.exe scripts\train_save_all.py
"""
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` can be imported when running as script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_raw_data
from src.data_cleaner import clean_data, split_data
from src.feature_engineering import identify_feature_types, preprocess_features
from src.model_trainer import ModelTrainer
from config.config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw data...")
    df = load_raw_data()

    logger.info("Cleaning data...")
    df_clean = clean_data(df)

    logger.info("Splitting data...")
    train_df, val_df, test_df = split_data(df_clean)

    target = 'churn'
    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]
    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]

    logger.info("Identifying feature types...")
    cat_features, num_features = identify_feature_types(X_train)

    logger.info("Preprocessing features (fit)...")
    X_train_proc, X_val_proc, X_test_proc, preprocessor = preprocess_features(
        X_train, X_train, X_test, cat_features, num_features, fit_preprocessor=True
    )

    # Save preprocessor
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    from src.feature_engineering import save_preprocessor
    save_preprocessor(preprocessor, str(preprocessor_path))
    logger.info(f"Preprocessor saved to {preprocessor_path}")

    # Train models and save all
    trainer = ModelTrainer()
    trainer.train_all_models(X_train_proc, y_train)
    trainer.evaluate_all_models(X_test_proc, y_test)

    logger.info("Saving all trained models to models/ ...")
    trainer.save_all_models(str(MODELS_DIR))

    logger.info("Saved models: %s", list(p.name for p in MODELS_DIR.iterdir()))


if __name__ == '__main__':
    main()
