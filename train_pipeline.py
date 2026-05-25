"""
Main training pipeline for Customer Churn prediction
"""
import logging
import json
from pathlib import Path

from src.data_loader import load_raw_data, save_processed_data, download_dataset
from src.data_cleaner import clean_data, split_data
from src.feature_engineering import (
    preprocess_features,
    identify_feature_types,
    save_preprocessor,
    get_feature_names
)
from src.model_trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.explainability import ModelExplainer
from config.config import MODELS_DIR, PROCESSED_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Execute the complete ML pipeline"""
    
    logger.info("=" * 80)
    logger.info("Starting Customer Churn Prediction Pipeline")
    logger.info("=" * 80)
    
    # Step 1: Download and load data
    logger.info("\n[Step 1] Loading dataset...")
    dataset_path = download_dataset()
    df = load_raw_data(dataset_path)
    logger.info(f"Dataset loaded: {df.shape}")
    
    # Step 2: Data cleaning
    logger.info("\n[Step 2] Cleaning data...")
    df_cleaned = clean_data(df)
    logger.info(f"Cleaned data shape: {df_cleaned.shape}")
    
    # Step 3: Data splitting
    logger.info("\n[Step 3] Splitting data...")
    train_df, val_df, test_df = split_data(
        df_cleaned,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15
    )
    
    # Save cleaned data
    save_processed_data(df_cleaned, "cleaned_data.csv")
    
    # Step 4: Feature engineering
    logger.info("\n[Step 4] Feature engineering...")
    target = 'churn'
    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]
    X_val = val_df.drop(target, axis=1)
    y_val = val_df[target]
    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]
    
    cat_features, num_features = identify_feature_types(X_train)
    logger.info(f"Categorical features: {len(cat_features)}, Numerical features: {len(num_features)}")
    
    # Preprocess features
    X_train_proc, X_val_proc, X_test_proc, preprocessor = preprocess_features(
        X_train, X_val, X_test, cat_features, num_features, fit_preprocessor=True
    )
    
    # Save preprocessor
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_preprocessor(preprocessor, str(MODELS_DIR / "preprocessor.pkl"))
    
    # Step 5: Model training
    logger.info("\n[Step 5] Training models...")
    trainer = ModelTrainer()
    trainer.train_all_models(X_train_proc, y_train)
    
    # Step 6: Model evaluation
    logger.info("\n[Step 6] Evaluating models...")
    evaluator = ModelEvaluator()
    all_metrics = trainer.evaluate_all_models(X_test_proc, y_test)
    
    # Display comparison
    comparison = evaluator.compare_models(all_metrics)
    logger.info(f"\nModel Comparison:\n{comparison}")
    
    # Step 7: Get best model
    logger.info("\n[Step 7] Selecting best model...")
    best_model_name, best_model = trainer.get_best_model()
    logger.info(f"Best model: {best_model_name}")
    
    # Save best model
    trainer.save_model(best_model_name, str(MODELS_DIR / f"{best_model_name}.pkl"))
    
    # Step 8: SHAP Explainability
    logger.info("\n[Step 8] Generating SHAP explanations...")
    explainer = ModelExplainer(best_model, X_train_proc)
    explainer.explain_prediction(X_test_proc[:100])  # Sample for efficiency
    
    # Get feature importance
    feature_importance = explainer.get_feature_importance()
    logger.info(f"Top 10 important features:\n{feature_importance.head(10)}")
    
    # Step 9: Save metrics
    logger.info("\n[Step 9] Saving metrics...")
    best_metrics = all_metrics[best_model_name]
    
    # Remove non-serializable objects
    best_metrics_clean = {
        k: v for k, v in best_metrics.items()
        if k not in ['roc_auc_curve', 'pr_curve']
    }
    
    with open(MODELS_DIR / "metrics.json", 'w') as f:
        json.dump(best_metrics_clean, f, indent=4)
    
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 80)
    logger.info(f"""
    Summary:
    - Best Model: {best_model_name}
    - ROC-AUC: {best_metrics['roc_auc']:.4f}
    - F1-Score: {best_metrics['f1']:.4f}
    - Precision: {best_metrics['precision']:.4f}
    - Recall: {best_metrics['recall']:.4f}
    
    Models saved in: {MODELS_DIR}
    """)


if __name__ == "__main__":
    main()
