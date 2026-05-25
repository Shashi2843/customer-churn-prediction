"""
04_evaluation_shap.py - Model Evaluation and SHAP Explainability
Run this script to evaluate models and generate SHAP explanations
"""
import logging
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run model evaluation and SHAP analysis"""
    logger.info("=" * 80)
    logger.info("MODEL EVALUATION & SHAP EXPLAINABILITY")
    logger.info("=" * 80)
    
    from src.data_loader import load_raw_data, download_dataset
    from src.data_cleaner import clean_data, split_data
    from src.feature_engineering import preprocess_features, identify_feature_types
    from src.model_trainer import ModelTrainer
    from src.evaluator import ModelEvaluator
    from src.explainability import ModelExplainer
    from config.config import MODELS_DIR
    
    # Load and prepare data
    logger.info("\n[Step 1] Loading and preparing data...")
    dataset_path = download_dataset()
    df = load_raw_data(dataset_path)
    df_cleaned = clean_data(df)
    train_df, val_df, test_df = split_data(df_cleaned)
    
    target = 'churn'
    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]
    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]
    
    cat_features, num_features = identify_feature_types(X_train)
    
    # Preprocess features
    logger.info("[Step 2] Preprocessing features...")
    X_train_proc, _, X_test_proc, preprocessor = preprocess_features(
        X_train, X_train, X_test, cat_features, num_features
    )
    
    # Load trained model
    logger.info("[Step 3] Loading trained model...")
    import joblib
    model_path = MODELS_DIR / "xgboost.pkl"
    
    if not model_path.exists():
        logger.error("Model not found. Run train_pipeline.py first.")
        return
    
    best_model = joblib.load(str(model_path))
    
    # Evaluate on test set
    logger.info("[Step 4] Evaluating model on test set...")
    trainer = ModelTrainer()
    trainer.models['xgboost'] = best_model
    test_metrics = trainer.evaluate_model('xgboost', X_test_proc, y_test)
    
    logger.info(f"\nTest Set Metrics:")
    logger.info(f"  ROC-AUC:  {test_metrics['roc_auc']:.4f}")
    logger.info(f"  F1-Score: {test_metrics['f1']:.4f}")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Plot evaluation charts
    logger.info("[Step 5] Generating evaluation visualizations...")
    evaluator = ModelEvaluator()
    
    y_pred = best_model.predict(X_test_proc)
    y_proba = best_model.predict_proba(X_test_proc)[:, 1]
    
    # Confusion matrix
    evaluator.plot_confusion_matrix(
        y_test.values,
        y_pred,
        model_name="XGBoost",
        save_path="confusion_matrix.png"
    )
    logger.info("Confusion matrix saved as 'confusion_matrix.png'")
    
    # ROC Curve
    evaluator.plot_roc_curve(
        y_test.values,
        y_proba,
        model_name="XGBoost",
        save_path="roc_curve.png"
    )
    logger.info("ROC curve saved as 'roc_curve.png'")
    
    # PR Curve
    evaluator.plot_pr_curve(
        y_test.values,
        y_proba,
        model_name="XGBoost",
        save_path="pr_curve.png"
    )
    logger.info("PR curve saved as 'pr_curve.png'")
    
    # SHAP Explainability
    logger.info("[Step 6] Generating SHAP explanations...")
    try:
        explainer = ModelExplainer(best_model, X_train_proc[:100])
        explainer.create_explainer(sample_size=100)
        explainer.explain_prediction(X_test_proc[:50])
        
        # Feature importance
        feature_importance = explainer.get_feature_importance()
        logger.info(f"\nTop 15 Important Features (SHAP):")
        logger.info(feature_importance.head(15))
        
        # Save feature importance
        feature_importance.to_csv('feature_importance_shap.csv')
        logger.info("Feature importance saved as 'feature_importance_shap.csv'")
        
    except Exception as e:
        logger.warning(f"SHAP explainability skipped: {e}")
    
    # Tree-based feature importance
    logger.info("[Step 7] Tree-based feature importance...")
    feature_importance_tree = best_model.feature_importances_
    feature_names = [f"feature_{i}" for i in range(len(feature_importance_tree))]
    
    evaluator.get_feature_importance_plot(
        feature_importance_tree,
        feature_names,
        top_n=15,
        model_name="XGBoost",
        save_path="feature_importance_tree.png"
    )
    logger.info("Tree-based feature importance saved as 'feature_importance_tree.png'")
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION & EXPLANATION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("\nGenerated files:")
    logger.info("  - confusion_matrix.png")
    logger.info("  - roc_curve.png")
    logger.info("  - pr_curve.png")
    logger.info("  - feature_importance_shap.csv")
    logger.info("  - feature_importance_tree.png")


if __name__ == "__main__":
    main()
