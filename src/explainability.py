"""
SHAP-based model explainability module
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Generate SHAP-based explanations for models"""
    
    def __init__(self, model, X_train: np.ndarray):
        """
        Initialize explainer
        
        Args:
            model: Trained model
            X_train: Training data for creating background data
        """
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self, sample_size: int = 100):
        """
        Create SHAP explainer
        
        Args:
            sample_size: Size of background data sample
        """
        logger.info(f"Creating SHAP explainer with sample size {sample_size}")
        
        # Create background dataset
        background = shap.sample(self.X_train, sample_size)
        
        # Create explainer based on model type
        try:
            # Try TreeExplainer first (for tree-based models)
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Using TreeExplainer")
        except:
            try:
                # Fallback to KernelExplainer
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    background
                )
                logger.info("Using KernelExplainer")
            except Exception as e:
                logger.error(f"Error creating explainer: {e}")
                raise
    
    def explain_prediction(self, X: np.ndarray) -> np.ndarray:
        """
        Get SHAP values for predictions
        
        Args:
            X: Input data
            
        Returns:
            SHAP values
        """
        if self.explainer is None:
            self.create_explainer()
        
        logger.info(f"Computing SHAP values for {len(X)} samples")
        self.shap_values = self.explainer.shap_values(X)
        return self.shap_values
    
    def plot_summary(self, save_path: str = None):
        """
        Plot SHAP summary plot
        
        Args:
            save_path: Path to save the figure
        """
        if self.shap_values is None:
            raise ValueError("Run explain_prediction first")
        
        logger.info("Creating SHAP summary plot")
        
        fig = plt.figure(figsize=(12, 8))
        
        # Handle binary classification
        if isinstance(self.shap_values, list):
            shap.summary_plot(self.shap_values[1], self.X_train, show=False)
        else:
            shap.summary_plot(self.shap_values, self.X_train, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
        
        return fig
    
    def plot_dependence(
        self,
        feature_idx: int,
        feature_names: list = None,
        save_path: str = None
    ):
        """
        Plot SHAP dependence plot for a feature
        
        Args:
            feature_idx: Index of feature
            feature_names: List of feature names
            save_path: Path to save the figure
        """
        if self.shap_values is None:
            raise ValueError("Run explain_prediction first")
        
        logger.info(f"Creating SHAP dependence plot for feature {feature_idx}")
        
        fig = plt.figure(figsize=(10, 6))
        
        # Handle binary classification
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1]
        else:
            shap_vals = self.shap_values
        
        shap.dependence_plot(feature_idx, shap_vals, self.X_train, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dependence plot saved to {save_path}")
        
        return fig
    
    def plot_force(self, instance_idx: int):
        """
        Plot SHAP force plot for a single prediction
        
        Args:
            instance_idx: Index of instance to explain
        """
        if self.shap_values is None:
            raise ValueError("Run explain_prediction first")
        
        logger.info(f"Creating SHAP force plot for instance {instance_idx}")
        
        # Handle binary classification
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1]
            base_value = self.explainer.expected_value[1]
        else:
            shap_vals = self.shap_values
            base_value = self.explainer.expected_value
        
        shap.force_plot(
            base_value,
            shap_vals[instance_idx],
            self.X_train[instance_idx],
            show=True
        )
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get mean absolute SHAP values (feature importance)
        
        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            raise ValueError("Run explain_prediction first")
        
        logger.info("Computing feature importance from SHAP values")
        
        # Handle binary classification
        if isinstance(self.shap_values, list):
            shap_vals = np.abs(self.shap_values[1])
        else:
            shap_vals = np.abs(self.shap_values)
        
        importance = np.mean(shap_vals, axis=0)
        
        importance_df = pd.DataFrame({
            'feature_importance': importance
        }).sort_values('feature_importance', ascending=False)
        
        return importance_df
    
    def explain_instance(self, X_instance: np.ndarray, feature_names: list = None) -> dict:
        """
        Get detailed explanation for a single instance
        
        Args:
            X_instance: Single instance to explain
            feature_names: List of feature names
            
        Returns:
            Dictionary with explanation details
        """
        if len(X_instance.shape) == 1:
            X_instance = X_instance.reshape(1, -1)
        
        shap_values = self.explain_prediction(X_instance)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]
        
        explanation = {
            'prediction': self.model.predict(X_instance)[0],
            'prediction_probability': self.model.predict_proba(X_instance)[0],
            'shap_values': shap_vals,
            'feature_contributions': pd.DataFrame({
                'feature': feature_names or [f'feature_{i}' for i in range(len(shap_vals))],
                'value': X_instance[0],
                'shap_value': shap_vals
            }).sort_values('shap_value', key=abs, ascending=False)
        }
        
        return explanation


if __name__ == "__main__":
    from model_trainer import ModelTrainer
    from feature_engineering import preprocess_features, identify_feature_types
    from data_loader import load_raw_data
    from data_cleaner import clean_data, split_data
    
    # Example usage
    df = load_raw_data()
    df_cleaned = clean_data(df)
    train_df, _, test_df = split_data(df_cleaned)
    
    target = 'churn'
    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]
    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]
    
    cat_features, num_features = identify_feature_types(X_train)
    X_train_proc, _, X_test_proc, _ = preprocess_features(X_train, X_train, X_test, cat_features, num_features)
    
    # Train model
    trainer = ModelTrainer()
    trainer.train_xgboost(X_train_proc, y_train)
    best_model = trainer.models['xgboost']
    
    # Explain
    explainer = ModelExplainer(best_model, X_train_proc)
    explainer.explain_prediction(X_test_proc)
    importance = explainer.get_feature_importance()
    print(importance.head())
