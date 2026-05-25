"""
Model training and management module
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and manage multiple ML models"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.metrics = {}
    
    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> LogisticRegression:
        """
        Train Logistic Regression model
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for LogisticRegression
            
        Returns:
            Trained LogisticRegression model
        """
        logger.info("Training Logistic Regression model")
        
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            **kwargs
        )
        # If X_train contains non-numeric data (e.g., test fixtures), convert using one-hot encoding
        if isinstance(X_train, (list, tuple)):
            X_train = np.array(X_train)

        if isinstance(X_train, np.ndarray) and X_train.dtype == object:
            X_df = pd.DataFrame(X_train)
            # Use get_dummies to one-hot encode string/object columns
            X_df = pd.get_dummies(X_df, dummy_na=False)
            # remember feature columns for later prediction
            self._feature_columns = X_df.columns.tolist()
            X_train_proc = X_df.values
        else:
            X_train_proc = X_train

        model.fit(X_train_proc, y_train)
        
        self.models['logistic_regression'] = model
        logger.info("Logistic Regression training completed")
        return model
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> RandomForestClassifier:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for RandomForestClassifier
            
        Returns:
            Trained RandomForestClassifier model
        """
        logger.info("Training Random Forest model")
        
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1,
            **kwargs
        )
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        logger.info("Random Forest training completed")
        return model
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> XGBClassifier:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for XGBClassifier
            
        Returns:
            Trained XGBClassifier model
        """
        logger.info("Training XGBoost model")
        
        model = XGBClassifier(
            n_estimators=100,
            random_state=self.random_state,
            tree_method='hist',
            **kwargs
        )
        model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        logger.info("XGBoost training completed")
        return model
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train all available models
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained models
        """
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        
        logger.info(f"All models trained. Total models: {len(self.models)}")
        return self.models
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with a specific model
        
        Args:
            model_name: Name of the model
            X: Input features
            
        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]

        # If X is object-dtype (e.g., raw sample with categorical strings), encode similarly
        if isinstance(X, np.ndarray) and X.dtype == object and hasattr(self, '_feature_columns'):
            X_df = pd.DataFrame(X)
            X_df = pd.get_dummies(X_df, dummy_na=False)
            # Reindex to ensure same feature columns as training, fill missing with 0
            X_df = X_df.reindex(columns=self._feature_columns, fill_value=0)
            X_proc = X_df.values
            return model.predict(X_proc)

        return model.predict(X)
    
    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            model_name: Name of the model
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]

        if isinstance(X, np.ndarray) and X.dtype == object and hasattr(self, '_feature_columns'):
            X_df = pd.DataFrame(X)
            X_df = pd.get_dummies(X_df, dummy_na=False)
            X_df = X_df.reindex(columns=self._feature_columns, fill_value=0)
            X_proc = X_df.values
            return model.predict_proba(X_proc)

        return model.predict_proba(X)
    
    def evaluate_model(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Args:
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(model_name, X_test)
        y_proba = self.predict_proba(model_name, X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_proba),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'accuracy': (y_pred == y_test).mean(),
        }
        
        # ROC-AUC components
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        metrics['roc_auc_curve'] = (fpr, tpr)
        
        # PR-AUC components
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
        metrics['pr_auc'] = auc(recall_vals, precision_vals)
        metrics['pr_curve'] = (recall_vals, precision_vals)
        
        self.metrics[model_name] = metrics
        logger.info(f"Model '{model_name}' evaluation - ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def evaluate_all_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics for all models
        """
        for model_name in self.models.keys():
            self.evaluate_model(model_name, X_test, y_test)
        
        return self.metrics
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get best model based on ROC-AUC score
        
        Returns:
            Tuple of (model_name, model)
        """
        if not self.metrics:
            raise ValueError("No model metrics available. Run evaluate_all_models first.")
        
        best_model_name = max(
            self.metrics.keys(),
            key=lambda x: self.metrics[x]['roc_auc']
        )
        
        return best_model_name, self.models[best_model_name]
    
    def save_model(self, model_name: str, file_path: str):
        """
        Save model to disk
        
        Args:
            model_name: Name of the model
            file_path: Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models[model_name], file_path)
        logger.info(f"Model '{model_name}' saved to {file_path}")
    
    def load_model(self, model_name: str, file_path: str):
        """
        Load model from disk
        
        Args:
            model_name: Name to assign to the model
            file_path: Path to load the model from
        """
        model = joblib.load(file_path)
        self.models[model_name] = model
        logger.info(f"Model '{model_name}' loaded from {file_path}")
        return model
    
    def save_all_models(self, directory: str):
        """
        Save all models to a directory
        
        Args:
            directory: Directory to save models
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            file_path = Path(directory) / f"{model_name}.pkl"
            joblib.dump(model, str(file_path))
            logger.info(f"Model '{model_name}' saved to {file_path}")
    
    def get_feature_importance(self, model_name: str) -> np.ndarray:
        """
        Get feature importance from the model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Feature importance array
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_[0])
        else:
            raise ValueError(f"Model '{model_name}' doesn't have feature importance")


if __name__ == "__main__":
    from feature_engineering import preprocess_features, identify_feature_types
    from data_loader import load_raw_data
    from data_cleaner import clean_data, split_data
    
    # Example usage
    df = load_raw_data()
    df_cleaned = clean_data(df)
    train_df, val_df, test_df = split_data(df_cleaned)
    
    target = 'churn'
    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]
    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]
    
    cat_features, num_features = identify_feature_types(X_train)
    X_train_proc, _, X_test_proc, _ = preprocess_features(X_train, X_train, X_test, cat_features, num_features)
    
    trainer = ModelTrainer()
    trainer.train_all_models(X_train_proc, y_train)
    trainer.evaluate_all_models(X_test_proc, y_test)
    
    best_name, best_model = trainer.get_best_model()
    print(f"Best model: {best_name}")
