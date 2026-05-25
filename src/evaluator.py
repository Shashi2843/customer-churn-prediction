"""
Model evaluation and metrics module
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, roc_auc_score
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate and compare model performance"""
    
    def __init__(self):
        self.metrics_history = {}
    
    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Get confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Get detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as dictionary
        """
        return classification_report(y_true, y_pred, output_dict=True)
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "Model",
        save_path: str = None
    ) -> Tuple[plt.Figure, float]:
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the figure
            
        Returns:
            Tuple of (figure, roc_auc_score)
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        return fig, roc_auc
    
    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "Model",
        save_path: str = None
    ) -> Tuple[plt.Figure, float]:
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the figure
            
        Returns:
            Tuple of (figure, pr_auc_score)
        """
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})', linewidth=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR curve saved to {save_path}")
        
        return fig, pr_auc
    
    def compare_models(
        self,
        models_metrics: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Compare metrics across multiple models
        
        Args:
            models_metrics: Dictionary of model names to metrics
            
        Returns:
            DataFrame with comparison
        """
        comparison_df = pd.DataFrame(models_metrics).T
        
        logger.info(f"Model Comparison:\n{comparison_df}")
        
        return comparison_df
    
    def plot_model_comparison(
        self,
        models_metrics: Dict[str, Dict[str, float]],
        metrics_to_compare: list = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot comparison of models
        
        Args:
            models_metrics: Dictionary of model names to metrics
            metrics_to_compare: List of metrics to compare
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if metrics_to_compare is None:
            metrics_to_compare = ['roc_auc', 'f1', 'precision', 'recall']
        
        comparison_df = self.compare_models(models_metrics)
        comparison_df = comparison_df[[m for m in metrics_to_compare if m in comparison_df.columns]]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_df.plot(kind='bar', ax=ax)
        ax.set_title('Model Comparison')
        ax.set_ylabel('Score')
        ax.set_xlabel('Metric')
        ax.legend(title='Model')
        plt.xticks(rotation=45)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        return fig
    
    def get_feature_importance_plot(
        self,
        feature_importance: np.ndarray,
        feature_names: list,
        top_n: int = 15,
        model_name: str = "Model",
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot feature importance
        
        Args:
            feature_importance: Feature importance scores
            feature_names: List of feature names
            top_n: Number of top features to show
            model_name: Name of the model
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Create dataframe and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(importance_df['feature'], importance_df['importance'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance - {model_name}')
        ax.invert_yaxis()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig


if __name__ == "__main__":
    import numpy as np
    
    # Example usage
    evaluator = ModelEvaluator()
    
    # Generate sample data
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_proba = np.random.rand(100)
    
    # Get metrics
    cm = evaluator.get_confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    # Plot ROC curve
    fig, roc_auc = evaluator.plot_roc_curve(y_true, y_proba)
    print(f"ROC-AUC: {roc_auc:.3f}")
