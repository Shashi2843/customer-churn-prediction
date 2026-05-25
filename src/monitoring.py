"""
Data drift monitoring using Evidently AI
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, RegressionPreset
    from evidently.metrics import (
        DataDriftTable,
        DataQualityTable,
        TargetDriftMetricTable
    )
except ImportError:
    logging.warning("Evidently not installed. Monitoring will be limited.")

logger = logging.getLogger(__name__)


class DataDriftMonitor:
    """Monitor data drift using Evidently AI"""
    
    def __init__(self, reference_data: pd.DataFrame):
        """
        Initialize monitor with reference data
        
        Args:
            reference_data: Reference dataset for drift detection
        """
        self.reference_data = reference_data
        self.drift_reports = []
    
    def check_data_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.05
    ) -> Dict:
        """
        Check for data drift in current data
        
        Args:
            current_data: Current dataset to check for drift
            threshold: Drift detection threshold
            
        Returns:
            Dictionary with drift information
        """
        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=self.reference_data,
                current_data=current_data
            )
            
            logger.info("Data drift check completed")
            
            result = {
                'drift_detected': report.as_dict()['metrics'][0]['result']['is_drift'],
                'report': report
            }
            
            self.drift_reports.append(result)
            return result
        
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
            return {'drift_detected': False, 'error': str(e)}
    
    def check_data_quality(
        self,
        data: pd.DataFrame
    ) -> Dict:
        """
        Check data quality
        
        Args:
            data: Dataset to check
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {
            'missing_values': data.isnull().sum().to_dict(),
            'duplicates': data.duplicated().sum(),
            'shape': data.shape,
            'data_types': data.dtypes.to_dict(),
        }
        
        logger.info("Data quality check completed")
        return quality_metrics
    
    def generate_drift_report(self, save_path: str = None) -> str:
        """
        Generate HTML report for data drift
        
        Args:
            save_path: Path to save HTML report
            
        Returns:
            Path to saved report
        """
        if not self.drift_reports:
            logger.warning("No drift reports available")
            return None
        
        try:
            latest_report = self.drift_reports[-1]['report']
            
            if save_path is None:
                save_path = f"drift_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            latest_report.save_html(save_path)
            logger.info(f"Drift report saved to {save_path}")
            
            return save_path
        
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None


class ModelPerformanceMonitor:
    """Monitor model performance over time"""
    
    def __init__(self):
        self.performance_history = []
    
    def log_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        timestamp: str = None
    ):
        """
        Log predictions for monitoring
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            timestamp: Timestamp of predictions
        """
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        if timestamp is None:
            timestamp = pd.Timestamp.now().isoformat()
        
        metrics = {
            'timestamp': timestamp,
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
        }
        
        self.performance_history.append(metrics)
        logger.info(f"Performance logged - Accuracy: {metrics['accuracy']:.3f}")
    
    def get_performance_trend(self) -> pd.DataFrame:
        """
        Get performance trend over time
        
        Returns:
            DataFrame with performance metrics over time
        """
        if not self.performance_history:
            logger.warning("No performance history available")
            return pd.DataFrame()
        
        return pd.DataFrame(self.performance_history)
    
    def check_performance_degradation(
        self,
        threshold: float = 0.05
    ) -> bool:
        """
        Check if model performance has degraded
        
        Args:
            threshold: Performance degradation threshold
            
        Returns:
            True if performance has degraded, False otherwise
        """
        if len(self.performance_history) < 2:
            return False
        
        metrics_df = self.get_performance_trend()
        initial_accuracy = metrics_df.iloc[0]['accuracy']
        current_accuracy = metrics_df.iloc[-1]['accuracy']
        
        degradation = initial_accuracy - current_accuracy
        
        if degradation > threshold:
            logger.warning(
                f"Performance degradation detected: {degradation:.4f} "
                f"(threshold: {threshold})"
            )
            return True
        
        return False


if __name__ == "__main__":
    from data_loader import load_raw_data
    
    # Example usage
    df = load_raw_data()
    
    # Create train-test split
    train_df = df.iloc[:5000]
    test_df = df.iloc[5000:]
    
    # Initialize monitor
    monitor = DataDriftMonitor(train_df)
    
    # Check drift
    drift_result = monitor.check_data_drift(test_df)
    print(f"Drift detected: {drift_result['drift_detected']}")
    
    # Check quality
    quality = monitor.check_data_quality(test_df)
    print(f"Data quality: {quality}")
