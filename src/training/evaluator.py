"""
Evaluation Metrics Module

Comprehensive evaluation metrics for anomaly detection model
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, roc_curve
from loguru import logger
import numpy as np

from src.model.anomaly_detector import AnomalyDetector
from src.model.scoring import AnomalyScorer


class ModelEvaluator:
    """Calculate comprehensive evaluation metrics"""
    
    def __init__(self, model: AnomalyDetector, device: str = None):
        """
        Initialize evaluator
        
        Args:
            model: AnomalyDetector model
            device: Device to run evaluation on
        """
        self.model = model
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()
    
    def evaluate(
        self,
        dataloader: DataLoader,
        threshold: float = 0.5,
        labels: Optional[List[int]] = None
    ) -> Dict:
        """
        Evaluate model on dataset
        
        Args:
            dataloader: DataLoader with test data
            threshold: Anomaly detection threshold
            labels: Optional ground truth labels (0=benign, 1=anomaly)
                   If None, assumes all are benign (for validation set)
        
        Returns:
            Dictionary with evaluation metrics
        """
        all_scores = []
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Get predictions
                outputs = self.model(input_ids, attention_mask)
                scores = outputs['anomaly_score'].cpu().numpy()
                
                # Convert to binary predictions
                predictions = (scores > threshold).astype(int)
                
                all_scores.extend(scores.tolist())
                all_predictions.extend(predictions.tolist())
                
                # Get labels if provided
                if labels is not None:
                    batch_labels = labels[len(all_labels):len(all_labels)+len(scores)]
                    all_labels.extend(batch_labels)
                else:
                    # Assume all are benign (0) if labels not provided
                    all_labels.extend([0] * len(scores))
        
        # Convert to numpy arrays
        all_scores = np.array(all_scores)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions, all_scores)
        
        return metrics
    
    def _calculate_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        scores: np.ndarray
    ) -> Dict:
        """Calculate all evaluation metrics"""
        # Confusion matrix components
        tp = np.sum((predictions == 1) & (labels == 1))  # True Positives
        fp = np.sum((predictions == 1) & (labels == 0))  # False Positives
        tn = np.sum((predictions == 0) & (labels == 0))  # True Negatives
        fn = np.sum((predictions == 0) & (labels == 1))  # False Negatives
        
        # Basic metrics
        total = len(labels)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        # TPR (True Positive Rate / Recall / Sensitivity)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # FPR (False Positive Rate)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall (same as TPR)
        recall = tpr
        
        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # ROC-AUC (if we have both classes)
        roc_auc = 0.0
        if len(np.unique(labels)) > 1:
            try:
                roc_auc = roc_auc_score(labels, scores)
            except ValueError:
                roc_auc = 0.0
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
        
        return {
            'accuracy': float(accuracy),
            'tpr': float(tpr),  # True Positive Rate / Recall
            'fpr': float(fpr),  # False Positive Rate
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'roc_auc': float(roc_auc),
            'specificity': float(specificity),
            'confusion_matrix': {
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            },
            'total_samples': int(total)
        }
    
    def evaluate_with_labels(
        self,
        dataloader: DataLoader,
        labels: List[int],
        threshold: float = 0.5
    ) -> Dict:
        """
        Evaluate with explicit labels
        
        Args:
            dataloader: DataLoader with test data
            labels: Ground truth labels (0=benign, 1=anomaly)
            threshold: Anomaly detection threshold
        
        Returns:
            Dictionary with evaluation metrics
        """
        return self.evaluate(dataloader, threshold, labels)
    
    def calculate_roc_curve(
        self,
        dataloader: DataLoader,
        labels: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ROC curve
        
        Args:
            dataloader: DataLoader with test data
            labels: Ground truth labels
        
        Returns:
            fpr, tpr, thresholds arrays
        """
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                scores = outputs['anomaly_score'].cpu().numpy()
                
                all_scores.extend(scores.tolist())
                batch_labels = labels[len(all_labels):len(all_labels)+len(scores)]
                all_labels.extend(batch_labels)
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        if len(np.unique(all_labels)) > 1:
            fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
            return fpr, tpr, thresholds
        else:
            logger.warning("Only one class in labels, cannot calculate ROC curve")
            return np.array([]), np.array([]), np.array([])
