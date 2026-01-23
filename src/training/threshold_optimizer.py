"""
Threshold Optimizer Module

Find optimal threshold for anomaly detection balancing TPR and FPR
"""
import numpy as np
from typing import Dict, Tuple, Optional, List
from loguru import logger
from sklearn.metrics import roc_curve

from src.training.evaluator import ModelEvaluator
from torch.utils.data import DataLoader


class ThresholdOptimizer:
    """Optimize threshold for anomaly detection"""
    
    def __init__(self, evaluator: ModelEvaluator):
        """
        Initialize threshold optimizer
        
        Args:
            evaluator: ModelEvaluator instance
        """
        self.evaluator = evaluator
    
    def find_optimal_threshold(
        self,
        dataloader: DataLoader,
        labels: List[int],
        method: str = "f1_maximize",
        target_fpr: float = 0.01
    ) -> Dict:
        """
        Find optimal threshold
        
        Args:
            dataloader: DataLoader with validation/test data
            labels: Ground truth labels (0=benign, 1=anomaly)
            method: Optimization method ("f1_maximize", "target_fpr", "roc_optimal")
            target_fpr: Target false positive rate (for target_fpr method)
        
        Returns:
            Dictionary with optimal threshold and metrics
        """
        # Get all predictions
        all_scores = []
        all_labels = []
        label_idx = 0
        
        import torch
        self.evaluator.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.evaluator.device)
                attention_mask = batch['attention_mask'].to(self.evaluator.device)
                
                outputs = self.evaluator.model(input_ids, attention_mask)
                scores = outputs['anomaly_score'].cpu().numpy()
                
                batch_size = len(scores)
                all_scores.extend(scores.tolist())
                batch_labels = labels[label_idx:label_idx + batch_size]
                all_labels.extend(batch_labels)
                label_idx += batch_size
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        if method == "f1_maximize":
            return self._maximize_f1(all_scores, all_labels)
        elif method == "target_fpr":
            return self._find_threshold_for_fpr(all_scores, all_labels, target_fpr)
        elif method == "roc_optimal":
            return self._find_roc_optimal(all_scores, all_labels)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _maximize_f1(self, scores: np.ndarray, labels: np.ndarray) -> Dict:
        """Find threshold that maximizes F1 score"""
        best_threshold = 0.5
        best_f1 = 0.0
        best_metrics = {}
        
        # Try thresholds from 0.1 to 0.9
        thresholds = np.arange(0.1, 1.0, 0.01)
        
        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)
            metrics = self._calculate_metrics_at_threshold(labels, predictions, scores, threshold)
            
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_threshold = threshold
                best_metrics = metrics
        
        logger.info(f"Optimal threshold (F1 maximize): {best_threshold:.3f}, F1: {best_f1:.4f}")
        
        return {
            'optimal_threshold': float(best_threshold),
            'method': 'f1_maximize',
            'metrics': best_metrics
        }
    
    def _find_threshold_for_fpr(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        target_fpr: float
    ) -> Dict:
        """Find threshold that achieves target FPR"""
        # Calculate ROC curve
        if len(np.unique(labels)) > 1:
            fpr, tpr, thresholds = roc_curve(labels, scores)
            
            # Find threshold closest to target FPR
            idx = np.argmin(np.abs(fpr - target_fpr))
            optimal_threshold = thresholds[idx] if idx < len(thresholds) else 0.5
            
            # Calculate metrics at this threshold
            predictions = (scores > optimal_threshold).astype(int)
            metrics = self._calculate_metrics_at_threshold(labels, predictions, scores, optimal_threshold)
            
            logger.info(f"Optimal threshold (FPR={target_fpr}): {optimal_threshold:.3f}")
            logger.info(f"Actual FPR: {metrics['fpr']:.4f}, TPR: {metrics['tpr']:.4f}")
            
            return {
                'optimal_threshold': float(optimal_threshold),
                'method': 'target_fpr',
                'target_fpr': target_fpr,
                'metrics': metrics
            }
        else:
            logger.warning("Only one class in labels, using default threshold")
            return {
                'optimal_threshold': 0.5,
                'method': 'target_fpr',
                'target_fpr': target_fpr,
                'metrics': {}
            }
    
    def _find_roc_optimal(self, scores: np.ndarray, labels: np.ndarray) -> Dict:
        """Find threshold using ROC optimal point (closest to top-left)"""
        if len(np.unique(labels)) > 1:
            fpr, tpr, thresholds = roc_curve(labels, scores)
            
            # Find point closest to (0, 1) - perfect classification
            distances = np.sqrt((fpr - 0) ** 2 + (tpr - 1) ** 2)
            idx = np.argmin(distances)
            optimal_threshold = thresholds[idx] if idx < len(thresholds) else 0.5
            
            # Calculate metrics
            predictions = (scores > optimal_threshold).astype(int)
            metrics = self._calculate_metrics_at_threshold(labels, predictions, scores, optimal_threshold)
            
            logger.info(f"Optimal threshold (ROC optimal): {optimal_threshold:.3f}")
            
            return {
                'optimal_threshold': float(optimal_threshold),
                'method': 'roc_optimal',
                'metrics': metrics
            }
        else:
            return {
                'optimal_threshold': 0.5,
                'method': 'roc_optimal',
                'metrics': {}
            }
    
    def _calculate_metrics_at_threshold(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        scores: np.ndarray,
        threshold: float
    ) -> Dict:
        """Calculate metrics at specific threshold"""
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'threshold': float(threshold),
            'tpr': float(tpr),
            'fpr': float(fpr),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'confusion_matrix': {
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            }
        }
