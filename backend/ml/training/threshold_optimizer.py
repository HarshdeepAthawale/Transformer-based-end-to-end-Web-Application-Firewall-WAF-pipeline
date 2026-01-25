"""
Threshold Optimizer Module

Optimize detection threshold for target false positive rate.
"""
import numpy as np
from typing import List, Dict, Tuple
from loguru import logger
from sklearn.metrics import roc_curve


class ThresholdOptimizer:
    """Optimize detection threshold"""
    
    @staticmethod
    def find_optimal_threshold(
        scores: List[float],
        labels: List[float],
        target_fpr: float = 0.01,
        min_tpr: float = 0.9
    ) -> Dict:
        """
        Find optimal threshold for target FPR
        
        Args:
            scores: Anomaly scores
            labels: True labels (0.0 = benign, 1.0 = malicious)
            target_fpr: Target false positive rate
            min_tpr: Minimum true positive rate to accept
        
        Returns:
            Dictionary with optimal threshold and metrics
        """
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # Find threshold closest to target FPR
        best_idx = np.argmin(np.abs(fpr - target_fpr))
        optimal_threshold = thresholds[best_idx]
        optimal_fpr = fpr[best_idx]
        optimal_tpr = tpr[best_idx]
        
        # Check if TPR meets minimum requirement
        if optimal_tpr < min_tpr:
            logger.warning(
                f"Optimal threshold TPR ({optimal_tpr:.4f}) is below minimum ({min_tpr:.4f})"
            )
            # Find threshold that meets min_tpr requirement
            valid_indices = np.where(tpr >= min_tpr)[0]
            if len(valid_indices) > 0:
                # Choose one with lowest FPR
                best_valid_idx = valid_indices[np.argmin(fpr[valid_indices])]
                optimal_threshold = thresholds[best_valid_idx]
                optimal_fpr = fpr[best_valid_idx]
                optimal_tpr = tpr[best_valid_idx]
        
        # Calculate metrics at optimal threshold
        predictions = (scores >= optimal_threshold).astype(int)
        labels_int = labels.astype(int)
        
        tp = np.sum((predictions == 1) & (labels_int == 1))
        fp = np.sum((predictions == 1) & (labels_int == 0))
        tn = np.sum((predictions == 0) & (labels_int == 0))
        fn = np.sum((predictions == 0) & (labels_int == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        result = {
            'optimal_threshold': float(optimal_threshold),
            'fpr': float(optimal_fpr),
            'tpr': float(optimal_tpr),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
        
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
        logger.info(f"FPR: {optimal_fpr:.4f} (target: {target_fpr:.4f})")
        logger.info(f"TPR: {optimal_tpr:.4f} (min: {min_tpr:.4f})")
        
        return result
    
    @staticmethod
    def evaluate_threshold_range(
        scores: List[float],
        labels: List[float],
        threshold_range: Tuple[float, float] = (0.0, 1.0),
        num_points: int = 100
    ) -> List[Dict]:
        """
        Evaluate multiple thresholds
        
        Args:
            scores: Anomaly scores
            labels: True labels
            threshold_range: (min, max) threshold range
            num_points: Number of threshold points to evaluate
        
        Returns:
            List of metrics dictionaries for each threshold
        """
        scores = np.array(scores)
        labels = np.array(labels)
        
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_points)
        results = []
        
        for threshold in thresholds:
            predictions = (scores >= threshold).astype(int)
            labels_int = labels.astype(int)
            
            tp = np.sum((predictions == 1) & (labels_int == 1))
            fp = np.sum((predictions == 1) & (labels_int == 0))
            tn = np.sum((predictions == 0) & (labels_int == 0))
            fn = np.sum((predictions == 0) & (labels_int == 1))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tpr
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            results.append({
                'threshold': float(threshold),
                'tpr': float(tpr),
                'fpr': float(fpr),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            })
        
        return results
