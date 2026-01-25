"""
Model Evaluator Module

Evaluation metrics and utilities for model performance assessment.
"""
import torch
import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from loguru import logger
from torch.utils.data import DataLoader

from backend.ml.model.anomaly_detector import AnomalyDetector
from backend.ml.model.scoring import AnomalyScorer


class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(
        self,
        model: AnomalyDetector,
        tokenizer,
        device: str = "cpu"
    ):
        """
        Initialize evaluator
        
        Args:
            model: AnomalyDetector model
            tokenizer: HTTPTokenizer instance
            device: Device for evaluation
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
    
    def evaluate_on_loader(
        self,
        data_loader: DataLoader,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate model on a DataLoader
        
        Args:
            data_loader: DataLoader with test data
            threshold: Detection threshold
        
        Returns:
            Dictionary with evaluation metrics
        """
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Get predictions
                outputs = self.model(input_ids, attention_mask)
                scores = outputs['anomaly_score'].cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_scores.extend(scores)
                all_labels.extend(labels_np)
        
        # Convert to numpy arrays
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        return self._calculate_metrics(all_scores, all_labels, threshold)
    
    def evaluate_on_texts(
        self,
        texts: List[str],
        labels: List[float],
        threshold: float = 0.5,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate model on list of texts
        
        Args:
            texts: List of normalized request strings
            labels: List of labels (0.0 for benign, 1.0 for malicious)
            threshold: Detection threshold
            batch_size: Batch size for evaluation
        
        Returns:
            Dictionary with evaluation metrics
        """
        from backend.ml.tokenization.sequence_prep import SequencePreparator
        
        preparator = SequencePreparator(self.tokenizer)
        all_scores = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Prepare batch
            batch_data = preparator.prepare_batch(
                batch_texts,
                return_tensors="pt"
            )
            
            input_ids = batch_data['input_ids'].to(self.device)
            attention_mask = batch_data['attention_mask'].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                scores = outputs['anomaly_score'].cpu().numpy()
                all_scores.extend(scores)
        
        all_scores = np.array(all_scores)
        all_labels = np.array(labels)
        
        return self._calculate_metrics(all_scores, all_labels, threshold)
    
    def _calculate_metrics(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: float
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        # Binary predictions
        predictions = (scores >= threshold).astype(int)
        labels_int = labels.astype(int)
        
        # True Positives, False Positives, True Negatives, False Negatives
        tp = np.sum((predictions == 1) & (labels_int == 1))
        fp = np.sum((predictions == 1) & (labels_int == 0))
        tn = np.sum((predictions == 0) & (labels_int == 0))
        fn = np.sum((predictions == 0) & (labels_int == 1))
        
        # Basic metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Recall)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Accuracy
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        
        # ROC-AUC (if we have both classes)
        try:
            if len(np.unique(labels_int)) > 1:
                roc_auc = roc_auc_score(labels_int, scores)
            else:
                roc_auc = 0.0
        except:
            roc_auc = 0.0
        
        # Average Precision
        try:
            if len(np.unique(labels_int)) > 1:
                avg_precision = average_precision_score(labels_int, scores)
            else:
                avg_precision = 0.0
        except:
            avg_precision = 0.0
        
        metrics = {
            'threshold': threshold,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'tpr': float(tpr),
            'fpr': float(fpr),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'avg_precision': float(avg_precision)
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Print evaluation metrics in a readable format"""
        logger.info("=" * 50)
        logger.info("Evaluation Metrics")
        logger.info("=" * 50)
        logger.info(f"Threshold: {metrics['threshold']:.3f}")
        logger.info(f"TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}")
        logger.info(f"TPR (Recall): {metrics['tpr']:.4f}")
        logger.info(f"FPR: {metrics['fpr']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Average Precision: {metrics['avg_precision']:.4f}")
        logger.info("=" * 50)
