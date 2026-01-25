"""
Anomaly Scoring Module

Utilities for scoring requests and making anomaly detection decisions.
"""
import torch
from typing import Dict, Optional
from .anomaly_detector import AnomalyDetector


class AnomalyScorer:
    """Score requests for anomaly detection"""
    
    def __init__(
        self,
        model: AnomalyDetector,
        threshold: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize scorer
        
        Args:
            model: AnomalyDetector model instance
            threshold: Anomaly detection threshold (0.0 to 1.0)
            device: Device for inference
        """
        self.model = model
        self.threshold = threshold
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Score a single request or batch
        
        Args:
            input_ids: Token IDs [batch_size, seq_len] or [seq_len]
            attention_mask: Attention mask [batch_size, seq_len] or [seq_len]
        
        Returns:
            Dictionary with 'anomaly_score' and 'is_anomaly'
        """
        # Ensure tensors are on correct device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Add batch dimension if single sequence
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            anomaly_scores = outputs['anomaly_score']
        
        # Handle batch vs single
        if anomaly_scores.numel() == 1:
            score = anomaly_scores.item()
            is_anomaly = score >= self.threshold
        else:
            # Batch processing
            scores = anomaly_scores.cpu().tolist()
            is_anomaly = [s >= self.threshold for s in scores]
            
            # Return first if single item batch, else return list
            if len(scores) == 1:
                return {
                    'anomaly_score': scores[0],
                    'is_anomaly': is_anomaly[0],
                    'threshold': self.threshold
                }
            else:
                return {
                    'anomaly_scores': scores,
                    'is_anomaly': is_anomaly,
                    'threshold': self.threshold
                }
        
        return {
            'anomaly_score': score,
            'is_anomaly': is_anomaly,
            'threshold': self.threshold
        }
    
    def update_threshold(self, new_threshold: float):
        """Update detection threshold"""
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {new_threshold}")
        self.threshold = new_threshold
    
    def batch_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict:
        """
        Score a batch of requests
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Dictionary with batch results
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            anomaly_scores = outputs['anomaly_score']
        
        scores = anomaly_scores.cpu().tolist()
        is_anomaly = [s >= self.threshold for s in scores]
        
        return {
            'anomaly_scores': scores,
            'is_anomaly': is_anomaly,
            'threshold': self.threshold
        }
