"""
Anomaly Scoring Module

Score requests for anomaly detection
"""
import torch
from typing import Dict, List
import numpy as np
from loguru import logger
from .anomaly_detector import AnomalyDetector


class AnomalyScorer:
    """Score requests for anomaly detection"""
    
    def __init__(
        self,
        model: AnomalyDetector,
        threshold: float = 0.5,
        device: str = None
    ):
        """
        Initialize anomaly scorer
        
        Args:
            model: Trained AnomalyDetector model
            threshold: Threshold for anomaly detection (default: 0.5)
            device: Device to run inference on (default: auto-detect)
        """
        self.model = model
        self.model.eval()
        self.threshold = threshold
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        logger.info(f"AnomalyScorer initialized with threshold={threshold}, device={self.device}")
    
    def score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict:
        """
        Score single request for anomaly
        
        Args:
            input_ids: Token IDs tensor [1, seq_length] or [seq_length]
            attention_mask: Attention mask tensor [1, seq_length] or [seq_length]
        
        Returns:
            Dictionary with anomaly_score, is_anomaly, and threshold
        """
        # Ensure tensors are on correct device and have batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            anomaly_score = outputs['anomaly_score'].item()
        
        is_anomaly = anomaly_score > self.threshold
        
        return {
            'anomaly_score': float(anomaly_score),
            'is_anomaly': bool(is_anomaly),
            'threshold': self.threshold
        }
    
    def score_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict:
        """
        Score batch of requests
        
        Args:
            input_ids: Token IDs tensor [batch_size, seq_length]
            attention_mask: Attention mask tensor [batch_size, seq_length]
        
        Returns:
            Dictionary with anomaly_scores, is_anomaly, and threshold
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            anomaly_scores = outputs['anomaly_score'].cpu().numpy()
        
        is_anomaly = anomaly_scores > self.threshold
        
        return {
            'anomaly_scores': anomaly_scores.tolist(),
            'is_anomaly': is_anomaly.tolist(),
            'threshold': self.threshold
        }
    
    def set_threshold(self, threshold: float):
        """Update anomaly detection threshold"""
        self.threshold = threshold
        logger.info(f"Threshold updated to {threshold}")
