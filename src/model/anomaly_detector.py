"""
Anomaly Detector Model Module

Transformer-based anomaly detection model using DistilBERT
"""
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
from typing import Dict, Optional
from loguru import logger


class AnomalyDetector(nn.Module):
    """Transformer-based anomaly detection model"""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        dropout: float = 0.1,
        max_length: int = 512
    ):
        super().__init__()
        
        # Use DistilBERT configuration
        config = DistilBertConfig(
            vocab_size=vocab_size,
            dim=hidden_size,
            n_layers=num_layers,
            n_heads=num_heads,
            max_position_embeddings=max_length,
            dropout=dropout,
            attention_dropout=dropout
        )
        
        # Transformer encoder
        self.transformer = DistilBertModel(config)
        
        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # Output anomaly probability [0, 1]
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"AnomalyDetector initialized: vocab_size={vocab_size}, hidden_size={hidden_size}, "
                   f"num_layers={num_layers}, num_heads={num_heads}, max_length={max_length}")
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.anomaly_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs tensor [batch_size, seq_length]
            attention_mask: Attention mask tensor [batch_size, seq_length]
        
        Returns:
            Dictionary with 'anomaly_score' and 'embeddings'
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Get anomaly score
        anomaly_score = self.anomaly_head(cls_embedding).squeeze(-1)
        
        return {
            'anomaly_score': anomaly_score,
            'embeddings': cls_embedding
        }
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> float:
        """
        Predict anomaly score for single input
        
        Args:
            input_ids: Token IDs tensor [1, seq_length]
            attention_mask: Attention mask tensor [1, seq_length]
        
        Returns:
            Anomaly score (float between 0 and 1)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs['anomaly_score'].item()
    
    def predict_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict anomaly scores for batch
        
        Args:
            input_ids: Token IDs tensor [batch_size, seq_length]
            attention_mask: Attention mask tensor [batch_size, seq_length]
        
        Returns:
            Anomaly scores tensor [batch_size]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs['anomaly_score']
