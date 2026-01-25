"""
Anomaly Detection Model

DistilBERT-based transformer model for HTTP request anomaly detection.
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
        """
        Initialize anomaly detection model
        
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_length: Maximum sequence length
        """
        super().__init__()
        
        # DistilBERT configuration
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
        # 3-layer MLP: 768 -> 384 -> 192 -> 1
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
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize anomaly head
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
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Dictionary with 'anomaly_score' and 'hidden_states'
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        # DistilBERT returns last_hidden_state: [batch_size, seq_len, hidden_size]
        cls_representation = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Get anomaly score
        anomaly_score = self.anomaly_head(cls_representation)  # [batch_size, 1]
        anomaly_score = anomaly_score.squeeze(-1)  # [batch_size]
        
        return {
            'anomaly_score': anomaly_score,
            'hidden_states': outputs.last_hidden_state
        }
    
    def save_checkpoint(
        self,
        path: str,
        epoch: Optional[int] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        vocab_size: Optional[int] = None
    ):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'vocab_size': vocab_size or self.transformer.config.vocab_size,
            'hidden_size': self.transformer.config.dim,
            'num_layers': self.transformer.config.n_layers,
            'num_heads': self.transformer.config.n_heads,
            'max_length': self.transformer.config.max_position_embeddings,
            'dropout': self.transformer.config.dropout
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if train_loss is not None:
            checkpoint['train_loss'] = train_loss
        if val_loss is not None:
            checkpoint['val_loss'] = val_loss
        
        torch.save(checkpoint, path)
        logger.info(f"Model checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        device: str = "cpu"
    ) -> 'AnomalyDetector':
        """
        Load model from checkpoint
        
        Args:
            path: Path to checkpoint file
            device: Device to load model on
        
        Returns:
            Loaded AnomalyDetector instance
        """
        checkpoint = torch.load(path, map_location=device)
        
        vocab_size = checkpoint.get('vocab_size', 10000)
        hidden_size = checkpoint.get('hidden_size', 768)
        num_layers = checkpoint.get('num_layers', 6)
        num_heads = checkpoint.get('num_heads', 12)
        max_length = checkpoint.get('max_length', 512)
        dropout = checkpoint.get('dropout', 0.1)
        
        # Create model
        model = cls(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_length=max_length
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded from {path}")
        return model
