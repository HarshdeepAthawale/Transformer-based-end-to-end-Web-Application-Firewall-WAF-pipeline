"""
Advanced Loss Functions Module

Advanced loss functions for anomaly detection training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedMSELoss(nn.Module):
    """Weighted MSE Loss - penalizes false positives more"""
    
    def __init__(self, false_positive_weight: float = 2.0):
        """
        Initialize weighted MSE loss
        
        Args:
            false_positive_weight: Weight multiplier for false positives
        """
        super().__init__()
        self.false_positive_weight = false_positive_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted MSE loss
        
        Args:
            predictions: Model predictions (anomaly scores)
            targets: Target values (0 for benign)
        
        Returns:
            Weighted MSE loss
        """
        # Calculate base MSE
        mse = (predictions - targets) ** 2
        
        # Identify potential false positives (high prediction, low target)
        # False positive: prediction > threshold but target = 0
        # We weight these more heavily
        weights = torch.ones_like(mse)
        false_positive_mask = (predictions > 0.3) & (targets < 0.1)
        weights[false_positive_mask] = self.false_positive_weight
        
        weighted_mse = weights * mse
        return weighted_mse.mean()


class FocalLoss(nn.Module):
    """Focal Loss - focuses learning on hard examples"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        """
        Initialize focal loss
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss
        
        Args:
            predictions: Model predictions (anomaly scores)
            targets: Target values (0 for benign)
        
        Returns:
            Focal loss
        """
        # Convert to binary cross entropy format
        # For anomaly detection, we want predictions close to 0 for benign
        # So we use BCE with targets = 0
        
        # Calculate BCE
        bce = F.binary_cross_entropy(predictions, targets, reduction='none')
        
        # Calculate focal weight
        # pt = probability of true class
        pt = torch.where(targets == 0, 1 - predictions, predictions)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Final focal loss
        focal_loss = alpha_t * focal_weight * bce
        
        return focal_loss.mean()


class ContrastiveLoss(nn.Module):
    """Contrastive Loss - learns better separation between classes"""
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss
        
        Args:
            margin: Margin for contrastive loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate contrastive loss
        
        Args:
            predictions: Model predictions
            targets: Target values
            embeddings: Optional embeddings for contrastive learning
        
        Returns:
            Contrastive loss
        """
        # For anomaly detection, we want:
        # - Benign samples (target=0) to have low scores
        # - Anomalous samples (target=1) to have high scores
        
        # Simple contrastive loss based on predictions
        # Pull benign samples towards 0, push anomalies away from 0
        benign_mask = (targets < 0.5)
        anomaly_mask = (targets >= 0.5)
        
        loss = torch.tensor(0.0, device=predictions.device)
        
        if benign_mask.any():
            # Benign samples should be close to 0
            benign_loss = predictions[benign_mask].mean()
            loss += benign_loss
        
        if anomaly_mask.any():
            # Anomalous samples should be far from 0
            anomaly_loss = F.relu(self.margin - predictions[anomaly_mask]).mean()
            loss += anomaly_loss
        
        return loss


class CombinedLoss(nn.Module):
    """Combined Loss - MSE + Regularization"""
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        l1_weight: float = 0.01,
        false_positive_weight: float = 2.0
    ):
        """
        Initialize combined loss
        
        Args:
            mse_weight: Weight for MSE component
            l1_weight: Weight for L1 regularization
            false_positive_weight: Weight for false positive penalty
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.false_positive_weight = false_positive_weight
        self.mse_loss = nn.MSELoss()
        self.weighted_mse = WeightedMSELoss(false_positive_weight=false_positive_weight)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss
        
        Args:
            predictions: Model predictions
            targets: Target values
        
        Returns:
            Combined loss
        """
        # Weighted MSE loss
        mse_loss = self.weighted_mse(predictions, targets)
        
        # L1 regularization on predictions (encourage sparsity)
        l1_loss = torch.abs(predictions).mean()
        
        # Combined loss
        total_loss = self.mse_weight * mse_loss + self.l1_weight * l1_loss
        
        return total_loss


def get_loss_function(loss_type: str = "mse", **kwargs):
    """
    Get loss function by type
    
    Args:
        loss_type: Type of loss ("mse", "weighted_mse", "focal", "contrastive", "combined")
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function instance
    """
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "weighted_mse":
        false_positive_weight = kwargs.get("false_positive_weight", 2.0)
        return WeightedMSELoss(false_positive_weight=false_positive_weight)
    elif loss_type == "focal":
        alpha = kwargs.get("alpha", 1.0)
        gamma = kwargs.get("gamma", 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == "contrastive":
        margin = kwargs.get("margin", 1.0)
        return ContrastiveLoss(margin=margin)
    elif loss_type == "combined":
        mse_weight = kwargs.get("mse_weight", 1.0)
        l1_weight = kwargs.get("l1_weight", 0.01)
        false_positive_weight = kwargs.get("false_positive_weight", 2.0)
        return CombinedLoss(
            mse_weight=mse_weight,
            l1_weight=l1_weight,
            false_positive_weight=false_positive_weight
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
