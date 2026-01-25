"""
Training Module

Main training pipeline for anomaly detection model.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from loguru import logger

from backend.ml.model.anomaly_detector import AnomalyDetector
from backend.ml.tokenization.tokenizer import HTTPTokenizer
from backend.ml.tokenization.dataloader import create_dataloader


class AnomalyDetectionTrainer:
    """Trainer for anomaly detection model"""
    
    def __init__(
        self,
        model: AnomalyDetector,
        tokenizer: HTTPTokenizer,
        device: str = None
    ):
        """
        Initialize trainer
        
        Args:
            model: AnomalyDetector model instance
            tokenizer: HTTPTokenizer instance
            device: Device for training (auto-detect if None)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = None
        self.scheduler = None
        
        logger.info(f"Trainer initialized on device: {device}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 2e-5,
        checkpoint_dir: str = "models/checkpoints",
        save_best: bool = True,
        early_stopping_patience: Optional[int] = None,
        min_delta: float = 0.0
    ):
        """
        Train the model
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save best model
            early_stopping_patience: Early stopping patience (None to disable)
            min_delta: Minimum change to qualify as improvement
        """
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Create checkpoint directory
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss = self._train_epoch(train_loader)
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Validation
            val_loss = self._validate(val_loader)
            logger.info(f"Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            self._save_checkpoint(checkpoint_path, epoch, train_loss, val_loss)
            
            # Save best model
            if save_best and val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                self._save_checkpoint(best_path, epoch, train_loss, val_loss)
                logger.info(f"Saved best model (val_loss: {val_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            logger.info(f"Learning rate: {current_lr:.2e}")
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            anomaly_scores = outputs['anomaly_score']
            
            # Loss: We want all benign requests to have low anomaly scores
            # Using MSE loss with target = 0 (no anomaly)
            target = torch.zeros_like(anomaly_scores)
            loss = nn.MSELoss()(anomaly_scores, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                anomaly_scores = outputs['anomaly_score']
                
                target = torch.zeros_like(anomaly_scores)
                loss = nn.MSELoss()(anomaly_scores, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _save_checkpoint(
        self,
        path: str,
        epoch: int,
        train_loss: float,
        val_loss: float
    ):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Get vocab size from tokenizer
        vocab_size = len(self.tokenizer.word_to_id) if self.tokenizer.vocab_built else 10000
        
        # Save using model's checkpoint method
        self.model.save_checkpoint(
            path,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            vocab_size=vocab_size
        )
