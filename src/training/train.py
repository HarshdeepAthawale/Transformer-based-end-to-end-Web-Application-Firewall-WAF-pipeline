"""
Training Module

Trainer for anomaly detection model with evaluation metrics and early stopping
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from loguru import logger
import os
from pathlib import Path
from typing import Optional, Dict, List

from src.model.anomaly_detector import AnomalyDetector
from src.tokenization.tokenizer import HTTPTokenizer
from src.training.losses import get_loss_function
from src.training.evaluator import ModelEvaluator


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
            model: AnomalyDetector model
            tokenizer: HTTPTokenizer instance
            device: Device to train on (default: auto-detect)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = None
        self.scheduler = None
        
        logger.info(f"AnomalyDetectionTrainer initialized on device: {device}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        gradient_clip: float = 1.0,
        checkpoint_dir: str = "models/checkpoints",
        save_best: bool = True,
        save_every_n_epochs: int = 1,
        loss_type: str = "mse",
        loss_kwargs: Optional[Dict] = None,
        early_stopping: Optional[Dict] = None,
        evaluator: Optional[ModelEvaluator] = None,
        val_labels: Optional[List[int]] = None,
        threshold: float = 0.5
    ):
        """
        Train the model
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            gradient_clip: Gradient clipping value
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save best model
            save_every_n_epochs: Save checkpoint every N epochs
            loss_type: Type of loss function ("mse", "weighted_mse", "focal", etc.)
            loss_kwargs: Additional arguments for loss function
            early_stopping: Early stopping config {"enabled": bool, "patience": int, "metric": str, "min_delta": float}
            evaluator: Optional ModelEvaluator for metrics calculation
            val_labels: Optional labels for validation set (for metrics)
            threshold: Threshold for anomaly detection (for metrics)
        """
        # Setup loss function
        loss_kwargs = loss_kwargs or {}
        self.criterion = get_loss_function(loss_type, **loss_kwargs)
        logger.info(f"Using loss function: {loss_type}")
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )
        
        # Early stopping setup
        early_stopping_enabled = early_stopping and early_stopping.get("enabled", False)
        early_stopping_patience = early_stopping.get("patience", 5) if early_stopping else 5
        early_stopping_metric = early_stopping.get("metric", "f1_score") if early_stopping else "f1_score"
        early_stopping_min_delta = early_stopping.get("min_delta", 0.001) if early_stopping else 0.001
        
        best_val_loss = float('inf')
        best_metric_value = 0.0 if early_stopping_metric in ["f1_score", "tpr", "precision", "recall"] else float('inf')
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        all_metrics = []
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        if early_stopping_enabled:
            logger.info(f"Early stopping: enabled (patience={early_stopping_patience}, metric={early_stopping_metric})")
        
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # Training
            train_loss = self._train_epoch(train_loader, gradient_clip)
            train_losses.append(train_loss)
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Validation
            val_loss = self._validate(val_loader)
            val_losses.append(val_loss)
            logger.info(f"Val Loss: {val_loss:.4f}")
            
            # Calculate evaluation metrics if evaluator provided
            epoch_metrics = {}
            if evaluator is not None:
                metrics = evaluator.evaluate(val_loader, threshold=threshold, labels=val_labels)
                epoch_metrics = metrics
                all_metrics.append(metrics)
                
                logger.info(f"Validation Metrics:")
                logger.info(f"  TPR (Recall): {metrics['tpr']:.4f}")
                logger.info(f"  FPR: {metrics['fpr']:.4f}")
                logger.info(f"  Precision: {metrics['precision']:.4f}")
                logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
                if metrics['roc_auc'] > 0:
                    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every_n_epochs == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                self._save_checkpoint(checkpoint_path, epoch, train_loss, val_loss, epoch_metrics)
            
            # Determine best model (loss or metric-based)
            is_best = False
            if early_stopping_enabled and epoch_metrics:
                metric_value = epoch_metrics.get(early_stopping_metric, 0.0)
                if early_stopping_metric in ["f1_score", "tpr", "precision", "recall"]:
                    # Higher is better
                    if metric_value > best_metric_value + early_stopping_min_delta:
                        best_metric_value = metric_value
                        is_best = True
                        patience_counter = 0
                    else:
                        patience_counter += 1
                else:
                    # Lower is better (for loss, fpr)
                    if metric_value < best_metric_value - early_stopping_min_delta:
                        best_metric_value = metric_value
                        is_best = True
                        patience_counter = 0
                    else:
                        patience_counter += 1
            else:
                # Use loss-based selection
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    is_best = True
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            # Save best model
            if save_best and is_best:
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                self._save_checkpoint(best_path, epoch, train_loss, val_loss, epoch_metrics)
                if early_stopping_enabled and epoch_metrics:
                    logger.info(f"✓ Saved best model ({early_stopping_metric}: {best_metric_value:.4f})")
                else:
                    logger.info(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Early stopping check
            if early_stopping_enabled and patience_counter >= early_stopping_patience:
                logger.info(f"\nEarly stopping triggered! No improvement for {patience_counter} epochs")
                logger.info(f"Best {early_stopping_metric}: {best_metric_value:.4f}")
                break
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.2e}")
        
        logger.info(f"\n{'='*60}")
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        if all_metrics:
            best_metrics = max(all_metrics, key=lambda x: x.get('f1_score', 0))
            logger.info(f"Best F1 Score: {best_metrics['f1_score']:.4f}")
            logger.info(f"Best TPR: {best_metrics['tpr']:.4f}")
            logger.info(f"Best FPR: {best_metrics['fpr']:.4f}")
        logger.info(f"{'='*60}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'metrics_history': all_metrics
        }
    
    def _train_epoch(self, train_loader: DataLoader, gradient_clip: float) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            anomaly_scores = outputs['anomaly_score']
            
            # Loss: We want all benign requests to have low anomaly scores
            # Target = 0 (no anomaly)
            target = torch.zeros_like(anomaly_scores)
            loss = self.criterion(anomaly_scores, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation", leave=False)
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                anomaly_scores = outputs['anomaly_score']
                
                target = torch.zeros_like(anomaly_scores)
                loss = self.criterion(anomaly_scores, target)
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _save_checkpoint(
        self,
        path: str,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: Optional[Dict] = None
    ):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'vocab_size': len(self.tokenizer.word_to_id) if hasattr(self.tokenizer, 'word_to_id') else None,
            'metrics': metrics or {}
        }
        
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint
