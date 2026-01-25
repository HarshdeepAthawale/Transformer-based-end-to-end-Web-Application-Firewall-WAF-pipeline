"""
Incremental Fine-Tuning Module

Fine-tune model on incremental data only (not full retraining).
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from loguru import logger
from typing import List, Optional
from pathlib import Path

from backend.ml.model.anomaly_detector import AnomalyDetector
from backend.ml.tokenization.tokenizer import HTTPTokenizer
from backend.ml.tokenization.dataloader import create_dataloader


class IncrementalFineTuner:
    """Fine-tune model on incremental data"""
    
    def __init__(
        self,
        base_model_path: str,
        vocab_path: str,
        device: str = None,
        learning_rate: float = 1e-5,  # Lower LR for fine-tuning
        num_epochs: int = 3  # Fewer epochs for incremental
    ):
        """
        Initialize fine-tuner
        
        Args:
            base_model_path: Path to base model checkpoint
            vocab_path: Path to vocabulary file
            device: Device for training
            learning_rate: Learning rate (lower for fine-tuning)
            num_epochs: Number of epochs (fewer for incremental)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Load base model
        logger.info(f"Loading base model from {base_model_path}...")
        self.model = AnomalyDetector.load_checkpoint(base_model_path, device=device)
        self.model.train()  # Set to training mode
        
        # Load tokenizer
        logger.info(f"Loading vocabulary from {vocab_path}...")
        self.tokenizer = HTTPTokenizer()
        self.tokenizer.load_vocab(vocab_path)
    
    def fine_tune(
        self,
        new_data: List[str],
        validation_data: Optional[List[str]] = None,
        output_path: str = None
    ) -> AnomalyDetector:
        """
        Fine-tune model on new data
        
        Args:
            new_data: List of new normalized request strings
            validation_data: Optional validation data
            output_path: Path to save fine-tuned model
        
        Returns:
            Fine-tuned model
        """
        logger.info(f"Fine-tuning on {len(new_data)} new samples...")
        
        # Create data loaders
        train_loader = create_dataloader(
            new_data,
            self.tokenizer,
            batch_size=16,  # Smaller batch for fine-tuning
            shuffle=True
        )
        
        val_loader = None
        if validation_data:
            val_loader = create_dataloader(
                validation_data,
                self.tokenizer,
                batch_size=16,
                shuffle=False
            )
        
        # Setup optimizer (lower learning rate)
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Fine-tune
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            logger.info(f"Fine-tuning epoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss = self._train_epoch(train_loader, optimizer)
            logger.info(f"Train loss: {train_loss:.4f}")
            
            # Validate
            if val_loader:
                val_loss = self._validate_epoch(val_loader)
                logger.info(f"Val loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if output_path:
                        self._save_checkpoint(output_path, epoch, train_loss, val_loss)
        
        logger.info("Fine-tuning complete")
        return self.model
    
    def _train_epoch(self, train_loader: DataLoader, optimizer) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            anomaly_scores = outputs['anomaly_score']
            
            # Loss: target is 0 (benign)
            target = torch.zeros_like(anomaly_scores)
            loss = nn.MSELoss()(anomaly_scores, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
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
        """Save checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        vocab_size = len(self.tokenizer.word_to_id) if self.tokenizer.vocab_built else 10000
        self.model.save_checkpoint(
            path,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            vocab_size=vocab_size
        )
