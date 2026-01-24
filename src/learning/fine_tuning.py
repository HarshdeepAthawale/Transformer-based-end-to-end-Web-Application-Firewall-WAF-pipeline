"""
Incremental Fine-Tuning Module

Fine-tune model on incremental data without full retraining
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from loguru import logger
from typing import List, Optional, Dict
from pathlib import Path
from tqdm import tqdm

from src.model.anomaly_detector import AnomalyDetector
from src.tokenization.tokenizer import HTTPTokenizer
from src.tokenization.dataloader import create_dataloader


class IncrementalFineTuner:
    """Fine-tune model on incremental data"""
    
    def __init__(
        self,
        base_model_path: str,
        vocab_path: str,
        device: str = None,
        learning_rate: float = 1e-5,  # Lower LR for fine-tuning
        num_epochs: int = 3,  # Fewer epochs for incremental
        batch_size: int = 16  # Smaller batch for fine-tuning
    ):
        """
        Initialize incremental fine-tuner
        
        Args:
            base_model_path: Path to base model checkpoint
            vocab_path: Path to vocabulary file
            device: Device to train on (default: auto-detect)
            learning_rate: Learning rate for fine-tuning (lower than training)
            num_epochs: Number of epochs for fine-tuning (fewer than training)
            batch_size: Batch size for fine-tuning
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        # Load base model
        logger.info(f"Loading base model from {base_model_path}")
        checkpoint = torch.load(base_model_path, map_location=device, weights_only=True)
        vocab_size = checkpoint.get('vocab_size', 10000)
        
        # Infer architecture from checkpoint
        state_dict = checkpoint['model_state_dict']
        
        # Get hidden size from word embeddings
        if 'transformer.embeddings.word_embeddings.weight' in state_dict:
            hidden_size = state_dict['transformer.embeddings.word_embeddings.weight'].shape[1]
        else:
            hidden_size = 768  # fallback
        
        # Get max length from position embeddings
        if 'transformer.embeddings.position_embeddings.weight' in state_dict:
            max_length = state_dict['transformer.embeddings.position_embeddings.weight'].shape[0]
        else:
            max_length = 512  # fallback
        
        # Infer number of layers
        num_layers = 0
        while f'transformer.transformer.layer.{num_layers}.attention.q_lin.weight' in state_dict:
            num_layers += 1
        
        # Infer number of heads
        possible_heads = [8, 12, 16]
        num_heads = 12  # default
        for heads in possible_heads:
            if hidden_size % heads == 0:
                num_heads = heads
                break
        
        logger.info(f"Model architecture: vocab_size={vocab_size}, hidden_size={hidden_size}, "
                   f"num_layers={num_layers}, num_heads={num_heads}, max_length={max_length}")
        
        self.model = AnomalyDetector(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_length=max_length
        )
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {vocab_path}")
        self.tokenizer = HTTPTokenizer()
        self.tokenizer.load_vocab(vocab_path)
        
        logger.info(f"IncrementalFineTuner initialized: device={device}, lr={learning_rate}, epochs={num_epochs}")
    
    def fine_tune(
        self,
        new_data: List[str],
        validation_data: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> AnomalyDetector:
        """
        Fine-tune model on new data
        
        Args:
            new_data: List of new normalized request texts
            validation_data: Optional validation data
            output_path: Optional path to save fine-tuned model
        
        Returns:
            Fine-tuned model
        """
        if len(new_data) < 10:
            logger.warning(f"Too few samples for fine-tuning: {len(new_data)}")
            return self.model
        
        logger.info(f"Fine-tuning on {len(new_data)} new samples...")
        
        # Create data loaders
        train_loader = create_dataloader(
            new_data,
            self.tokenizer,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_loader = None
        if validation_data and len(validation_data) > 0:
            val_loader = create_dataloader(
                validation_data,
                self.tokenizer,
                batch_size=self.batch_size,
                shuffle=False
            )
        
        # Setup optimizer (lower learning rate for fine-tuning)
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Fine-tune
        best_val_loss = float('inf')
        best_model_state = None
        
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
                    best_model_state = self.model.state_dict().copy()
                    logger.info(f"New best validation loss: {best_val_loss:.4f}")
            else:
                # No validation, save current model
                best_model_state = self.model.state_dict().copy()
        
        # Load best model state
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # Save checkpoint
        if output_path:
            self._save_checkpoint(output_path, train_loss, best_val_loss if val_loader else None)
        
        logger.info("Fine-tuning complete")
        return self.model
    
    def _train_epoch(self, train_loader: DataLoader, optimizer) -> float:
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
            
            # Loss: target is 0 (benign traffic)
            target = torch.zeros_like(anomaly_scores)
            loss = nn.MSELoss()(anomaly_scores, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validating", leave=False)
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                anomaly_scores = outputs['anomaly_score']
                
                target = torch.zeros_like(anomaly_scores)
                loss = nn.MSELoss()(anomaly_scores, target)
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _save_checkpoint(
        self,
        path: str,
        train_loss: float,
        val_loss: Optional[float] = None
    ):
        """Save fine-tuned model"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'vocab_size': len(self.tokenizer.word_to_id),
            'fine_tuned': True,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Fine-tuned checkpoint saved: {path}")
