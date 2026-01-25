#!/usr/bin/env python3
"""
Main Training Script

Train the transformer-based anomaly detection model.
"""
import argparse
import json
from pathlib import Path
from loguru import logger
import torch

from backend.ml.tokenization.tokenizer import HTTPTokenizer
from backend.ml.tokenization.dataloader import create_dataloader
from backend.ml.model.anomaly_detector import AnomalyDetector
from backend.ml.training.train import AnomalyDetectionTrainer
from backend.ml.training.dataset_generator import DatasetGenerator
from backend.ml.training.evaluator import ModelEvaluator
from backend.ml.training.threshold_optimizer import ThresholdOptimizer
from backend.ml.training.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection model")
    parser.add_argument("--data", type=str, required=True, help="Path to training data JSON file")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for models")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto-detect if None)")
    parser.add_argument("--early-stopping", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--split-data", action="store_true", help="Split data into train/val/test")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info(f"Loading data from {args.data}...")
    if args.split_data:
        # Load and split data
        texts = DatasetGenerator.load_data(args.data)
        train_texts, val_texts, test_texts = DatasetGenerator.split_data(
            texts,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        
        # Save splits
        output_dir = Path(args.output_dir)
        DatasetGenerator.save_datasets(train_texts, val_texts, test_texts, str(output_dir / "datasets"))
        
        # Use train and val for training
        train_data = train_texts
        val_data = val_texts
    else:
        # Assume data is already split or use single file
        data = DatasetGenerator.load_data(args.data)
        # Simple split if single file
        split_idx = int(len(data) * args.train_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
    
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    
    # Build vocabulary
    logger.info("Building vocabulary...")
    tokenizer = HTTPTokenizer(
        vocab_size=args.vocab_size,
        max_length=args.max_length
    )
    tokenizer.build_vocab(train_data)
    
    # Save vocabulary
    vocab_path = Path(args.output_dir) / "vocabularies" / "vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save_vocab(str(vocab_path))
    logger.info(f"Vocabulary saved to {vocab_path}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = create_dataloader(
        train_data,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True
    )
    
    val_loader = create_dataloader(
        val_data,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False
    )
    
    # Create model
    logger.info("Creating model...")
    model = AnomalyDetector(
        vocab_size=len(tokenizer.word_to_id),
        max_length=args.max_length
    )
    
    # Create trainer
    trainer = AnomalyDetectionTrainer(model, tokenizer, device=device)
    
    # Train
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    logger.info("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=str(checkpoint_dir),
        save_best=True,
        early_stopping_patience=args.early_stopping
    )
    
    # Load best model for evaluation
    best_model_path = checkpoint_dir / "best_model.pt"
    if best_model_path.exists():
        logger.info("Loading best model for evaluation...")
        model = AnomalyDetector.load_checkpoint(str(best_model_path), device=device)
        
        # Evaluate
        evaluator = ModelEvaluator(model, tokenizer, device=device)
        val_metrics = evaluator.evaluate_on_loader(val_loader, threshold=0.5)
        evaluator.print_metrics(val_metrics)
        
        # Save final model
        final_model_path = Path(args.output_dir) / "deployed" / "model.pt"
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(str(final_model_path), vocab_size=len(tokenizer.word_to_id))
        logger.info(f"Final model saved to {final_model_path}")
        
        # Generate report
        ReportGenerator.generate_training_report(
            metrics=val_metrics,
            output_path=str(Path(args.output_dir) / "reports" / "training_report.json"),
            model_path=str(final_model_path),
            vocab_path=str(vocab_path),
            training_config={
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'vocab_size': len(tokenizer.word_to_id)
            }
        )
    else:
        logger.warning("Best model not found, skipping evaluation")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
