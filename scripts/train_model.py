#!/usr/bin/env python3
"""
Training Pipeline Script

End-to-end training pipeline for anomaly detection model
"""
import argparse
from pathlib import Path
from loguru import logger
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.ingestion import LogIngestionSystem
from src.parsing.pipeline import ParsingPipeline
from src.tokenization.tokenizer import HTTPTokenizer
from src.tokenization.dataloader import create_dataloader
from src.model.anomaly_detector import AnomalyDetector
from src.training.train import AnomalyDetectionTrainer
from src.training.dataset_generator import SyntheticDatasetGenerator
from src.training.evaluator import ModelEvaluator
from src.training.threshold_optimizer import ThresholdOptimizer
from src.training.data_augmentation import augment_dataset
from src.training.report_generator import ReportGenerator
from tests.payloads.malicious_payloads import generate_malicious_requests


def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection model")
    parser.add_argument(
        "--log_paths",
        nargs="+",
        default=[],
        help="Log file paths (optional, will use synthetic data if not provided)"
    )
    parser.add_argument(
        "--vocab_path",
        default="models/vocabularies/http_vocab.json",
        help="Path to vocabulary file"
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="models/checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=768,
        help="Hidden size for transformer"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=12,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Training split ratio"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (None = use all)"
    )
    parser.add_argument(
        "--use_synthetic",
        action="store_true",
        help="Use synthetic data generation"
    )
    parser.add_argument(
        "--synthetic_samples",
        type=int,
        default=10000,
        help="Number of synthetic samples to generate"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["mse", "weighted_mse", "focal", "contrastive", "combined"],
        help="Loss function type"
    )
    parser.add_argument(
        "--false_positive_weight",
        type=float,
        default=2.0,
        help="Weight for false positives (for weighted_mse)"
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        default="f1_score",
        choices=["f1_score", "tpr", "fpr", "precision", "recall", "val_loss"],
        help="Metric for early stopping"
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.001,
        help="Minimum delta for early stopping"
    )
    parser.add_argument(
        "--optimize_threshold",
        action="store_true",
        help="Optimize threshold after training"
    )
    parser.add_argument(
        "--target_fpr",
        type=float,
        default=0.01,
        help="Target false positive rate for threshold optimization"
    )
    parser.add_argument(
        "--threshold_method",
        type=str,
        default="f1_maximize",
        choices=["f1_maximize", "target_fpr", "roc_optimal"],
        help="Threshold optimization method"
    )
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        help="Use data augmentation"
    )
    parser.add_argument(
        "--augmentation_factor",
        type=int,
        default=1,
        help="Number of augmented versions per sample"
    )
    parser.add_argument(
        "--test_payloads",
        action="store_true",
        help="Include malicious payloads in test set"
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="reports",
        help="Directory for evaluation reports"
    )
    
    args = parser.parse_args()
    
    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.vocab_path).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Starting Model Training Pipeline")
    logger.info("="*60)
    
    # Load or build tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = HTTPTokenizer()
    
    vocab_path = Path(args.vocab_path)
    if vocab_path.exists():
        logger.info(f"Loading vocabulary from {vocab_path}")
        tokenizer.load_vocab(str(vocab_path))
    else:
        logger.warning(f"Vocabulary not found at {vocab_path}")
        logger.info("Vocabulary will be built from training data")
    
    # Prepare training data
    logger.info("Preparing training data...")
    pipeline = ParsingPipeline()
    ingestion = LogIngestionSystem()
    
    texts = []
    
    # Option 1: Use real log files
    if args.log_paths:
        logger.info(f"Processing {len(args.log_paths)} log file(s)...")
        for log_path in args.log_paths:
            if not Path(log_path).exists():
                logger.warning(f"Log file not found: {log_path}, skipping...")
                continue
            
            logger.info(f"Processing {log_path}...")
            line_count = 0
            for log_line in ingestion.ingest_batch(log_path, max_lines=args.max_samples):
                normalized = pipeline.process_log_line(log_line)
                if normalized:
                    texts.append(normalized)
                line_count += 1
                if args.max_samples and line_count >= args.max_samples:
                    break
            
            logger.info(f"Processed {line_count} lines from {log_path}")
    
    # Option 2: Use synthetic data
    if args.use_synthetic or (not args.log_paths and len(texts) == 0):
        logger.info("Generating synthetic training data...")
        generator = SyntheticDatasetGenerator()
        
        # Try to load patterns from logs if available
        if args.log_paths:
            available_logs = [lp for lp in args.log_paths if Path(lp).exists()]
            if available_logs:
                generator.load_from_applications(available_logs, max_lines=10000)
        
        synthetic_texts = generator.generate_dataset(args.synthetic_samples)
        texts.extend(synthetic_texts)
        logger.info(f"Generated {len(synthetic_texts)} synthetic samples")
    
    if len(texts) == 0:
        logger.error("No training data available! Please provide log files or use --use_synthetic")
        sys.exit(1)
    
    logger.info(f"Total training samples: {len(texts)}")
    
    # Apply data augmentation if enabled
    if args.use_augmentation:
        logger.info(f"Applying data augmentation (factor={args.augmentation_factor})...")
        texts = augment_dataset(texts, num_augmentations=args.augmentation_factor)
        logger.info(f"After augmentation: {len(texts)} samples")
    
    # Build vocabulary if needed
    if not tokenizer.vocab_built:
        logger.info("Building vocabulary from training data...")
        tokenizer.build_vocab(texts, save_path=str(vocab_path))
    else:
        logger.info(f"Using existing vocabulary: {len(tokenizer.word_to_id)} tokens")
    
    # Split data (train/val/test)
    train_split = args.train_split
    val_split = 1.0 - train_split
    test_split = 0.1  # 10% for test
    
    # Adjust splits
    if test_split > 0:
        train_split = train_split * (1 - test_split)
        val_split = val_split * (1 - test_split)
    
    train_idx = int(len(texts) * train_split)
    val_idx = train_idx + int(len(texts) * val_split)
    
    train_texts = texts[:train_idx]
    val_texts = texts[train_idx:val_idx]
    test_texts = texts[val_idx:] if test_split > 0 else []
    
    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")
    if test_texts:
        logger.info(f"Test samples: {len(test_texts)}")
    
    # Add malicious payloads to test set if requested
    test_labels = [0] * len(test_texts)  # All benign initially
    if args.test_payloads:
        logger.info("Generating malicious test payloads...")
        malicious_requests = generate_malicious_requests()
        # Normalize malicious requests
        malicious_normalized = []
        for req in malicious_requests:
            normalized = pipeline.process_log_line(req)
            if normalized:
                malicious_normalized.append(normalized)
        
        test_texts.extend(malicious_normalized)
        test_labels.extend([1] * len(malicious_normalized))  # Label as anomalies
        logger.info(f"Added {len(malicious_normalized)} malicious samples to test set")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = create_dataloader(
        train_texts,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True
    )
    
    val_loader = create_dataloader(
        val_texts,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False
    )
    
    test_loader = None
    if test_texts:
        test_loader = create_dataloader(
            test_texts,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            shuffle=False
        )
    
    # Create model
    logger.info("Creating model...")
    vocab_size = len(tokenizer.word_to_id)
    model = AnomalyDetector(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_length=args.max_length
    )
    
    logger.info(f"Model created with vocab_size={vocab_size}")
    
    # Setup evaluator for metrics
    evaluator = ModelEvaluator(model)
    val_labels = [0] * len(val_texts)  # All validation samples are benign
    
    # Setup early stopping
    early_stopping_config = None
    if args.early_stopping:
        early_stopping_config = {
            "enabled": True,
            "patience": args.early_stopping_patience,
            "metric": args.early_stopping_metric,
            "min_delta": args.early_stopping_min_delta
        }
    
    # Setup loss function
    loss_kwargs = {}
    if args.loss_type == "weighted_mse":
        loss_kwargs["false_positive_weight"] = args.false_positive_weight
    
    # Train
    logger.info("Starting training...")
    trainer = AnomalyDetectionTrainer(model, tokenizer)
    training_results = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        loss_type=args.loss_type,
        loss_kwargs=loss_kwargs,
        early_stopping=early_stopping_config,
        evaluator=evaluator,
        val_labels=val_labels,
        threshold=0.5  # Default threshold for metrics
    )
    
    logger.info("="*60)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {training_results['best_val_loss']:.4f}")
    logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")
    
    # Threshold optimization
    optimal_threshold = 0.5
    threshold_metrics = None
    if args.optimize_threshold:
        logger.info("\nOptimizing threshold...")
        threshold_optimizer = ThresholdOptimizer(evaluator)
        
        if test_loader and test_labels:
            # Use test set for threshold optimization
            result = threshold_optimizer.find_optimal_threshold(
                test_loader,
                test_labels,
                method=args.threshold_method,
                target_fpr=args.target_fpr
            )
        else:
            # Use validation set
            result = threshold_optimizer.find_optimal_threshold(
                val_loader,
                val_labels,
                method=args.threshold_method,
                target_fpr=args.target_fpr
            )
        
        optimal_threshold = result['optimal_threshold']
        threshold_metrics = result
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Final evaluation on test set
    test_metrics = None
    if test_loader and test_labels:
        logger.info("\nEvaluating on test set...")
        test_metrics = evaluator.evaluate_with_labels(
            test_loader,
            test_labels,
            threshold=optimal_threshold
        )
        logger.info(f"Test Metrics:")
        logger.info(f"  TPR: {test_metrics['tpr']:.4f}")
        logger.info(f"  FPR: {test_metrics['fpr']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  F1 Score: {test_metrics['f1_score']:.4f}")
    
    # Generate reports
    logger.info("\nGenerating evaluation reports...")
    report_generator = ReportGenerator(output_dir=args.report_dir)
    
    training_report_path = report_generator.generate_training_report(
        training_results,
        training_results.get('metrics_history', []),
        optimal_threshold=optimal_threshold,
        threshold_metrics=threshold_metrics
    )
    
    if test_metrics:
        eval_report_path = report_generator.generate_evaluation_report(test_metrics)
        logger.info(f"Evaluation report: {eval_report_path}")
    
    logger.info(f"Training report: {training_report_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
