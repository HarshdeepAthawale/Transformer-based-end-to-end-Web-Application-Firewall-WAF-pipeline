#!/usr/bin/env python3
"""
Quick Model Training Script

Trains the Transformer model with minimal configuration for Phase 5 completion
"""
import sys
import torch
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger.add("logs/training.log", rotation="10 MB", retention="7 days")


def main():
    """Train the model"""
    logger.info("=" * 60)
    logger.info("QUICK MODEL TRAINING")
    logger.info("=" * 60)
    
    try:
        from src.model.anomaly_detector import AnomalyDetector
        from src.tokenization.tokenizer import HTTPTokenizer
        from src.tokenization.dataloader import create_dataloader
        from src.training.train import AnomalyDetectionTrainer
        
        # Load vocabulary
        vocab_path = project_root / "models" / "vocabularies" / "http_vocab.json"
        if not vocab_path.exists():
            logger.error(f"Vocabulary not found: {vocab_path}")
            logger.info("Please build vocabulary first using scripts/build_vocabulary.py")
            return 1
        
        tokenizer = HTTPTokenizer()
        tokenizer.load_vocab(str(vocab_path))
        vocab_size = len(tokenizer.word_to_id)
        logger.info(f"✅ Loaded vocabulary with {vocab_size} tokens")
        
        # Generate or load training data
        training_file = project_root / "data" / "training" / "benign_requests.txt"
        
        if not training_file.exists() or training_file.stat().st_size < 1000:
            logger.info("Generating training data...")
            
            # Generate synthetic data
            from src.training.dataset_generator import SyntheticDatasetGenerator
            
            generator = SyntheticDatasetGenerator()
            
            # Try to load from logs
            log_path = Path("/var/log/nginx/access.log")
            if log_path.exists():
                generator.load_from_applications([str(log_path)], max_lines=5000)
            else:
                generator._load_default_patterns()
            
            # Generate samples
            texts = []
            for _ in range(1000):  # Generate 1000 samples
                method = generator.methods[torch.randint(0, len(generator.methods), (1,)).item()]
                path = generator.paths[torch.randint(0, len(generator.paths), (1,)).item()] if generator.paths else "/api/data"
                request = f"{method} {path} HTTP/1.1"
                texts.append(request)
            
            # Save
            training_file.parent.mkdir(parents=True, exist_ok=True)
            with open(training_file, 'w') as f:
                for text in texts:
                    f.write(text + "\n")
            
            logger.info(f"✅ Generated {len(texts)} training samples")
        else:
            # Load existing
            with open(training_file, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
            logger.info(f"✅ Loaded {len(texts)} training samples")
        
        if len(texts) < 100:
            logger.error(f"Not enough training data: {len(texts)} samples. Need at least 100.")
            return 1
        
        # Split data
        split_idx = int(len(texts) * 0.8)
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        
        logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}")
        
        # Create data loaders
        max_length = 128  # Reduced for faster training
        batch_size = 16  # Smaller batch for memory efficiency
        
        train_loader = create_dataloader(
            train_texts,
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=True
        )
        
        val_loader = create_dataloader(
            val_texts,
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=False
        )
        
        # Create model
        model = AnomalyDetector(
            vocab_size=vocab_size,
            hidden_size=256,  # Reduced for faster training
            num_layers=3,  # Reduced
            num_heads=4,  # Reduced
            dropout=0.1,
            max_length=max_length
        )
        
        logger.info("✅ Model created")
        logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create trainer
        trainer = AnomalyDetectionTrainer(model, tokenizer)
        
        # Train with minimal epochs
        checkpoint_dir = project_root / "models" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting training (this may take a few minutes)...")
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=3,  # Minimal epochs for quick completion
            learning_rate=2e-5,
            weight_decay=0.01,
            gradient_clip=1.0,
            checkpoint_dir=str(checkpoint_dir),
            save_best=True,
            loss_type="mse"
        )
        
        # Check if best model was saved
        best_model_path = checkpoint_dir / "best_model.pt"
        if best_model_path.exists():
            size_mb = best_model_path.stat().st_size / 1024 / 1024
            logger.info(f"✅ Model trained and saved!")
            logger.info(f"   Path: {best_model_path}")
            logger.info(f"   Size: {size_mb:.2f} MB")
            logger.info(f"   Best validation loss: {results['best_val_loss']:.4f}")
            return 0
        else:
            logger.error("❌ Model checkpoint not found after training")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
