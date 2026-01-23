#!/usr/bin/env python3
"""
Test Training Pipeline

Quick test to verify training pipeline components work together
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.anomaly_detector import AnomalyDetector
from src.model.scoring import AnomalyScorer
from src.tokenization.tokenizer import HTTPTokenizer
from src.tokenization.dataloader import create_dataloader
from src.training.dataset_generator import SyntheticDatasetGenerator
import torch

def test_complete_pipeline():
    """Test complete training pipeline components"""
    print("="*60)
    print("Testing Complete Training Pipeline")
    print("="*60)
    
    # 1. Generate synthetic data
    print("\n1. Generating synthetic data...")
    generator = SyntheticDatasetGenerator()
    texts = generator.generate_dataset(50)
    print(f"   Generated {len(texts)} samples")
    
    # 2. Build tokenizer
    print("\n2. Building tokenizer...")
    tokenizer = HTTPTokenizer(vocab_size=1000)
    tokenizer.build_vocab(texts)
    print(f"   Vocabulary size: {len(tokenizer.word_to_id)}")
    
    # 3. Create data loader
    print("\n3. Creating data loader...")
    dataloader = create_dataloader(
        texts,
        tokenizer,
        batch_size=8,
        max_length=128,
        shuffle=False
    )
    print(f"   Batches: {len(dataloader)}")
    
    # 4. Create model
    print("\n4. Creating model...")
    vocab_size = len(tokenizer.word_to_id)
    model = AnomalyDetector(
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=2,
        num_heads=2,
        max_length=128
    )
    print(f"   Model created with vocab_size={vocab_size}")
    
    # 5. Test forward pass
    print("\n5. Testing forward pass...")
    batch = next(iter(dataloader))
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    
    outputs = model(input_ids, attention_mask)
    print(f"   Anomaly scores shape: {outputs['anomaly_score'].shape}")
    print(f"   Score range: [{outputs['anomaly_score'].min():.4f}, {outputs['anomaly_score'].max():.4f}]")
    
    # 6. Test scorer
    print("\n6. Testing anomaly scorer...")
    scorer = AnomalyScorer(model, threshold=0.5)
    result = scorer.score_batch(input_ids, attention_mask)
    print(f"   Scored {len(result['anomaly_scores'])} samples")
    print(f"   Anomalies detected: {sum(result['is_anomaly'])}")
    
    # 7. Test checkpoint saving/loading
    print("\n7. Testing checkpoint...")
    checkpoint_path = "models/checkpoints/test_checkpoint.pt"
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"   Checkpoint saved to {checkpoint_path}")
    
    # Load checkpoint
    loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"   Checkpoint loaded: epoch={loaded_checkpoint['epoch']}")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)

if __name__ == "__main__":
    test_complete_pipeline()
