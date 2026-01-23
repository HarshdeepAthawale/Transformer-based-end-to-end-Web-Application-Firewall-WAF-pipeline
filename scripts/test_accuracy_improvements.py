#!/usr/bin/env python3
"""
Test Accuracy Improvements

Quick test to verify all accuracy improvement components work
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.model.anomaly_detector import AnomalyDetector
from src.tokenization.tokenizer import HTTPTokenizer
from src.tokenization.dataloader import create_dataloader
from src.training.evaluator import ModelEvaluator
from src.training.losses import get_loss_function, WeightedMSELoss, FocalLoss
from src.training.threshold_optimizer import ThresholdOptimizer
from src.training.data_augmentation import RequestAugmenter, augment_dataset
from src.training.report_generator import ReportGenerator
from tests.payloads.malicious_payloads import get_all_malicious_payloads, generate_malicious_requests

def test_all_components():
    """Test all accuracy improvement components"""
    print("="*60)
    print("Testing Accuracy Improvement Components")
    print("="*60)
    
    # 1. Test malicious payloads
    print("\n1. Testing malicious payloads...")
    all_payloads = get_all_malicious_payloads()
    total_payloads = sum(len(p) for p in all_payloads.values())
    print(f"   ✓ Loaded {len(all_payloads)} categories, {total_payloads} total payloads")
    
    malicious_requests = generate_malicious_requests()
    print(f"   ✓ Generated {len(malicious_requests)} malicious requests")
    
    # 2. Test data augmentation
    print("\n2. Testing data augmentation...")
    test_requests = ["GET /api/users HTTP/1.1", "POST /api/data?id=123 HTTP/1.1"]
    augmenter = RequestAugmenter()
    augmented = [augmenter.augment(req) for req in test_requests]
    print(f"   ✓ Augmented {len(test_requests)} requests")
    
    # 3. Test loss functions
    print("\n3. Testing loss functions...")
    predictions = torch.tensor([0.1, 0.2, 0.3, 0.4])
    targets = torch.zeros(4)
    
    mse_loss = get_loss_function("mse")
    weighted_mse = get_loss_function("weighted_mse", false_positive_weight=2.0)
    focal_loss = get_loss_function("focal", alpha=1.0, gamma=2.0)
    
    loss1 = mse_loss(predictions, targets)
    loss2 = weighted_mse(predictions, targets)
    loss3 = focal_loss(predictions, targets)
    
    print(f"   ✓ MSE Loss: {loss1.item():.4f}")
    print(f"   ✓ Weighted MSE Loss: {loss2.item():.4f}")
    print(f"   ✓ Focal Loss: {loss3.item():.4f}")
    
    # 4. Test evaluator
    print("\n4. Testing evaluator...")
    vocab_size = 100
    model = AnomalyDetector(vocab_size=vocab_size, hidden_size=128, num_layers=2, num_heads=2)
    evaluator = ModelEvaluator(model)
    
    # Create dummy data
    tokenizer = HTTPTokenizer(vocab_size=100)
    tokenizer.build_vocab(["GET /api/users HTTP/1.1", "POST /api/data HTTP/1.1"])
    texts = ["GET /api/users HTTP/1.1"] * 10
    dataloader = create_dataloader(texts, tokenizer, batch_size=4, max_length=128, shuffle=False)
    
    metrics = evaluator.evaluate(dataloader, threshold=0.5)
    print(f"   ✓ Calculated metrics: TPR={metrics['tpr']:.4f}, FPR={metrics['fpr']:.4f}, F1={metrics['f1_score']:.4f}")
    
    # 5. Test threshold optimizer
    print("\n5. Testing threshold optimizer...")
    optimizer = ThresholdOptimizer(evaluator)
    labels = [0] * 10  # All benign
    result = optimizer.find_optimal_threshold(dataloader, labels, method="f1_maximize")
    print(f"   ✓ Optimal threshold: {result['optimal_threshold']:.4f}")
    
    # 6. Test report generator
    print("\n6. Testing report generator...")
    report_gen = ReportGenerator(output_dir="reports")
    print(f"   ✓ Report generator initialized")
    
    print("\n" + "="*60)
    print("All components tested successfully! ✓")
    print("="*60)

if __name__ == "__main__":
    test_all_components()
