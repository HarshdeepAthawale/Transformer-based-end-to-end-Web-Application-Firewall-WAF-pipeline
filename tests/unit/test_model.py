"""
Unit Tests for Model Components

Tests for anomaly detection model, scoring, and training
"""
import pytest
import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.anomaly_detector import AnomalyDetector
from src.model.scoring import AnomalyScorer
from src.tokenization.tokenizer import HTTPTokenizer


def test_model_forward():
    """Test model forward pass"""
    vocab_size = 1000
    model = AnomalyDetector(vocab_size=vocab_size, hidden_size=128, num_layers=2, num_heads=2)
    
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    outputs = model(input_ids, attention_mask)
    
    assert 'anomaly_score' in outputs
    assert 'embeddings' in outputs
    assert outputs['anomaly_score'].shape == (batch_size,)
    assert outputs['embeddings'].shape == (batch_size, 128)
    assert torch.all(outputs['anomaly_score'] >= 0.0)
    assert torch.all(outputs['anomaly_score'] <= 1.0)


def test_model_predict():
    """Test model prediction"""
    vocab_size = 1000
    model = AnomalyDetector(vocab_size=vocab_size, hidden_size=128, num_layers=2, num_heads=2)
    
    seq_length = 128
    input_ids = torch.randint(0, vocab_size, (1, seq_length))
    attention_mask = torch.ones(1, seq_length)
    
    score = model.predict(input_ids, attention_mask)
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_model_predict_batch():
    """Test batch prediction"""
    vocab_size = 1000
    model = AnomalyDetector(vocab_size=vocab_size, hidden_size=128, num_layers=2, num_heads=2)
    
    batch_size = 4
    seq_length = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    scores = model.predict_batch(input_ids, attention_mask)
    
    assert scores.shape == (batch_size,)
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 1.0)


def test_anomaly_scorer():
    """Test anomaly scorer"""
    vocab_size = 1000
    model = AnomalyDetector(vocab_size=vocab_size, hidden_size=128, num_layers=2, num_heads=2)
    scorer = AnomalyScorer(model, threshold=0.5)
    
    seq_length = 128
    input_ids = torch.randint(0, vocab_size, (1, seq_length))
    attention_mask = torch.ones(1, seq_length)
    
    result = scorer.score(input_ids, attention_mask)
    
    assert 'anomaly_score' in result
    assert 'is_anomaly' in result
    assert 'threshold' in result
    assert isinstance(result['anomaly_score'], float)
    assert isinstance(result['is_anomaly'], bool)
    assert 0.0 <= result['anomaly_score'] <= 1.0


def test_anomaly_scorer_batch():
    """Test batch scoring"""
    vocab_size = 1000
    model = AnomalyDetector(vocab_size=vocab_size, hidden_size=128, num_layers=2, num_heads=2)
    scorer = AnomalyScorer(model, threshold=0.5)
    
    batch_size = 4
    seq_length = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    result = scorer.score_batch(input_ids, attention_mask)
    
    assert 'anomaly_scores' in result
    assert 'is_anomaly' in result
    assert 'threshold' in result
    assert len(result['anomaly_scores']) == batch_size
    assert len(result['is_anomaly']) == batch_size
    assert all(0.0 <= score <= 1.0 for score in result['anomaly_scores'])


def test_anomaly_scorer_threshold():
    """Test threshold adjustment"""
    vocab_size = 1000
    model = AnomalyDetector(vocab_size=vocab_size, hidden_size=128, num_layers=2, num_heads=2)
    scorer = AnomalyScorer(model, threshold=0.5)
    
    assert scorer.threshold == 0.5
    
    scorer.set_threshold(0.7)
    assert scorer.threshold == 0.7


def test_model_different_sequence_lengths():
    """Test model with different sequence lengths"""
    vocab_size = 1000
    model = AnomalyDetector(vocab_size=vocab_size, hidden_size=128, num_layers=2, num_heads=2)
    
    # Test with shorter sequence
    input_ids_short = torch.randint(0, vocab_size, (2, 64))
    attention_mask_short = torch.ones(2, 64)
    outputs_short = model(input_ids_short, attention_mask_short)
    assert outputs_short['anomaly_score'].shape == (2,)
    
    # Test with longer sequence (within max_length)
    input_ids_long = torch.randint(0, vocab_size, (2, 256))
    attention_mask_long = torch.ones(2, 256)
    outputs_long = model(input_ids_long, attention_mask_long)
    assert outputs_long['anomaly_score'].shape == (2,)


def test_model_gradient_flow():
    """Test that gradients flow through the model"""
    vocab_size = 1000
    model = AnomalyDetector(vocab_size=vocab_size, hidden_size=128, num_layers=2, num_heads=2)
    
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    outputs = model(input_ids, attention_mask)
    loss = outputs['anomaly_score'].mean()
    loss.backward()
    
    # Check that gradients exist
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    assert has_gradients, "Gradients should flow through the model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
