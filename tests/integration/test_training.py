"""
Training Pipeline Tests

Test training pipeline components.
"""
import pytest
import torch
from pathlib import Path
import tempfile
import json

from backend.ml.tokenization.tokenizer import HTTPTokenizer
from backend.ml.tokenization.dataloader import create_dataloader
from backend.ml.model.anomaly_detector import AnomalyDetector
from backend.ml.training.train import AnomalyDetectionTrainer
from backend.ml.training.dataset_generator import DatasetGenerator
from backend.ml.training.evaluator import ModelEvaluator


class TestDatasetGenerator:
    """Test dataset generation"""
    
    def test_load_data(self):
        """Test loading data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = ["text1", "text2", "text3"]
            json.dump(data, f)
            data_path = f.name
        
        try:
            texts = DatasetGenerator.load_data(data_path)
            assert len(texts) == 3
            assert texts == data
        finally:
            Path(data_path).unlink()
    
    def test_split_data(self):
        """Test data splitting"""
        texts = [f"text{i}" for i in range(100)]
        train, val, test = DatasetGenerator.split_data(texts)
        
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == 100


class TestTraining:
    """Test training components"""
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        vocab_size = 1000
        model = AnomalyDetector(vocab_size=vocab_size)
        tokenizer = HTTPTokenizer(vocab_size=vocab_size)
        tokenizer.build_vocab(["GET /test HTTP/1.1"])
        
        trainer = AnomalyDetectionTrainer(model, tokenizer)
        assert trainer.model == model
        assert trainer.tokenizer == tokenizer
    
    def test_training_step(self):
        """Test single training step"""
        vocab_size = 1000
        texts = ["GET /test HTTP/1.1"] * 10
        
        tokenizer = HTTPTokenizer(vocab_size=vocab_size)
        tokenizer.build_vocab(texts)
        
        train_loader = create_dataloader(texts, tokenizer, batch_size=2, shuffle=False)
        
        model = AnomalyDetector(vocab_size=len(tokenizer.word_to_id))
        trainer = AnomalyDetectionTrainer(model, tokenizer)
        
        # Single epoch
        loss = trainer._train_epoch(train_loader)
        assert isinstance(loss, float)
        assert loss >= 0


class TestEvaluator:
    """Test evaluation components"""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        vocab_size = 1000
        model = AnomalyDetector(vocab_size=vocab_size)
        tokenizer = HTTPTokenizer(vocab_size=vocab_size)
        tokenizer.build_vocab(["GET /test HTTP/1.1"])
        
        evaluator = ModelEvaluator(model, tokenizer)
        assert evaluator.model == model
        assert evaluator.tokenizer == tokenizer
    
    def test_evaluation_metrics(self):
        """Test metric calculation"""
        vocab_size = 1000
        texts = ["GET /test HTTP/1.1"] * 10
        labels = [0.0] * 10  # All benign
        
        tokenizer = HTTPTokenizer(vocab_size=vocab_size)
        tokenizer.build_vocab(texts)
        
        model = AnomalyDetector(vocab_size=len(tokenizer.word_to_id))
        evaluator = ModelEvaluator(model, tokenizer)
        
        metrics = evaluator.evaluate_on_texts(texts, labels, threshold=0.5)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'fpr' in metrics
        assert 'tpr' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
