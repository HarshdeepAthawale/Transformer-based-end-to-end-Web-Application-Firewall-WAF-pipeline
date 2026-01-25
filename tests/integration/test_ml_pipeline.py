"""
Integration Tests for ML Pipeline

Test the complete ML pipeline from tokenization to inference.
"""
import pytest
import torch
from pathlib import Path
import json
import tempfile

from backend.ml.tokenization.tokenizer import HTTPTokenizer
from backend.ml.tokenization.sequence_prep import SequencePreparator
from backend.ml.model.anomaly_detector import AnomalyDetector
from backend.ml.model.scoring import AnomalyScorer
from backend.ml.parsing.pipeline import ParsingPipeline


class TestTokenization:
    """Test tokenization pipeline"""
    
    def test_tokenizer_build_vocab(self):
        """Test vocabulary building"""
        texts = [
            "GET /api/users HTTP/1.1",
            "POST /api/login HTTP/1.1",
            "GET /dashboard?page=1 HTTP/1.1"
        ]
        
        tokenizer = HTTPTokenizer(vocab_size=1000)
        tokenizer.build_vocab(texts)
        
        assert tokenizer.vocab_built
        assert len(tokenizer.word_to_id) > 0
        assert HTTPTokenizer.PAD_TOKEN in tokenizer.word_to_id
        assert HTTPTokenizer.UNK_TOKEN in tokenizer.word_to_id
    
    def test_tokenizer_encode_decode(self):
        """Test encoding and decoding"""
        texts = ["GET /api/users HTTP/1.1"]
        
        tokenizer = HTTPTokenizer(vocab_size=1000)
        tokenizer.build_vocab(texts)
        
        token_ids = tokenizer.encode(texts[0])
        decoded = tokenizer.decode(token_ids)
        
        assert len(token_ids) > 0
        assert isinstance(decoded, str)
    
    def test_sequence_preparation(self):
        """Test sequence preparation"""
        texts = ["GET /api/users HTTP/1.1"]
        
        tokenizer = HTTPTokenizer(vocab_size=1000, max_length=128)
        tokenizer.build_vocab(texts)
        
        preparator = SequencePreparator(tokenizer)
        token_ids, attention_mask = preparator.prepare_sequence(texts[0])
        
        assert len(token_ids) == 128
        assert len(attention_mask) == 128
        assert sum(attention_mask) > 0  # At least some real tokens


class TestModel:
    """Test model architecture"""
    
    def test_model_creation(self):
        """Test model creation"""
        vocab_size = 1000
        model = AnomalyDetector(vocab_size=vocab_size)
        
        assert model is not None
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'anomaly_head')
    
    def test_model_forward(self):
        """Test model forward pass"""
        vocab_size = 1000
        model = AnomalyDetector(vocab_size=vocab_size)
        model.eval()
        
        batch_size = 2
        seq_len = 128
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        
        assert 'anomaly_score' in outputs
        assert outputs['anomaly_score'].shape == (batch_size,)
        assert torch.all((outputs['anomaly_score'] >= 0) & (outputs['anomaly_score'] <= 1))
    
    def test_model_save_load(self):
        """Test model checkpoint saving and loading"""
        vocab_size = 1000
        model = AnomalyDetector(vocab_size=vocab_size)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            # Save
            model.save_checkpoint(checkpoint_path, vocab_size=vocab_size)
            assert Path(checkpoint_path).exists()
            
            # Load
            loaded_model = AnomalyDetector.load_checkpoint(checkpoint_path)
            assert loaded_model is not None
            
            # Test forward pass
            input_ids = torch.randint(0, vocab_size, (1, 128))
            attention_mask = torch.ones(1, 128)
            
            with torch.no_grad():
                outputs = loaded_model(input_ids, attention_mask)
            
            assert 'anomaly_score' in outputs
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)


class TestScoring:
    """Test anomaly scoring"""
    
    def test_scorer_initialization(self):
        """Test scorer initialization"""
        vocab_size = 1000
        model = AnomalyDetector(vocab_size=vocab_size)
        scorer = AnomalyScorer(model, threshold=0.5)
        
        assert scorer.model == model
        assert scorer.threshold == 0.5
    
    def test_scorer_score(self):
        """Test scoring"""
        vocab_size = 1000
        model = AnomalyDetector(vocab_size=vocab_size)
        model.eval()
        scorer = AnomalyScorer(model, threshold=0.5)
        
        input_ids = torch.randint(0, vocab_size, (1, 128))
        attention_mask = torch.ones(1, 128)
        
        result = scorer.score(input_ids, attention_mask)
        
        assert 'anomaly_score' in result
        assert 'is_anomaly' in result
        assert isinstance(result['anomaly_score'], float)
        assert isinstance(result['is_anomaly'], bool)


class TestPipeline:
    """Test complete pipeline"""
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline"""
        # Sample log line
        log_line = '127.0.0.1 - - [25/Dec/2023:10:00:00 +0000] "GET /api/users HTTP/1.1" 200 1234'
        
        # Parse
        parser = ParsingPipeline()
        normalized = parser.process_log_line(log_line)
        assert normalized is not None
        
        # Tokenize
        texts = [normalized]
        tokenizer = HTTPTokenizer(vocab_size=1000)
        tokenizer.build_vocab(texts)
        
        preparator = SequencePreparator(tokenizer)
        token_ids, attention_mask = preparator.prepare_sequence(normalized)
        
        # Model inference
        vocab_size = len(tokenizer.word_to_id)
        model = AnomalyDetector(vocab_size=vocab_size)
        model.eval()
        
        input_ids_tensor = torch.tensor([token_ids])
        attn_mask_tensor = torch.tensor([attention_mask])
        
        with torch.no_grad():
            outputs = model(input_ids_tensor, attn_mask_tensor)
        
        assert 'anomaly_score' in outputs
        assert outputs['anomaly_score'].shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
