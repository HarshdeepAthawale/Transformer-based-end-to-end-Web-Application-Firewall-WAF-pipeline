"""
Unit tests for tokenization system
"""
import pytest
from src.tokenization.tokenizer import HTTPTokenizer
from src.tokenization.sequence_prep import SequencePreparator
from src.tokenization.dataloader import HTTPRequestDataset, create_dataloader
import torch


def test_tokenizer():
    """Test tokenizer functionality"""
    tokenizer = HTTPTokenizer(vocab_size=1000, min_frequency=1)
    
    # Build vocab from sample texts
    texts = [
        "GET /api/users/123 HTTP/1.1",
        "POST /api/login HTTP/1.1",
        "GET /test?param=value HTTP/1.1"
    ]
    
    tokenizer.build_vocab(texts)
    assert tokenizer.vocab_built
    assert len(tokenizer.word_to_id) > 0
    
    # Test encoding
    text = "GET /api/users/123 HTTP/1.1"
    token_ids = tokenizer.encode(text)
    assert len(token_ids) > 0
    assert token_ids[0] == tokenizer.word_to_id[tokenizer.CLS_TOKEN]  # CLS token first
    
    # Test decoding
    decoded = tokenizer.decode(token_ids)
    assert "GET" in decoded or "get" in decoded


def test_sequence_preparation():
    """Test sequence preparation"""
    tokenizer = HTTPTokenizer(vocab_size=1000, min_frequency=1)
    texts = ["GET /test HTTP/1.1", "POST /api HTTP/1.1"]
    tokenizer.build_vocab(texts)
    
    preparator = SequencePreparator(tokenizer)
    token_ids, attention_mask = preparator.prepare_sequence(
        "GET /test HTTP/1.1",
        max_length=128
    )
    
    assert len(token_ids) == 128
    assert len(attention_mask) == 128
    assert sum(attention_mask) > 0  # At least some tokens are not padding


def test_batch_preparation():
    """Test batch preparation"""
    tokenizer = HTTPTokenizer(vocab_size=1000, min_frequency=1)
    texts = ["GET /test HTTP/1.1", "POST /api HTTP/1.1", "GET /users HTTP/1.1"]
    tokenizer.build_vocab(texts)
    
    preparator = SequencePreparator(tokenizer)
    batch = preparator.prepare_batch(
        ["GET /test HTTP/1.1", "POST /api HTTP/1.1"],
        max_length=128,
        return_tensors="pt"
    )
    
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert batch['input_ids'].shape[0] == 2  # Batch size
    assert batch['input_ids'].shape[1] == 128  # Sequence length


def test_dataset():
    """Test HTTPRequestDataset"""
    tokenizer = HTTPTokenizer(vocab_size=1000, min_frequency=1)
    texts = ["GET /test HTTP/1.1", "POST /api HTTP/1.1"]
    tokenizer.build_vocab(texts)
    
    dataset = HTTPRequestDataset(texts, tokenizer, max_length=128)
    assert len(dataset) == 2
    
    sample = dataset[0]
    assert 'input_ids' in sample
    assert 'attention_mask' in sample
    assert 'text' in sample
    assert sample['input_ids'].shape[0] == 128


def test_dataloader():
    """Test DataLoader creation"""
    tokenizer = HTTPTokenizer(vocab_size=1000, min_frequency=1)
    texts = ["GET /test HTTP/1.1", "POST /api HTTP/1.1", "GET /users HTTP/1.1"]
    tokenizer.build_vocab(texts)
    
    dataloader = create_dataloader(
        texts,
        tokenizer,
        batch_size=2,
        max_length=128,
        shuffle=False,
        num_workers=0
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert 'text' in batch
    assert batch['input_ids'].shape[0] <= 2  # Batch size


def test_vocab_save_load():
    """Test vocabulary save and load"""
    tokenizer = HTTPTokenizer(vocab_size=1000, min_frequency=1)
    texts = ["GET /test HTTP/1.1", "POST /api HTTP/1.1"]
    tokenizer.build_vocab(texts)
    
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        vocab_path = f.name
    
    try:
        tokenizer.save_vocab(vocab_path)
        
        # Create new tokenizer and load
        new_tokenizer = HTTPTokenizer(vocab_size=1000, min_frequency=1)
        new_tokenizer.load_vocab(vocab_path)
        
        assert new_tokenizer.vocab_built
        assert len(new_tokenizer.word_to_id) == len(tokenizer.word_to_id)
        
        # Test encoding with loaded vocab
        text = "GET /test HTTP/1.1"
        ids1 = tokenizer.encode(text)
        ids2 = new_tokenizer.encode(text)
        assert ids1 == ids2
    finally:
        os.unlink(vocab_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
