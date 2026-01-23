#!/usr/bin/env python3
"""
Test Tokenization System

Tests the tokenization system with real normalized data
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.ingestion import LogIngestionSystem
from src.parsing.pipeline import ParsingPipeline
from src.tokenization.tokenizer import HTTPTokenizer
from src.tokenization.sequence_prep import SequencePreparator
from src.tokenization.dataloader import create_dataloader
from loguru import logger


def test_tokenizer_basic():
    """Test basic tokenizer functionality"""
    logger.info("Testing basic tokenizer...")
    
    tokenizer = HTTPTokenizer(vocab_size=1000, min_frequency=1)
    texts = [
        "GET /api/users/123 HTTP/1.1",
        "POST /api/login HTTP/1.1",
        "GET /test?param=value HTTP/1.1"
    ]
    
    tokenizer.build_vocab(texts)
    assert tokenizer.vocab_built, "Vocabulary not built"
    assert len(tokenizer.word_to_id) > 0, "Empty vocabulary"
    
    # Test encoding
    text = "GET /api/users/123 HTTP/1.1"
    token_ids = tokenizer.encode(text)
    assert len(token_ids) > 0, "Empty token IDs"
    
    logger.info(f"✓ Tokenizer: {len(tokenizer.word_to_id)} tokens in vocab")
    logger.info(f"✓ Encoded text to {len(token_ids)} tokens")


def test_sequence_preparation():
    """Test sequence preparation"""
    logger.info("\nTesting sequence preparation...")
    
    tokenizer = HTTPTokenizer(vocab_size=1000, min_frequency=1)
    texts = ["GET /test HTTP/1.1", "POST /api HTTP/1.1"]
    tokenizer.build_vocab(texts)
    
    preparator = SequencePreparator(tokenizer)
    token_ids, attention_mask = preparator.prepare_sequence(
        "GET /test HTTP/1.1",
        max_length=128
    )
    
    assert len(token_ids) == 128, "Wrong sequence length"
    assert len(attention_mask) == 128, "Wrong attention mask length"
    assert sum(attention_mask) > 0, "All tokens are padding"
    
    logger.info(f"✓ Sequence prepared: {len(token_ids)} tokens, {sum(attention_mask)} non-padding")


def test_integration_with_parsing():
    """Test integration with parsing pipeline"""
    logger.info("\nTesting integration with parsing pipeline...")
    
    # Initialize systems
    ingestion = LogIngestionSystem(config_path="config/config.yaml")
    pipeline = ParsingPipeline()
    tokenizer = HTTPTokenizer(vocab_size=1000, min_frequency=1)
    
    # Get normalized texts
    log_path = project_root / "data" / "raw" / "comprehensive_parsing_test.log"
    if not log_path.exists():
        logger.warning("Test log not found, creating sample...")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            '127.0.0.1 - - [24/Jan/2026:01:38:33 +0530] "GET /api/data HTTP/1.1" 404 153 "-" "curl/8.18.0"\n'
            '127.0.0.1 - - [24/Jan/2026:01:38:34 +0530] "GET /test HTTP/1.1" 200 1234 "-" "curl/8.18.0"\n'
        )
    
    texts = []
    for line in ingestion.ingest_batch(str(log_path), max_lines=10):
        normalized = pipeline.process_log_line(line)
        if normalized:
            texts.append(normalized)
    
    assert len(texts) > 0, "No texts collected"
    
    # Build vocabulary
    tokenizer.build_vocab(texts)
    
    # Test encoding
    encoded = tokenizer.encode(texts[0])
    assert len(encoded) > 0, "Failed to encode"
    
    logger.info(f"✓ Integrated: {len(texts)} texts, {len(tokenizer.word_to_id)} vocab tokens")
    logger.info(f"✓ Sample encoding: {len(encoded)} tokens")


def test_dataloader():
    """Test DataLoader"""
    logger.info("\nTesting DataLoader...")
    
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
    
    batch = next(iter(dataloader))
    assert 'input_ids' in batch, "Missing input_ids"
    assert 'attention_mask' in batch, "Missing attention_mask"
    assert batch['input_ids'].shape[0] <= 2, "Wrong batch size"
    
    logger.info(f"✓ DataLoader: batch shape {batch['input_ids'].shape}")


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Tokenization System - Test Suite")
    logger.info("=" * 60)
    
    try:
        test_tokenizer_basic()
        test_sequence_preparation()
        test_integration_with_parsing()
        test_dataloader()
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ All tests passed!")
        logger.info("=" * 60)
        return 0
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
