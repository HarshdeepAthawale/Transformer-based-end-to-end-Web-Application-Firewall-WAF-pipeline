#!/usr/bin/env python3
"""
Integrate Phase 3 and Phase 4

Demonstrates end-to-end flow from parsing to tokenization
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.ingestion import LogIngestionSystem
from src.parsing.pipeline import ParsingPipeline
from src.tokenization.tokenizer import HTTPTokenizer
from src.tokenization.sequence_prep import SequencePreparator
from loguru import logger


def main():
    """Demonstrate Phase 3 -> Phase 4 integration"""
    logger.info("=" * 60)
    logger.info("Phase 3 -> Phase 4 Integration")
    logger.info("=" * 60)
    
    # Initialize systems
    ingestion = LogIngestionSystem(config_path="config/config.yaml")
    pipeline = ParsingPipeline()
    tokenizer = HTTPTokenizer(vocab_size=1000, min_frequency=1)
    
    # Get log path
    log_path = project_root / "data" / "raw" / "comprehensive_parsing_test.log"
    if not log_path.exists():
        logger.error(f"Log file not found: {log_path}")
        return 1
    
    logger.info(f"Processing logs from: {log_path}")
    logger.info("")
    
    # Step 1: Parse and normalize (Phase 3)
    logger.info("Phase 3: Parsing & Normalization...")
    normalized_texts = []
    for raw_line in ingestion.ingest_batch(str(log_path), max_lines=10):
        normalized = pipeline.process_log_line(raw_line)
        if normalized:
            normalized_texts.append(normalized)
            logger.info(f"  Normalized: {normalized[:60]}...")
    
    logger.info(f"✓ Phase 3: {len(normalized_texts)} requests normalized")
    logger.info("")
    
    # Step 2: Build vocabulary (Phase 4)
    logger.info("Phase 4: Tokenization...")
    tokenizer.build_vocab(normalized_texts)
    logger.info(f"✓ Vocabulary built: {len(tokenizer.word_to_id)} tokens")
    logger.info("")
    
    # Step 3: Tokenize sequences
    logger.info("Tokenizing sequences...")
    preparator = SequencePreparator(tokenizer)
    
    for i, text in enumerate(normalized_texts[:3], 1):
        token_ids, attention_mask = preparator.prepare_sequence(text, max_length=128)
        logger.info(f"  Request {i}: {len([x for x in attention_mask if x > 0])} tokens")
        logger.info(f"    Token IDs: {token_ids[:10]}...")
    
    logger.info("")
    logger.info("✓ Phase 3 -> Phase 4 integration working!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
