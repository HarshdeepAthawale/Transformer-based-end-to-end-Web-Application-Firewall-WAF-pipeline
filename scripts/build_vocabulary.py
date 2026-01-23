#!/usr/bin/env python3
"""
Build Vocabulary Script

Builds vocabulary from normalized HTTP request logs
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.ingestion import LogIngestionSystem
from src.parsing.pipeline import ParsingPipeline
from src.tokenization.tokenizer import HTTPTokenizer
from loguru import logger
import argparse


def build_vocabulary(
    log_path: str,
    vocab_size: int = 10000,
    min_frequency: int = 2,
    max_lines: int = None,
    output_path: str = None
):
    """Build vocabulary from log files"""
    logger.info("=" * 60)
    logger.info("Vocabulary Building Process")
    logger.info("=" * 60)
    logger.info(f"Log path: {log_path}")
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Min frequency: {min_frequency}")
    logger.info(f"Max lines: {max_lines or 'unlimited'}")
    
    # Initialize components
    ingestion = LogIngestionSystem(config_path="config/config.yaml")
    pipeline = ParsingPipeline()
    tokenizer = HTTPTokenizer(
        vocab_size=vocab_size,
        min_frequency=min_frequency
    )
    
    # Collect normalized texts
    texts = []
    logger.info("\nProcessing logs and collecting texts...")
    
    count = 0
    for log_line in ingestion.ingest_batch(log_path, max_lines=max_lines):
        normalized_text = pipeline.process_log_line(log_line)
        if normalized_text:
            texts.append(normalized_text)
            count += 1
            
            if count % 1000 == 0:
                logger.info(f"Processed {count} requests...")
    
    logger.info(f"\nTotal texts collected: {len(texts)}")
    
    if len(texts) == 0:
        logger.error("No texts collected. Cannot build vocabulary.")
        return 1
    
    # Build vocabulary
    if output_path is None:
        output_path = project_root / "models" / "vocabularies" / "http_vocab.json"
    
    logger.info(f"\nBuilding vocabulary from {len(texts)} texts...")
    tokenizer.build_vocab(texts, save_path=str(output_path))
    
    logger.info(f"\n✓ Vocabulary built and saved to {output_path}")
    logger.info(f"✓ Vocabulary size: {len(tokenizer.word_to_id)} tokens")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build vocabulary from logs")
    parser.add_argument("--log_path", type=str, help="Path to log file", default=None)
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--min_frequency", type=int, default=2, help="Minimum token frequency")
    parser.add_argument("--max_lines", type=int, default=None, help="Maximum lines to process")
    parser.add_argument("--output", type=str, default=None, help="Output vocabulary path")
    
    args = parser.parse_args()
    
    # Get log path
    if args.log_path:
        log_path = args.log_path
    else:
        ingestion = LogIngestionSystem(config_path="config/config.yaml")
        log_path = ingestion.config.get('web_server', {}).get('log_path', '/var/log/nginx/access.log')
    
    sys.exit(build_vocabulary(
        log_path,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        max_lines=args.max_lines,
        output_path=args.output
    ))
