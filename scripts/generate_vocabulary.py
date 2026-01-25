#!/usr/bin/env python3
"""
Generate Vocabulary from Training Data

Build vocabulary from normalized request strings.
"""
import argparse
import json
from pathlib import Path
from loguru import logger

from backend.ml.tokenization.tokenizer import HTTPTokenizer
from backend.ml.training.dataset_generator import DatasetGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate vocabulary from training data")
    parser.add_argument("--data", type=str, required=True, help="Path to training data JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output vocabulary file path")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum token frequency")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data}...")
    texts = DatasetGenerator.load_data(args.data)
    logger.info(f"Loaded {len(texts)} training samples")
    
    # Build vocabulary
    logger.info("Building vocabulary...")
    tokenizer = HTTPTokenizer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        max_length=args.max_length
    )
    tokenizer.build_vocab(texts, save_path=args.output)
    
    logger.info(f"Vocabulary saved to {args.output}")
    logger.info(f"Vocabulary size: {len(tokenizer.word_to_id)}")


if __name__ == "__main__":
    main()
