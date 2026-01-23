#!/usr/bin/env python3
"""
Batch Log Ingestion Script

Ingests logs from file in batch mode and saves to data/raw directory
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.ingestion import LogIngestionSystem
from loguru import logger
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Batch log ingestion")
    parser.add_argument("--log_path", type=str, help="Path to log file", default=None)
    parser.add_argument("--max_lines", type=int, help="Maximum lines to process", default=None)
    parser.add_argument("--output", type=str, help="Output file path", default=None)
    parser.add_argument("--skip_lines", type=int, help="Lines to skip", default=0)
    
    args = parser.parse_args()
    
    # Initialize ingestion system
    ingestion = LogIngestionSystem(config_path="config/config.yaml")
    
    # Get log path
    if args.log_path:
        log_path = args.log_path
    else:
        log_path = ingestion.config.get('web_server', {}).get('log_path', '/var/log/nginx/access.log')
    
    logger.info("=" * 60)
    logger.info("Batch Log Ingestion")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_path}")
    logger.info(f"Max lines: {args.max_lines or 'unlimited'}")
    logger.info(f"Skip lines: {args.skip_lines}")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = project_root / "data" / "raw" / f"ingested_{timestamp}.log"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ingest logs
    processed = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in ingestion.ingest_batch(
            log_path,
            max_lines=args.max_lines,
            skip_lines=args.skip_lines
        ):
            f.write(line + '\n')
            processed += 1
            
            if processed % 1000 == 0:
                logger.info(f"Processed {processed} lines...")
    
    logger.info(f"✓ Batch ingestion complete: {processed} lines")
    logger.info(f"✓ Output saved to: {output_path}")
    logger.info(f"✓ File size: {output_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
