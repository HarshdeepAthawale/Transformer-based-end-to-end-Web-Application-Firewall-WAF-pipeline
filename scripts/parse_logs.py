#!/usr/bin/env python3
"""
Parse and Normalize Logs

Processes logs through parsing and normalization pipeline
"""
import sys
from pathlib import Path
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parsing.pipeline import ParsingPipeline
from src.ingestion.ingestion import LogIngestionSystem
from loguru import logger
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Parse and normalize logs")
    parser.add_argument("--log_path", type=str, help="Path to log file", default=None)
    parser.add_argument("--output", type=str, help="Output file path", default=None)
    parser.add_argument("--max_lines", type=int, help="Maximum lines to process", default=None)
    parser.add_argument("--mode", type=str, choices=['batch', 'stream'], default='batch')
    
    args = parser.parse_args()
    
    # Initialize systems
    ingestion = LogIngestionSystem(config_path="config/config.yaml")
    pipeline = ParsingPipeline()
    
    # Get log path
    if args.log_path:
        log_path = args.log_path
    else:
        log_path = ingestion.config.get('web_server', {}).get('log_path', '/var/log/nginx/access.log')
    
    logger.info("=" * 60)
    logger.info("Log Parsing & Normalization")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_path}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Max lines: {args.max_lines or 'unlimited'}")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = project_root / "data" / "normalized" / f"normalized_{timestamp}.txt"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process logs
    processed = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        if args.mode == 'batch':
            # Batch mode
            for line in ingestion.ingest_batch(log_path, max_lines=args.max_lines):
                result = pipeline.process_log_line(line)
                if result:
                    f.write(result + '\n')
                    processed += 1
                    
                    if processed % 100 == 0:
                        logger.info(f"Processed {processed} lines...")
        else:
            # Stream mode
            logger.info("Streaming mode - press Ctrl+C to stop")
            try:
                for line in ingestion.ingest_stream(log_path, follow=True):
                    result = pipeline.process_log_line(line)
                    if result:
                        f.write(result + '\n')
                        f.flush()
                        processed += 1
                        
                        if processed % 10 == 0:
                            logger.info(f"Processed {processed} lines...")
                        
                        if args.max_lines and processed >= args.max_lines:
                            break
            except KeyboardInterrupt:
                logger.info("Streaming interrupted by user")
    
    logger.info(f"✓ Processing complete: {processed} lines")
    logger.info(f"✓ Output saved to: {output_path}")
    logger.info(f"✓ File size: {output_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
