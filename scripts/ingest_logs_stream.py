#!/usr/bin/env python3
"""
Streaming Log Ingestion Script

Tails log file in real-time and processes new entries
"""
import sys
from pathlib import Path
import signal

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ml.ingestion.ingestion import LogIngestionSystem
from loguru import logger
import argparse


def main():
    parser = argparse.ArgumentParser(description="Streaming log ingestion")
    parser.add_argument("--log_path", type=str, help="Path to log file", default=None)
    parser.add_argument("--output", type=str, help="Output file path", default=None)
    parser.add_argument("--max_lines", type=int, help="Maximum lines to process", default=None)
    
    args = parser.parse_args()
    
    # Initialize ingestion system
    ingestion = LogIngestionSystem(config_path="config/config.yaml")
    
    # Get log path
    if args.log_path:
        log_path = args.log_path
    else:
        log_path = ingestion.config.get('web_server', {}).get('log_path', '/var/log/nginx/access.log')
    
    logger.info("=" * 60)
    logger.info("Streaming Log Ingestion")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_path}")
    logger.info("Press Ctrl+C to stop")
    
    # Determine output path
    output_file = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = open(output_path, 'w', encoding='utf-8')
        logger.info(f"Output file: {output_path}")
    
    # Handle interrupt
    def signal_handler(sig, frame):
        logger.info("\nStopping stream ingestion...")
        if output_file:
            output_file.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Stream logs
    processed = 0
    try:
        for line in ingestion.ingest_stream(log_path, follow=True):
            processed += 1
            
            if output_file:
                output_file.write(line + '\n')
                output_file.flush()
            
            logger.info(f"Line #{processed}: {line[:80]}...")
            
            if args.max_lines and processed >= args.max_lines:
                logger.info(f"Reached max lines limit: {args.max_lines}")
                break
                
    except KeyboardInterrupt:
        logger.info("Streaming interrupted by user")
    finally:
        if output_file:
            output_file.close()
    
    logger.info(f"✓ Stream ingestion complete: {processed} lines processed")


if __name__ == "__main__":
    main()
