#!/usr/bin/env python3
"""
Log Ingestion Example

Demonstrates batch and streaming log ingestion with real Nginx logs
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.ingestion import LogIngestionSystem
from loguru import logger
import time


def process_log_line(line: str):
    """Process a single log line"""
    # This is where you would parse, normalize, and process the log
    logger.debug(f"Processing line: {line[:100]}...")


def main():
    """Main example function"""
    # Initialize ingestion system
    ingestion = LogIngestionSystem(config_path="config/config.yaml")
    
    # Get log path from config
    log_path = ingestion.config.get('web_server', {}).get('log_path', '/var/log/nginx/access.log')
    
    logger.info("=" * 60)
    logger.info("Log Ingestion System - Example Usage")
    logger.info("=" * 60)
    
    # Example 1: Batch ingestion
    logger.info("\nExample 1: Batch Ingestion")
    logger.info("-" * 60)
    
    count = 0
    for line in ingestion.ingest_batch(log_path, max_lines=100):
        count += 1
        if count % 10 == 0:
            logger.info(f"Processed {count} lines")
    
    logger.info(f"Batch ingestion complete: {count} lines processed")
    
    # Example 2: Streaming ingestion (limited time)
    logger.info("\nExample 2: Streaming Ingestion (10 seconds)")
    logger.info("-" * 60)
    logger.info("Starting stream ingestion (will run for 10 seconds)...")
    
    start_time = time.time()
    count = 0
    try:
        for line in ingestion.ingest_stream(log_path, follow=True):
            count += 1
            logger.info(f"New log line #{count}: {line[:80]}...")
            
            # Stop after 10 seconds for demo
            if time.time() - start_time > 10:
                logger.info("Stopping stream ingestion after 10 seconds")
                break
    except KeyboardInterrupt:
        logger.info("Streaming stopped by user")
    
    logger.info(f"Stream ingestion complete: {count} lines processed")
    
    # Example 3: Streaming with queue
    logger.info("\nExample 3: Streaming with Queue (5 seconds)")
    logger.info("-" * 60)
    
    def queue_processor(line: str):
        """Process lines from queue"""
        logger.debug(f"Queue processed: {line[:60]}...")
    
    ingestion.queue.processor = queue_processor
    
    # Start streaming in background (will be interrupted)
    import threading
    stream_thread = threading.Thread(
        target=ingestion.start_streaming_with_queue,
        args=(log_path,),
        daemon=True
    )
    stream_thread.start()
    
    # Let it run for 5 seconds
    time.sleep(5)
    
    # Stop the queue processor
    ingestion.queue.stop_processor()
    logger.info(f"Queue processing complete. Queue size: {ingestion.queue.size()}")


if __name__ == "__main__":
    main()
