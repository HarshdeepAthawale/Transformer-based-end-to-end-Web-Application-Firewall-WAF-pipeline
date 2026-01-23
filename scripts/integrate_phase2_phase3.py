#!/usr/bin/env python3
"""
Integrate Phase 2 and Phase 3

Demonstrates end-to-end flow from log ingestion to parsing/normalization
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.ingestion import LogIngestionSystem
from src.parsing.pipeline import ParsingPipeline
from loguru import logger


def main():
    """Demonstrate Phase 2 -> Phase 3 integration"""
    logger.info("=" * 60)
    logger.info("Phase 2 -> Phase 3 Integration")
    logger.info("=" * 60)
    
    # Initialize systems
    ingestion = LogIngestionSystem(config_path="config/config.yaml")
    pipeline = ParsingPipeline()
    
    # Get log path
    log_path = ingestion.config.get('web_server', {}).get('log_path', '/var/log/nginx/access.log')
    
    # Use sample log if main log not accessible
    sample_log = project_root / "data" / "raw" / "final_test.log"
    if not Path(log_path).exists() and sample_log.exists():
        log_path = str(sample_log)
    
    logger.info(f"Processing logs from: {log_path}")
    logger.info("")
    
    # Process logs: Ingestion -> Parsing -> Normalization
    count = 0
    for raw_line in ingestion.ingest_batch(log_path, max_lines=10):
        count += 1
        logger.info(f"Raw log #{count}: {raw_line[:80]}...")
        
        # Parse and normalize
        normalized = pipeline.process_log_line(raw_line)
        if normalized:
            logger.info(f"Normalized: {normalized[:80]}...")
        else:
            logger.warning("Failed to parse/normalize")
        
        logger.info("")
    
    logger.info(f"✓ Processed {count} log lines")
    logger.info("✓ Phase 2 -> Phase 3 integration working!")


if __name__ == "__main__":
    main()
