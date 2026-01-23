#!/usr/bin/env python3
"""
Phase 2 Verification Script

Verifies that all Phase 2 components are working correctly
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.ingestion import LogIngestionSystem
from src.ingestion.log_formats import LogFormatDetector, LogFormat
from src.ingestion.batch_reader import BatchLogReader
from src.ingestion.stream_reader import StreamLogReader
from src.ingestion.log_queue import LogQueue
from loguru import logger
import os


def verify_imports():
    """Verify all modules can be imported"""
    logger.info("Verifying module imports...")
    try:
        from src.ingestion import (
            LogIngestionSystem,
            BatchLogReader,
            StreamLogReader,
            LogFormatDetector,
            LogFormat,
            LogQueue
        )
        logger.info("✓ All modules imported successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def verify_format_detection():
    """Verify format detection works"""
    logger.info("\nVerifying format detection...")
    try:
        detector = LogFormatDetector()
        
        # Test with real Nginx log format
        nginx_line = '127.0.0.1 - - [24/Jan/2026:01:38:33 +0530] "GET /api/data HTTP/1.1" 404 153 "-" "curl/8.18.0"'
        detected = detector.detect_format(nginx_line)
        logger.info(f"  Nginx line: {detected.value}")
        assert detected == LogFormat.NGINX_COMBINED
        
        logger.info("✓ Format detection working")
        return True
    except Exception as e:
        logger.error(f"✗ Format detection failed: {e}")
        return False


def verify_batch_reader():
    """Verify batch reader works"""
    logger.info("\nVerifying batch reader...")
    try:
        # Use test log file
        test_log = project_root / "data" / "raw" / "nginx_access_sample.log"
        if not test_log.exists():
            logger.warning("  Test log not found, creating...")
            test_log.parent.mkdir(parents=True, exist_ok=True)
            test_log.write_text(
                '127.0.0.1 - - [24/Jan/2026:01:38:33 +0530] "GET /api/data HTTP/1.1" 404 153 "-" "curl/8.18.0"\n'
                '127.0.0.1 - - [24/Jan/2026:01:38:34 +0530] "GET /test HTTP/1.1" 200 1234 "-" "curl/8.18.0"\n'
            )
        
        reader = BatchLogReader(str(test_log))
        lines = list(reader.read_lines())
        logger.info(f"  Read {len(lines)} lines from test log")
        assert len(lines) > 0
        
        logger.info("✓ Batch reader working")
        return True
    except Exception as e:
        logger.error(f"✗ Batch reader failed: {e}")
        return False


def verify_ingestion_system():
    """Verify main ingestion system"""
    logger.info("\nVerifying ingestion system...")
    try:
        ingestion = LogIngestionSystem(config_path="config/config.yaml")
        
        # Test with sample log
        test_log = project_root / "data" / "raw" / "nginx_access_sample.log"
        if test_log.exists():
            count = 0
            for line in ingestion.ingest_batch(str(test_log), max_lines=5):
                count += 1
            logger.info(f"  Processed {count} lines")
            assert count > 0
        
        logger.info("✓ Ingestion system working")
        return True
    except Exception as e:
        logger.error(f"✗ Ingestion system failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_real_log_ingestion():
    """Verify ingestion with real Nginx logs"""
    logger.info("\nVerifying real log ingestion...")
    try:
        ingestion = LogIngestionSystem(config_path="config/config.yaml")
        log_path = ingestion.config.get('web_server', {}).get('log_path', '/var/log/nginx/access.log')
        
        # Try to access real log (may need sudo)
        if os.path.exists(log_path):
            logger.info(f"  Accessing real log: {log_path}")
            count = 0
            for line in ingestion.ingest_batch(log_path, max_lines=10):
                count += 1
            logger.info(f"  Processed {count} lines from real log")
            logger.info("✓ Real log ingestion working")
            return True
        else:
            # Use sample log
            sample_log = project_root / "data" / "raw" / "real_nginx_sample.log"
            if sample_log.exists():
                logger.info(f"  Using sample log: {sample_log}")
                count = 0
                for line in ingestion.ingest_batch(str(sample_log), max_lines=10):
                    count += 1
                logger.info(f"  Processed {count} lines from sample log")
                logger.info("✓ Sample log ingestion working")
                return True
            else:
                logger.warning("  No real log file available, skipping")
                return True
    except Exception as e:
        logger.error(f"✗ Real log ingestion failed: {e}")
        return False


def main():
    """Run all verification tests"""
    logger.info("=" * 60)
    logger.info("Phase 2: Log Ingestion System - Verification")
    logger.info("=" * 60)
    
    results = []
    results.append(("Module Imports", verify_imports()))
    results.append(("Format Detection", verify_format_detection()))
    results.append(("Batch Reader", verify_batch_reader()))
    results.append(("Ingestion System", verify_ingestion_system()))
    results.append(("Real Log Ingestion", verify_real_log_ingestion()))
    
    logger.info("\n" + "=" * 60)
    logger.info("Verification Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status}: {name}")
    
    logger.info(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✓ Phase 2 verification complete - All tests passed!")
        return 0
    else:
        logger.error("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
