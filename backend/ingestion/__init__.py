"""
Log Ingestion Pipeline

Batch and streaming ingestion from Apache/Nginx access logs
with format auto-detection and async queue for downstream processing.
"""

from backend.ingestion.format_detector import LogFormat, detect_format, detect_from_file
from backend.ingestion.batch_reader import read_chunks
from backend.ingestion.stream_tailer import tail_lines
from backend.ingestion.queue import IngestionQueue
from backend.ingestion.retry import RetryConfig, retry, with_retry_sync, with_retry_async
from backend.ingestion.config import (
    BatchConfig,
    StreamingConfig,
    IngestionConfig,
    load_ingestion_config,
)

__all__ = [
    "LogFormat",
    "detect_format",
    "detect_from_file",
    "read_chunks",
    "tail_lines",
    "IngestionQueue",
    "RetryConfig",
    "retry",
    "with_retry_sync",
    "with_retry_async",
    "BatchConfig",
    "StreamingConfig",
    "IngestionConfig",
    "load_ingestion_config",
]
