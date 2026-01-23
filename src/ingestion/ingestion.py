"""
Main Log Ingestion System

Orchestrates batch and streaming log ingestion with format detection and validation
"""
from typing import Optional, Iterator, Callable
from pathlib import Path
from loguru import logger
import yaml

from .batch_reader import BatchLogReader
from .stream_reader import StreamLogReader
from .log_formats import LogFormatDetector, LogFormat
from .log_queue import LogQueue
from .retry_handler import RetryHandler


class LogIngestionSystem:
    """Main log ingestion system"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.format_detector = LogFormatDetector()
        self.queue = LogQueue(maxsize=self.config.get('ingestion', {}).get('streaming', {}).get('buffer_size', 10000))
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_path}, using defaults")
                return {}
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config or {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def ingest_batch(
        self,
        log_path: str,
        max_lines: Optional[int] = None,
        skip_lines: int = 0,
        callback: Optional[Callable] = None
    ) -> Iterator[str]:
        """Ingest logs in batch mode"""
        logger.info(f"Starting batch ingestion from: {log_path}")
        
        # Apply config defaults
        batch_config = self.config.get('ingestion', {}).get('batch', {})
        if max_lines is None:
            max_lines = batch_config.get('max_lines')
        if skip_lines == 0:
            skip_lines = batch_config.get('skip_lines', 0)
        
        reader = BatchLogReader(
            log_path=log_path,
            format_detector=self.format_detector.detect_format,
            max_lines=max_lines,
            skip_lines=skip_lines
        )
        
        total_lines = reader.get_line_count()
        file_size = reader.get_file_size()
        logger.info(f"Log file: {log_path} ({file_size} bytes, ~{total_lines} lines)")
        
        # Detect format
        detected_format = self.format_detector.detect_from_file(log_path)
        logger.info(f"Detected log format: {detected_format.value}")
        
        processed = 0
        for line in reader.read_lines():
            processed += 1
            
            # Validate line
            if not self._validate_line(line):
                continue
            
            # Process or yield
            if callback:
                callback(line)
            else:
                yield line
            
            # Progress logging
            if processed % 10000 == 0:
                logger.info(f"Processed {processed}/{total_lines} lines ({processed*100//total_lines if total_lines > 0 else 0}%)")
        
        logger.info(f"Batch ingestion complete: {processed} lines processed")
    
    def ingest_stream(
        self,
        log_path: str,
        callback: Optional[Callable] = None,
        follow: bool = True
    ) -> Iterator[str]:
        """Ingest logs in streaming mode"""
        logger.info(f"Starting stream ingestion from: {log_path}")
        
        # Apply config defaults
        stream_config = self.config.get('ingestion', {}).get('streaming', {})
        poll_interval = stream_config.get('poll_interval', 0.1)
        follow = stream_config.get('follow', follow)
        
        reader = StreamLogReader(
            log_path=log_path,
            format_detector=self.format_detector.detect_format,
            poll_interval=poll_interval,
            follow=follow
        )
        
        # Detect format from existing content
        if Path(log_path).exists():
            detected_format = self.format_detector.detect_from_file(log_path, sample_lines=5)
            logger.info(f"Detected log format: {detected_format.value}")
        
        try:
            for line in reader.stream_lines():
                # Validate line
                if not self._validate_line(line):
                    continue
                
                # Process or yield
                if callback:
                    callback(line)
                else:
                    yield line
        finally:
            reader.close()
    
    def _validate_line(self, line: str) -> bool:
        """Validate log line format"""
        if not line or not line.strip():
            return False
        
        # Basic validation
        if len(line) > 100000:  # Sanity check
            logger.warning(f"Line too long: {len(line)} characters")
            return False
        
        return True
    
    def start_streaming_with_queue(self, log_path: str, processor: Optional[Callable] = None):
        """Start streaming with queue-based processing"""
        if processor:
            self.queue.processor = processor
        
        def enqueue_line(line: str):
            self.queue.put(line)
        
        # Start queue processor
        if self.queue.processor:
            self.queue.start_processor()
        
        # Start streaming
        try:
            for line in self.ingest_stream(log_path, callback=enqueue_line):
                pass
        finally:
            self.queue.stop_processor()
    
    def get_queue(self) -> LogQueue:
        """Get the log queue instance"""
        return self.queue
