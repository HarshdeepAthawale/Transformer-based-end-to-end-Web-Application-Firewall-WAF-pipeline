"""
Batch Log Reader Module

Reads logs from files in batch mode, supporting both plain text and gzip compressed files
"""
import os
import gzip
from typing import Iterator, Optional, Callable
from pathlib import Path
from loguru import logger
import time


class BatchLogReader:
    """Read logs from files in batch mode"""
    
    def __init__(
        self,
        log_path: str,
        format_detector: Optional[Callable] = None,
        max_lines: Optional[int] = None,
        skip_lines: int = 0
    ):
        self.log_path = Path(log_path)
        self.format_detector = format_detector
        self.max_lines = max_lines
        self.skip_lines = skip_lines
        self.line_count = 0
        
    def _open_file(self):
        """Open log file, handling gzip compression"""
        if self.log_path.suffix == '.gz':
            return gzip.open(self.log_path, 'rt', encoding='utf-8', errors='ignore')
        else:
            return open(self.log_path, 'r', encoding='utf-8', errors='ignore')
    
    def read_lines(self) -> Iterator[str]:
        """Read log lines from file"""
        if not self.log_path.exists():
            logger.error(f"Log file not found: {self.log_path}")
            return
        
        try:
            with self._open_file() as f:
                # Skip initial lines if specified
                for _ in range(self.skip_lines):
                    try:
                        next(f)
                    except StopIteration:
                        break
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    self.line_count += 1
                    
                    # Check max lines limit
                    if self.max_lines and self.line_count > self.max_lines:
                        break
                    
                    yield line
                    
        except Exception as e:
            logger.error(f"Error reading log file {self.log_path}: {e}")
            raise
    
    def get_file_size(self) -> int:
        """Get log file size in bytes"""
        try:
            return self.log_path.stat().st_size
        except:
            return 0
    
    def get_line_count(self) -> int:
        """Get approximate line count (for progress tracking)"""
        try:
            with self._open_file() as f:
                return sum(1 for line in f if line.strip())
        except:
            return 0
    
    def read_chunks(self, chunk_size: int = 1000) -> Iterator[list]:
        """Read logs in chunks for batch processing"""
        chunk = []
        for line in self.read_lines():
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
