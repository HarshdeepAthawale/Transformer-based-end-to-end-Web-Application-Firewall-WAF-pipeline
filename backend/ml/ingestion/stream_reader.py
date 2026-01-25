"""
Streaming Log Reader Module

Tails log files in real-time for streaming log ingestion
"""
import os
import time
from pathlib import Path
from typing import Iterator, Optional, Callable
from loguru import logger


class StreamLogReader:
    """Tail log files in real-time (streaming mode)"""
    
    def __init__(
        self,
        log_path: str,
        format_detector: Optional[Callable] = None,
        poll_interval: float = 0.1,
        follow: bool = True
    ):
        self.log_path = Path(log_path)
        self.format_detector = format_detector
        self.poll_interval = poll_interval
        self.follow = follow
        self.file_position = 0
        self.file_handle = None
        
    def _get_file_size(self) -> int:
        """Get current file size"""
        try:
            return self.log_path.stat().st_size
        except:
            return 0
    
    def _open_file(self):
        """Open log file for reading"""
        if not self.log_path.exists():
            # Wait for file to be created
            logger.info(f"Waiting for log file to be created: {self.log_path}")
            while not self.log_path.exists():
                time.sleep(self.poll_interval)
        
        return open(self.log_path, 'r', encoding='utf-8', errors='ignore')
    
    def _read_new_lines(self) -> Iterator[str]:
        """Read new lines since last position"""
        try:
            current_size = self._get_file_size()
            
            # File was truncated or rotated
            if current_size < self.file_position:
                logger.warning(f"Log file rotated or truncated: {self.log_path}")
                self.file_position = 0
                if self.file_handle:
                    self.file_handle.close()
                    self.file_handle = None
            
            # Open file if not already open
            if self.file_handle is None:
                self.file_handle = self._open_file()
                self.file_handle.seek(self.file_position)
            
            # Read new content
            while True:
                line = self.file_handle.readline()
                if not line:
                    break
                
                line = line.strip()
                if line:
                    yield line
                    self.file_position = self.file_handle.tell()
            
        except Exception as e:
            logger.error(f"Error reading stream: {e}")
            if self.file_handle:
                try:
                    self.file_handle.close()
                except:
                    pass
                self.file_handle = None
    
    def stream_lines(self) -> Iterator[str]:
        """Stream log lines in real-time"""
        logger.info(f"Starting to tail log file: {self.log_path}")
        
        # Read existing content first
        for line in self._read_new_lines():
            yield line
        
        # Follow new lines
        if self.follow:
            while True:
                try:
                    new_lines = list(self._read_new_lines())
                    for line in new_lines:
                        yield line
                    
                    if not new_lines:
                        time.sleep(self.poll_interval)
                        
                except KeyboardInterrupt:
                    logger.info("Streaming interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error in streaming loop: {e}")
                    time.sleep(self.poll_interval)
        
        if self.file_handle:
            try:
                self.file_handle.close()
            except:
                pass
            self.file_handle = None
    
    def close(self):
        """Close file handle"""
        if self.file_handle:
            try:
                self.file_handle.close()
            except:
                pass
            self.file_handle = None
