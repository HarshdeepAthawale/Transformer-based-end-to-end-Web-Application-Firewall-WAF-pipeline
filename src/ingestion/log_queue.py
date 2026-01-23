"""
Log Queue Module

Thread-safe queue for asynchronous log processing
"""
from queue import Queue, Empty
from threading import Thread, Event
from typing import Optional, Callable
from loguru import logger
import time


class LogQueue:
    """Thread-safe queue for log processing"""
    
    def __init__(
        self,
        maxsize: int = 10000,
        timeout: float = 1.0,
        processor: Optional[Callable] = None
    ):
        self.queue = Queue(maxsize=maxsize)
        self.timeout = timeout
        self.processor = processor
        self.stop_event = Event()
        self.processor_thread = None
        
    def put(self, item, block: bool = True, timeout: Optional[float] = None):
        """Add item to queue"""
        try:
            self.queue.put(item, block=block, timeout=timeout or self.timeout)
        except Exception as e:
            logger.warning(f"Failed to add item to queue: {e}")
    
    def get(self, block: bool = True, timeout: Optional[float] = None):
        """Get item from queue"""
        try:
            return self.queue.get(block=block, timeout=timeout or self.timeout)
        except Empty:
            return None
    
    def start_processor(self):
        """Start background processor thread"""
        if self.processor and not self.processor_thread:
            self.stop_event.clear()
            self.processor_thread = Thread(target=self._process_loop, daemon=True)
            self.processor_thread.start()
            logger.info("Log queue processor started")
    
    def stop_processor(self):
        """Stop background processor"""
        if self.processor_thread:
            self.stop_event.set()
            self.processor_thread.join(timeout=5.0)
            self.processor_thread = None
            logger.info("Log queue processor stopped")
    
    def _process_loop(self):
        """Background processing loop"""
        while not self.stop_event.is_set():
            try:
                item = self.get(timeout=0.1)
                if item and self.processor:
                    self.processor(item)
            except Exception as e:
                logger.error(f"Error in processor loop: {e}")
                time.sleep(0.1)
    
    def size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return self.queue.empty()
    
    def clear(self):
        """Clear all items from queue"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break
