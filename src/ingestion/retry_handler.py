"""
Retry Handler Module

Handles retries with exponential backoff for robust error handling
"""
from functools import wraps
from typing import Callable, Any
from loguru import logger
import time


class RetryHandler:
    """Handle retries with exponential backoff"""
    
    @staticmethod
    def retry_with_backoff(
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        """Decorator for retry logic"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                delay = initial_delay
                last_exception = None
                
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                                f"Retrying in {delay:.2f}s..."
                            )
                            time.sleep(delay)
                            delay = min(delay * exponential_base, max_delay)
                        else:
                            logger.error(f"All {max_retries} attempts failed")
                
                raise last_exception
            return wrapper
        return decorator
