"""
Retry Utility

Exponential backoff retry for transient I/O errors during log ingestion.
Supports both sync and async callables.
"""

import asyncio
import functools
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retryable_exceptions: tuple[Type[Exception], ...] = field(
        default_factory=lambda: (OSError, IOError, ConnectionError, TimeoutError)
    )


def _compute_delay(attempt: int, config: RetryConfig) -> float:
    """Compute delay for a given attempt using exponential backoff."""
    delay = config.initial_delay * (config.exponential_base ** attempt)
    return min(delay, config.max_delay)


def with_retry_sync(
    fn: Callable[..., Any],
    config: Optional[RetryConfig] = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call a sync function with exponential backoff on retryable exceptions.

    Args:
        fn: The function to call.
        config: Retry configuration (uses defaults if None).
        *args, **kwargs: Arguments passed to fn.

    Returns:
        The return value of fn.

    Raises:
        The last exception if all retries are exhausted.
    """
    if config is None:
        config = RetryConfig()

    last_exception: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e
            if attempt < config.max_retries:
                delay = _compute_delay(attempt, config)
                logger.warning(
                    "Retry %d/%d for %s after error: %s (delay=%.1fs)",
                    attempt + 1, config.max_retries, fn.__name__, e, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "All %d retries exhausted for %s: %s",
                    config.max_retries, fn.__name__, e,
                )

    raise last_exception  # type: ignore[misc]


async def with_retry_async(
    fn: Callable[..., Any],
    config: Optional[RetryConfig] = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call an async function with exponential backoff on retryable exceptions.

    Args:
        fn: The async function to call.
        config: Retry configuration (uses defaults if None).
        *args, **kwargs: Arguments passed to fn.

    Returns:
        The return value of fn.

    Raises:
        The last exception if all retries are exhausted.
    """
    if config is None:
        config = RetryConfig()

    last_exception: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e
            if attempt < config.max_retries:
                delay = _compute_delay(attempt, config)
                logger.warning(
                    "Retry %d/%d for %s after error: %s (delay=%.1fs)",
                    attempt + 1, config.max_retries, fn.__name__, e, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "All %d retries exhausted for %s: %s",
                    config.max_retries, fn.__name__, e,
                )

    raise last_exception  # type: ignore[misc]


def retry(config: Optional[RetryConfig] = None) -> Callable:
    """Decorator for adding retry logic to sync or async functions.

    Usage:
        @retry()
        def read_file(path):
            ...

        @retry(RetryConfig(max_retries=5))
        async def fetch_data():
            ...
    """
    if config is None:
        config = RetryConfig()

    def decorator(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await with_retry_async(fn, config, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return with_retry_sync(fn, config, *args, **kwargs)
            return sync_wrapper

    return decorator
