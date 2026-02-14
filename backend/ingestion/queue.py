"""
Async Ingestion Queue

Producer/consumer pattern using asyncio.Queue for buffering
log lines between ingestion and downstream processing stages.
"""

import asyncio
import logging
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

# Sentinel value to signal shutdown
_SHUTDOWN = object()


class IngestionQueue:
    """Async queue for buffering log lines between pipeline stages.

    Supports multiple producers and consumers with graceful shutdown.

    Args:
        maxsize: Maximum queue size (0 = unlimited).
        name: Queue name for logging.
    """

    def __init__(self, maxsize: int = 10000, name: str = "ingestion"):
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
        self._name = name
        self._produced = 0
        self._consumed = 0
        self._running = False

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def produced(self) -> int:
        return self._produced

    @property
    def consumed(self) -> int:
        return self._consumed

    @property
    def is_running(self) -> bool:
        return self._running

    async def put(self, item: Any, timeout: Optional[float] = None) -> None:
        """Add an item to the queue.

        Args:
            item: The item to enqueue.
            timeout: Max seconds to wait if queue is full (None = wait forever).

        Raises:
            asyncio.QueueFull: If timeout expires and queue is still full.
        """
        if timeout is not None:
            await asyncio.wait_for(self._queue.put(item), timeout=timeout)
        else:
            await self._queue.put(item)
        self._produced += 1

    async def get(self, timeout: Optional[float] = None) -> Any:
        """Get an item from the queue.

        Args:
            timeout: Max seconds to wait if queue is empty (None = wait forever).

        Returns:
            The next item, or None if shutdown was signaled.

        Raises:
            asyncio.QueueEmpty: If timeout expires and queue is still empty.
        """
        if timeout is not None:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        else:
            item = await self._queue.get()

        if item is _SHUTDOWN:
            # Re-add sentinel for other consumers
            await self._queue.put(_SHUTDOWN)
            return None

        self._consumed += 1
        return item

    async def consume(
        self,
        handler: Callable[[Any], Coroutine[Any, Any, None]],
    ) -> None:
        """Run a consumer loop that processes items until shutdown.

        Args:
            handler: Async callable invoked for each item.
        """
        self._running = True
        logger.info("Queue '%s' consumer started", self._name)

        try:
            while True:
                item = await self.get()
                if item is None:
                    break
                try:
                    await handler(item)
                except Exception:
                    logger.exception(
                        "Queue '%s' handler error on item: %s",
                        self._name, str(item)[:100],
                    )
        finally:
            self._running = False
            logger.info(
                "Queue '%s' consumer stopped (produced=%d, consumed=%d)",
                self._name, self._produced, self._consumed,
            )

    async def shutdown(self) -> None:
        """Signal all consumers to stop."""
        logger.info("Queue '%s' shutting down", self._name)
        await self._queue.put(_SHUTDOWN)

    def stats(self) -> dict:
        """Return queue statistics."""
        return {
            "name": self._name,
            "size": self.size,
            "produced": self._produced,
            "consumed": self._consumed,
            "running": self._running,
        }
