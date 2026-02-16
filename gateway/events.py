"""
Fire-and-forget event reporting from gateway to backend.
Events are batched and POSTed in bulk to reduce backend load.
"""

import asyncio
from typing import Any, Dict

import httpx
from loguru import logger

from gateway.config import gateway_config

# In-memory queue and lock for event batching
_event_queue: list[Dict[str, Any]] = []
_event_lock = asyncio.Lock()
_batcher_task: asyncio.Task | None = None


async def _send_events(events: list[Dict[str, Any]]) -> None:
    """Send events to backend. Non-blocking; errors are logged only."""
    if not gateway_config.BACKEND_EVENTS_ENABLED or not gateway_config.BACKEND_EVENTS_URL:
        return

    url = gateway_config.BACKEND_EVENTS_URL.rstrip("/")
    if not url:
        return

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                url,
                json={"events": events},
                headers={"Content-Type": "application/json"},
            )
    except Exception as e:
        logger.debug(f"Event reporting failed: {e}")


async def _enqueue_event(event: Dict[str, Any]) -> None:
    """Add event to the batch queue."""
    async with _event_lock:
        _event_queue.append(event)


async def _batcher_loop() -> None:
    """Background task: drain queue and send events in batches."""
    batch_size = gateway_config.EVENTS_BATCH_SIZE
    interval = gateway_config.EVENTS_BATCH_INTERVAL_SECONDS
    if batch_size < 1:
        batch_size = 50
    if interval < 0.1:
        interval = 2.0
    while True:
        await asyncio.sleep(interval)
        batch: list[Dict[str, Any]] = []
        async with _event_lock:
            take = min(len(_event_queue), batch_size)
            if take > 0:
                batch = _event_queue[:take]
                del _event_queue[:take]
        if batch:
            await _send_events(batch)


def report_event(event: Dict[str, Any]) -> None:
    """
    Fire-and-forget: enqueue event for batched delivery to backend.
    Does not block the request path.
    """
    if not gateway_config.BACKEND_EVENTS_ENABLED:
        return
    asyncio.create_task(_enqueue_event(event))


def report_events(events: list[Dict[str, Any]]) -> None:
    """
    Fire-and-forget: enqueue multiple events for batched delivery to backend.
    """
    if not gateway_config.BACKEND_EVENTS_ENABLED or not events:
        return
    for ev in events:
        asyncio.create_task(_enqueue_event(ev))


def start_event_batcher() -> None:
    """Start the background batcher task. Call from app lifespan."""
    global _batcher_task
    if not gateway_config.BACKEND_EVENTS_ENABLED or not gateway_config.BACKEND_EVENTS_URL:
        return
    if _batcher_task is None or _batcher_task.done():
        _batcher_task = asyncio.create_task(_batcher_loop())
        logger.info("Event batcher started (batch POST to backend)")


async def stop_event_batcher() -> None:
    """Flush remaining events and stop the batcher. Call on shutdown."""
    global _batcher_task
    if _batcher_task:
        _batcher_task.cancel()
        try:
            await _batcher_task
        except asyncio.CancelledError:
            pass
        _batcher_task = None
    async with _event_lock:
        if _event_queue:
            await _send_events(_event_queue[:])
            _event_queue.clear()
