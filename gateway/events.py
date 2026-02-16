"""
Fire-and-forget event reporting from gateway to backend.
"""

import asyncio
from typing import Any, Dict

import httpx
from loguru import logger

from gateway.config import gateway_config


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


def report_event(event: Dict[str, Any]) -> None:
    """
    Fire-and-forget: report a single event to backend.
    Does not block the request path.
    """
    if not gateway_config.BACKEND_EVENTS_ENABLED:
        return

    asyncio.create_task(_send_events([event]))


def report_events(events: list[Dict[str, Any]]) -> None:
    """
    Fire-and-forget: report multiple events to backend.
    """
    if not gateway_config.BACKEND_EVENTS_ENABLED or not events:
        return

    asyncio.create_task(_send_events(events))
