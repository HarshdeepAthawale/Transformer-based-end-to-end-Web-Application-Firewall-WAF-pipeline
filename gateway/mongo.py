"""
MongoDB event store for the WAF gateway.

Uses Motor (async MongoDB driver) to log every request with its WAF decision,
latency, and metadata. Writes are non-blocking to avoid adding proxy latency.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient

from gateway.config import gateway_config

# Headers whose values must be redacted before storage
_SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-auth-token",
        "proxy-authorization",
    }
)

# Max body sample stored per event (4 KB)
_BODY_SAMPLE_MAX = 4096


def _sanitize_headers(headers: dict) -> dict:
    """Redact sensitive header values."""
    sanitized = {}
    for k, v in headers.items():
        if k.lower() in _SENSITIVE_HEADERS:
            sanitized[k] = "[REDACTED]"
        else:
            sanitized[k] = v
    return sanitized


def _body_sample(body: Optional[bytes]) -> Optional[str]:
    """Return a truncated text sample of the request body."""
    if not body:
        return None
    return body[:_BODY_SAMPLE_MAX].decode("utf-8", errors="replace")


class MongoEventStore:
    """Async MongoDB event store for gateway request logging."""

    def __init__(self):
        self._client: Optional[AsyncIOMotorClient] = None
        self._db = None
        self._events = None

    async def connect(self) -> None:
        """Connect to MongoDB and ensure indexes exist."""
        uri = gateway_config.MONGODB_URI
        db_name = gateway_config.MONGODB_DB
        retention_days = gateway_config.EVENT_RETENTION_DAYS

        try:
            self._client = AsyncIOMotorClient(
                uri, serverSelectionTimeoutMS=5000
            )
            # Verify connection
            await self._client.admin.command("ping")
            self._db = self._client[db_name]
            self._events = self._db["events"]

            # Create indexes (idempotent)
            await self._events.create_index("timestamp", expireAfterSeconds=retention_days * 86400)
            await self._events.create_index("request_id")
            await self._events.create_index("decision")
            await self._events.create_index("client_ip")
            await self._events.create_index("path")
            await self._events.create_index([("timestamp", -1), ("decision", 1)])

            logger.info(
                f"MongoDB connected: {db_name} "
                f"(retention: {retention_days}d)"
            )
        except Exception as exc:
            logger.error(f"MongoDB connection failed: {exc}")
            self._client = None
            self._db = None
            self._events = None

    async def close(self) -> None:
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")

    async def is_ready(self) -> bool:
        """Check if MongoDB is reachable."""
        if not self._client:
            return False
        try:
            await self._client.admin.command("ping")
            return True
        except Exception:
            return False

    async def _insert_event(self, event: dict) -> None:
        """Insert a single event document. Errors are logged, never raised."""
        if self._events is None:
            return
        try:
            await self._events.insert_one(event)
        except Exception as exc:
            logger.debug(f"Failed to insert event: {exc}")

    def log_event(
        self,
        *,
        request_id: str,
        client_ip: str,
        method: str,
        path: str,
        query_string: str,
        headers: dict,
        body: Optional[bytes],
        decision: str,
        anomaly_score: Optional[float] = None,
        waf_latency_ms: Optional[float] = None,
        upstream_status: Optional[int] = None,
        total_latency_ms: float,
        blocked_by: Optional[str] = None,
        user_agent: str = "",
        content_length: Optional[int] = None,
    ) -> None:
        """
        Fire-and-forget: schedule an event insert without blocking the caller.

        This is called from the request path — the actual MongoDB write
        happens in the background via asyncio.create_task.
        """
        if self._events is None:
            return

        doc: Dict[str, Any] = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc),
            "client_ip": client_ip,
            "method": method,
            "path": path,
            "query_string": query_string,
            "headers_snapshot": _sanitize_headers(headers),
            "body_sample": _body_sample(body),
            "decision": decision,
            "anomaly_score": round(anomaly_score, 4) if anomaly_score is not None else None,
            "waf_threshold": gateway_config.WAF_THRESHOLD,
            "waf_mode": gateway_config.WAF_MODE,
            "waf_latency_ms": round(waf_latency_ms, 2) if waf_latency_ms is not None else None,
            "upstream_status": upstream_status,
            "total_latency_ms": round(total_latency_ms, 1),
            "blocked_by": blocked_by,
            "user_agent": user_agent,
            "content_length": content_length,
        }

        asyncio.create_task(self._insert_event(doc))
