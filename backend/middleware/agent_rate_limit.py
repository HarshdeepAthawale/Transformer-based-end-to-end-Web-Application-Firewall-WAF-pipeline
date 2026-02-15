"""Sliding-window rate limiter for /api/agent/* endpoints."""

from __future__ import annotations

import os
import time
from collections import defaultdict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Default: 10 requests per minute per IP
MAX_REQUESTS = int(os.getenv("AGENT_RATE_LIMIT_RPM", "10"))
WINDOW_SECONDS = 60


class AgentRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = MAX_REQUESTS) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        # ip -> list of timestamps
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Only apply to agent endpoints
        if not request.url.path.startswith("/api/agent"):
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - WINDOW_SECONDS

        # Prune old entries
        timestamps = self._requests[ip]
        self._requests[ip] = [t for t in timestamps if t > window_start]

        if len(self._requests[ip]) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "message": f"Rate limit exceeded. Max {self.max_requests} requests per minute for AI agent.",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )

        self._requests[ip].append(now)
        return await call_next(request)
