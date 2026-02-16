"""
Gateway Configuration
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


class GatewayConfig:
    """Configuration for the WAF gateway reverse proxy."""

    # Gateway server
    GATEWAY_HOST: str = os.getenv("GATEWAY_HOST", "0.0.0.0")
    GATEWAY_PORT: int = int(os.getenv("GATEWAY_PORT", "8080"))

    # Upstream target
    UPSTREAM_URL: str = os.getenv("UPSTREAM_URL", "http://localhost:3000")

    # WAF behaviour
    WAF_ENABLED: bool = os.getenv("WAF_ENABLED", "true").lower() == "true"
    WAF_MODE: str = os.getenv("WAF_MODE", "monitor")  # "monitor" or "block"
    WAF_THRESHOLD: float = float(os.getenv("WAF_THRESHOLD", "0.65"))
    WAF_FAIL_OPEN: bool = os.getenv("WAF_FAIL_OPEN", "true").lower() == "true"
    WAF_TIMEOUT: float = float(os.getenv("WAF_TIMEOUT", "5.0"))

    # Body limits
    BODY_MAX_BYTES: int = int(os.getenv("BODY_MAX_BYTES", str(10 * 1024 * 1024)))

    # Proxy tuning
    PROXY_TIMEOUT: float = float(os.getenv("PROXY_TIMEOUT", "30.0"))
    PROXY_MAX_CONNECTIONS: int = int(os.getenv("PROXY_MAX_CONNECTIONS", "100"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Redis (for rate limiting and DDoS)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "120"))
    RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    RATE_LIMIT_BURST: int = int(os.getenv("RATE_LIMIT_BURST", "20"))
    RATE_LIMIT_FAIL_OPEN: bool = os.getenv("RATE_LIMIT_FAIL_OPEN", "true").lower() == "true"

    # DDoS protection
    DDOS_ENABLED: bool = os.getenv("DDOS_ENABLED", "true").lower() == "true"
    DDOS_MAX_BODY_BYTES: int = int(os.getenv("DDOS_MAX_BODY_BYTES", str(10 * 1024 * 1024)))
    DDOS_BURST_THRESHOLD: int = int(os.getenv("DDOS_BURST_THRESHOLD", "50"))
    DDOS_BURST_WINDOW_SECONDS: int = int(os.getenv("DDOS_BURST_WINDOW_SECONDS", "5"))
    DDOS_BLOCK_DURATION_SECONDS: int = int(os.getenv("DDOS_BLOCK_DURATION_SECONDS", "60"))
    DDOS_FAIL_OPEN: bool = os.getenv("DDOS_FAIL_OPEN", "true").lower() == "true"

    # Event reporting to backend
    BACKEND_EVENTS_URL: str = os.getenv("BACKEND_EVENTS_URL", "")
    BACKEND_EVENTS_ENABLED: bool = os.getenv("BACKEND_EVENTS_ENABLED", "false").lower() == "true"

    # MongoDB (event store)
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB: str = os.getenv("MONGODB_DB", "waf_gateway")
    EVENT_RETENTION_DAYS: int = int(os.getenv("EVENT_RETENTION_DAYS", "30"))

    # WAF body inspection limit (separate from proxy body limit)
    WAF_BODY_INSPECT_MAX: int = int(os.getenv("WAF_BODY_INSPECT_MAX", str(1 * 1024 * 1024)))  # 1MB


gateway_config = GatewayConfig()
