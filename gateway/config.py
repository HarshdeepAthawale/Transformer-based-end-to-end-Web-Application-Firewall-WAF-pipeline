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


gateway_config = GatewayConfig()
