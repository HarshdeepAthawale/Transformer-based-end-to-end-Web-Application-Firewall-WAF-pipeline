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
    WAF_ATTACK_SCORE_BLOCK_THRESHOLD: int = int(os.getenv("WAF_ATTACK_SCORE_BLOCK_THRESHOLD", "70"))
    WAF_ATTACK_SCORE_CHALLENGE_THRESHOLD: int = int(os.getenv("WAF_ATTACK_SCORE_CHALLENGE_THRESHOLD", "0"))

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

    # IP blacklist (backend syncs to Redis; gateway enforces)
    BLACKLIST_ENABLED: bool = os.getenv("BLACKLIST_ENABLED", "true").lower() == "true"
    BLACKLIST_TENANT_ID: str = os.getenv("BLACKLIST_TENANT_ID", "default")
    BLACKLIST_FAIL_OPEN: bool = os.getenv("BLACKLIST_FAIL_OPEN", "true").lower() == "true"

    # DDoS protection
    DDOS_ENABLED: bool = os.getenv("DDOS_ENABLED", "true").lower() == "true"
    DDOS_MAX_BODY_BYTES: int = int(os.getenv("DDOS_MAX_BODY_BYTES", str(10 * 1024 * 1024)))
    DDOS_BURST_THRESHOLD: int = int(os.getenv("DDOS_BURST_THRESHOLD", "50"))
    DDOS_BURST_WINDOW_SECONDS: int = int(os.getenv("DDOS_BURST_WINDOW_SECONDS", "5"))
    DDOS_BLOCK_DURATION_SECONDS: int = int(os.getenv("DDOS_BLOCK_DURATION_SECONDS", "60"))
    DDOS_FAIL_OPEN: bool = os.getenv("DDOS_FAIL_OPEN", "true").lower() == "true"
    # Adaptive DDoS: read threshold from Redis; record requests for learning
    ADAPTIVE_DDOS_ENABLED: bool = os.getenv("ADAPTIVE_DDOS_ENABLED", "false").lower() == "true"
    ADAPTIVE_DDOS_REDIS_KEY: str = os.getenv("ADAPTIVE_DDOS_REDIS_KEY", "waf:ddos:adaptive_threshold")
    ADAPTIVE_DDOS_LEARNING_WINDOW_MINUTES: int = int(os.getenv("ADAPTIVE_DDOS_LEARNING_WINDOW_MINUTES", "60"))
    ADAPTIVE_DDOS_REFRESH_SECONDS: int = int(os.getenv("ADAPTIVE_DDOS_REFRESH_SECONDS", "60"))

    # Bot management (score + bands)
    BOT_ENABLED: bool = os.getenv("BOT_ENABLED", "true").lower() == "true"
    BOT_BACKEND_URL: str = os.getenv("BOT_BACKEND_URL", "")
    BOT_FAIL_OPEN: bool = os.getenv("BOT_FAIL_OPEN", "true").lower() == "true"
    BOT_TIMEOUT_SECONDS: float = float(os.getenv("BOT_TIMEOUT_SECONDS", "1.0"))
    BOT_CHALLENGE_RETRY_AFTER: int = int(os.getenv("BOT_CHALLENGE_RETRY_AFTER", "60"))

    # Managed rules (fetch from backend, evaluate on request)
    MANAGED_RULES_ENABLED: bool = os.getenv("MANAGED_RULES_ENABLED", "false").lower() == "true"
    MANAGED_RULES_BACKEND_URL: str = os.getenv("MANAGED_RULES_BACKEND_URL", "")
    MANAGED_RULES_CACHE_TTL_SECONDS: int = int(os.getenv("MANAGED_RULES_CACHE_TTL_SECONDS", "300"))
    MANAGED_RULES_FAIL_OPEN: bool = os.getenv("MANAGED_RULES_FAIL_OPEN", "true").lower() == "true"

    # Event reporting to backend
    BACKEND_EVENTS_URL: str = os.getenv("BACKEND_EVENTS_URL", "")
    BACKEND_EVENTS_ENABLED: bool = os.getenv("BACKEND_EVENTS_ENABLED", "true").lower() == "true"
    EVENTS_BATCH_SIZE: int = int(os.getenv("EVENTS_BATCH_SIZE", "50"))
    EVENTS_BATCH_INTERVAL_SECONDS: float = float(os.getenv("EVENTS_BATCH_INTERVAL_SECONDS", "2.0"))

    # MongoDB (event store)
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB: str = os.getenv("MONGODB_DB", "waf_gateway")
    EVENT_RETENTION_DAYS: int = int(os.getenv("EVENT_RETENTION_DAYS", "30"))

    # WAF body inspection limit (separate from proxy body limit)
    WAF_BODY_INSPECT_MAX: int = int(os.getenv("WAF_BODY_INSPECT_MAX", str(1 * 1024 * 1024)))  # 1MB

    # Malicious upload scanning (gateway buffers body, backend performs scan)
    UPLOAD_SCAN_ENABLED: bool = os.getenv("UPLOAD_SCAN_ENABLED", "false").lower() == "true"
    UPLOAD_SCAN_BACKEND_URL: str = os.getenv("UPLOAD_SCAN_BACKEND_URL", "")  # e.g. http://localhost:3001
    UPLOAD_SCAN_MAX_FILE_BYTES: int = int(os.getenv("UPLOAD_SCAN_MAX_FILE_BYTES", str(50 * 1024 * 1024)))  # 50MB
    UPLOAD_SCAN_SKIP_IF_OVER_MAX: bool = os.getenv("UPLOAD_SCAN_SKIP_IF_OVER_MAX", "true").lower() == "true"
    UPLOAD_SCAN_PATH_PREFIXES: str = os.getenv("UPLOAD_SCAN_PATH_PREFIXES", "")  # comma-separated; empty = all paths
    UPLOAD_SCAN_POLICY_INFECTED: str = os.getenv("UPLOAD_SCAN_POLICY_INFECTED", "block")  # block | quarantine | log
    UPLOAD_SCAN_QUARANTINE_DIR: str = os.getenv("UPLOAD_SCAN_QUARANTINE_DIR", "")
    UPLOAD_SCAN_TIMEOUT_SECONDS: float = float(os.getenv("UPLOAD_SCAN_TIMEOUT_SECONDS", "30"))

    # Firewall for AI (LLM endpoint protection)
    FIREWALL_AI_ENABLED: bool = os.getenv("FIREWALL_AI_ENABLED", "false").lower() == "true"
    FIREWALL_AI_BACKEND_URL: str = os.getenv("FIREWALL_AI_BACKEND_URL", "")
    FIREWALL_AI_TIMEOUT: float = float(os.getenv("FIREWALL_AI_TIMEOUT", "5.0"))
    FIREWALL_AI_FAIL_OPEN: bool = os.getenv("FIREWALL_AI_FAIL_OPEN", "true").lower() == "true"
    FIREWALL_AI_CACHE_TTL_SECONDS: int = int(os.getenv("FIREWALL_AI_CACHE_TTL_SECONDS", "60"))

    # Credential leak protection (HIBP; gateway calls backend check)
    CREDENTIAL_LEAK_ENABLED: bool = os.getenv("CREDENTIAL_LEAK_ENABLED", "false").lower() == "true"
    CREDENTIAL_LEAK_BACKEND_URL: str = os.getenv("CREDENTIAL_LEAK_BACKEND_URL", "")
    CREDENTIAL_LEAK_LOGIN_PATHS: str = os.getenv("CREDENTIAL_LEAK_LOGIN_PATHS", "/login,/api/auth/login")
    CREDENTIAL_LEAK_PASSWORD_FIELD: str = os.getenv("CREDENTIAL_LEAK_PASSWORD_FIELD", "password")
    CREDENTIAL_LEAK_USERNAME_FIELD: str = os.getenv("CREDENTIAL_LEAK_USERNAME_FIELD", "username")
    CREDENTIAL_LEAK_ACTION: str = os.getenv("CREDENTIAL_LEAK_ACTION", "block")
    CREDENTIAL_LEAK_TIMEOUT_SECONDS: float = float(os.getenv("CREDENTIAL_LEAK_TIMEOUT_SECONDS", "5"))
    CREDENTIAL_LEAK_BODY_MAX_BYTES: int = int(os.getenv("CREDENTIAL_LEAK_BODY_MAX_BYTES", str(64 * 1024)))
    CREDENTIAL_LEAK_FAIL_OPEN: bool = os.getenv("CREDENTIAL_LEAK_FAIL_OPEN", "true").lower() == "true"


gateway_config = GatewayConfig()
