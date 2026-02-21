"""
API Server Configuration
"""
import os
from pathlib import Path
from typing import Optional
import yaml

# Load .env file if it exists
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


class Config:
    """API server configuration"""
    
    # Server settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "3001"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "4"))
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{Path(__file__).parent.parent}/data/waf_dashboard.db"
    )
    
    # WebSocket
    WEBSOCKET_ENABLED: bool = os.getenv("WEBSOCKET_ENABLED", "true").lower() == "true"
    WEBSOCKET_PING_INTERVAL: int = int(os.getenv("WEBSOCKET_PING_INTERVAL", "30"))
    
    # CORS
    CORS_ORIGINS: list = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001"
    ).split(",")
    
    # Data retention (days)
    METRICS_RETENTION_DAYS: int = int(os.getenv("METRICS_RETENTION_DAYS", "30"))
    TRAFFIC_RETENTION_DAYS: int = int(os.getenv("TRAFFIC_RETENTION_DAYS", "7"))
    ALERTS_RETENTION_DAYS: int = int(os.getenv("ALERTS_RETENTION_DAYS", "90"))
    THREATS_RETENTION_DAYS: int = int(os.getenv("THREATS_RETENTION_DAYS", "90"))
    
    # WAF Service
    WAF_SERVICE_URL: str = os.getenv("WAF_SERVICE_URL", "http://localhost:8000")
    
    # WAF Middleware Configuration
    WAF_ENABLED: bool = os.getenv("WAF_ENABLED", "true").lower() == "true"
    WAF_FAIL_OPEN: bool = os.getenv("WAF_FAIL_OPEN", "true").lower() == "true"  # Allow requests if WAF unavailable
    WAF_THRESHOLD: float = float(os.getenv("WAF_THRESHOLD", "0.65"))
    WAF_MODEL_PATH: Optional[str] = os.getenv("WAF_MODEL_PATH", None)
    WAF_VOCAB_PATH: Optional[str] = os.getenv("WAF_VOCAB_PATH", None)
    WAF_TIMEOUT: float = float(os.getenv("WAF_TIMEOUT", "5.0"))  # Timeout in seconds
    
    # Log ingestion
    LOG_INGESTION_ENABLED: bool = os.getenv("LOG_INGESTION_ENABLED", "true").lower() == "true"
    LOG_PATH: Optional[str] = os.getenv("LOG_PATH")

    # Events ingest: also write to traffic_logs so Request Volume chart shows gateway traffic
    EVENTS_INGEST_WRITE_TRAFFIC_LOG: bool = os.getenv("EVENTS_INGEST_WRITE_TRAFFIC_LOG", "true").lower() == "true"

    # Bot management (verified bots, score bands)
    BOT_VERIFIED_SYNC_URL: str = os.getenv("BOT_VERIFIED_SYNC_URL", "")
    BOT_VERIFIED_SYNC_CRON: str = os.getenv("BOT_VERIFIED_SYNC_CRON", "0 */6 * * *")
    BOT_VERIFIED_SYNC_HEADER: Optional[str] = os.getenv("BOT_VERIFIED_SYNC_HEADER", None)
    BOT_SCORE_VERIFIED: int = int(os.getenv("BOT_SCORE_VERIFIED", "95"))
    BOT_SCORE_MISSING_UA: int = int(os.getenv("BOT_SCORE_MISSING_UA", "25"))
    BOT_SCORE_BEHAVIORAL: int = int(os.getenv("BOT_SCORE_BEHAVIORAL", "25"))
    BOT_SCORE_SIGNATURE_MATCHED: int = int(os.getenv("BOT_SCORE_SIGNATURE_MATCHED", "34"))
    BOT_DEFAULT_SCORE_UNKNOWN: int = int(os.getenv("BOT_DEFAULT_SCORE_UNKNOWN", "50"))
    BOT_BAND_BLOCK_MAX: int = int(os.getenv("BOT_BAND_BLOCK_MAX", "29"))
    BOT_BAND_CHALLENGE_MAX: int = int(os.getenv("BOT_BAND_CHALLENGE_MAX", "69"))

    # Managed rules (OWASP CRS + auto-update)
    MANAGED_RULES_FEED_URL: str = os.getenv("MANAGED_RULES_FEED_URL", "")
    MANAGED_RULES_FEED_FORMAT: str = os.getenv("MANAGED_RULES_FEED_FORMAT", "json")  # owasp_crs | json
    MANAGED_RULES_UPDATE_INTERVAL_HOURS: int = int(os.getenv("MANAGED_RULES_UPDATE_INTERVAL_HOURS", "24"))
    MANAGED_RULES_PACK_ID: str = os.getenv("MANAGED_RULES_PACK_ID", "owasp_crs")
    MANAGED_RULES_FEED_HEADER: Optional[str] = os.getenv("MANAGED_RULES_FEED_HEADER", None)  # e.g. Bearer token

    # Malicious upload scanning (ClamAV or cloud API)
    UPLOAD_SCAN_ENABLED: bool = os.getenv("UPLOAD_SCAN_ENABLED", "false").lower() == "true"
    UPLOAD_SCAN_ENGINE: str = os.getenv("UPLOAD_SCAN_ENGINE", "clamav")  # clamav | cloud
    CLAMAV_SOCKET: str = os.getenv("CLAMAV_SOCKET", "")
    CLAMAV_TIMEOUT_SECONDS: float = float(os.getenv("CLAMAV_TIMEOUT_SECONDS", "30"))
    UPLOAD_SCAN_CLOUD_URL: str = os.getenv("UPLOAD_SCAN_CLOUD_URL", "")
    UPLOAD_SCAN_CLOUD_API_KEY: Optional[str] = os.getenv("UPLOAD_SCAN_CLOUD_API_KEY", None)
    UPLOAD_SCAN_POLICY_INFECTED: str = os.getenv("UPLOAD_SCAN_POLICY_INFECTED", "block")  # block | quarantine | log
    UPLOAD_SCAN_QUARANTINE_DIR: str = os.getenv("UPLOAD_SCAN_QUARANTINE_DIR", "")
    UPLOAD_SCAN_MAX_FILE_BYTES: int = int(os.getenv("UPLOAD_SCAN_MAX_FILE_BYTES", str(50 * 1024 * 1024)))  # 50MB
    UPLOAD_SCAN_PATH_PREFIXES: str = os.getenv("UPLOAD_SCAN_PATH_PREFIXES", "")  # comma-separated; empty = all
    UPLOAD_SCAN_SKIP_IF_OVER_MAX: bool = os.getenv("UPLOAD_SCAN_SKIP_IF_OVER_MAX", "true").lower() == "true"

    # Adaptive DDoS protection (learn baseline, auto-tune burst threshold)
    ADAPTIVE_DDOS_ENABLED: bool = os.getenv("ADAPTIVE_DDOS_ENABLED", "false").lower() == "true"
    ADAPTIVE_DDOS_LEARNING_WINDOW_MINUTES: int = int(os.getenv("ADAPTIVE_DDOS_LEARNING_WINDOW_MINUTES", "60"))
    ADAPTIVE_DDOS_PERCENTILE: float = float(os.getenv("ADAPTIVE_DDOS_PERCENTILE", "95"))
    ADAPTIVE_DDOS_MULTIPLIER: float = float(os.getenv("ADAPTIVE_DDOS_MULTIPLIER", "1.5"))
    ADAPTIVE_DDOS_THRESHOLD_MIN: int = int(os.getenv("ADAPTIVE_DDOS_THRESHOLD_MIN", "20"))
    ADAPTIVE_DDOS_THRESHOLD_MAX: int = int(os.getenv("ADAPTIVE_DDOS_THRESHOLD_MAX", "500"))
    ADAPTIVE_DDOS_UPDATE_INTERVAL_MINUTES: int = int(os.getenv("ADAPTIVE_DDOS_UPDATE_INTERVAL_MINUTES", "15"))
    ADAPTIVE_DDOS_REDIS_KEY: str = os.getenv("ADAPTIVE_DDOS_REDIS_KEY", "waf:ddos:adaptive_threshold")
    ADAPTIVE_DDOS_REDIS_KEY_META: str = os.getenv("ADAPTIVE_DDOS_REDIS_KEY_META", "waf:ddos:adaptive_meta")

    # Firewall for AI (LLM endpoint protection)
    FIREWALL_AI_ENABLED: bool = os.getenv("FIREWALL_AI_ENABLED", "false").lower() == "true"
    FIREWALL_AI_PROMPT_PATTERNS_URL: str = os.getenv("FIREWALL_AI_PROMPT_PATTERNS_URL", "")
    FIREWALL_AI_PII_PATTERNS_URL: str = os.getenv("FIREWALL_AI_PII_PATTERNS_URL", "")
    FIREWALL_AI_ACTION_PROMPT_MATCH: str = os.getenv("FIREWALL_AI_ACTION_PROMPT_MATCH", "block")  # block | log | challenge
    FIREWALL_AI_ACTION_PII: str = os.getenv("FIREWALL_AI_ACTION_PII", "log")  # block | log | redact
    FIREWALL_AI_ABUSE_RATE_PER_MINUTE: int = int(os.getenv("FIREWALL_AI_ABUSE_RATE_PER_MINUTE", "60"))
    FIREWALL_AI_PATTERNS_REFRESH_SECONDS: int = int(os.getenv("FIREWALL_AI_PATTERNS_REFRESH_SECONDS", "300"))

    # Credential leak protection (HIBP k-anonymity)
    CREDENTIAL_LEAK_ENABLED: bool = os.getenv("CREDENTIAL_LEAK_ENABLED", "false").lower() == "true"
    CREDENTIAL_LEAK_API_URL: str = os.getenv("CREDENTIAL_LEAK_API_URL", "https://api.pwnedpasswords.com/range/")
    CREDENTIAL_LEAK_API_KEY: Optional[str] = os.getenv("CREDENTIAL_LEAK_API_KEY", None)
    CREDENTIAL_LEAK_LOGIN_PATHS: str = os.getenv("CREDENTIAL_LEAK_LOGIN_PATHS", "/login,/api/auth/login")
    CREDENTIAL_LEAK_PASSWORD_FIELD: str = os.getenv("CREDENTIAL_LEAK_PASSWORD_FIELD", "password")
    CREDENTIAL_LEAK_USERNAME_FIELD: str = os.getenv("CREDENTIAL_LEAK_USERNAME_FIELD", "username")
    CREDENTIAL_LEAK_ACTION: str = os.getenv("CREDENTIAL_LEAK_ACTION", "block")  # block | flag
    CREDENTIAL_LEAK_TIMEOUT_SECONDS: float = float(os.getenv("CREDENTIAL_LEAK_TIMEOUT_SECONDS", "5"))
    CREDENTIAL_LEAK_INCLUDE_HASH_PREFIX_IN_EVENTS: bool = os.getenv("CREDENTIAL_LEAK_INCLUDE_HASH_PREFIX_IN_EVENTS", "false").lower() == "true"

    # Alert notifications (email)
    SMTP_HOST: Optional[str] = os.getenv("SMTP_HOST", None)
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: Optional[str] = os.getenv("SMTP_USER", None)
    SMTP_PASSWORD: Optional[str] = os.getenv("SMTP_PASSWORD", None)
    SMTP_USE_TLS: bool = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
    ALERT_FROM_EMAIL: str = os.getenv("ALERT_FROM_EMAIL", "waf-alerts@localhost")
    DASHBOARD_BASE_URL: str = os.getenv("DASHBOARD_BASE_URL", "http://localhost:3000")
    
    @classmethod
    def load_from_yaml(cls, config_path: Optional[str] = None):
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        if not Path(config_path).exists():
            return cls()
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update config from YAML
        if "api_server" in config_data:
            api_config = config_data["api_server"]
            cls.API_HOST = api_config.get("host", cls.API_HOST)
            cls.API_PORT = api_config.get("port", cls.API_PORT)
            cls.API_WORKERS = api_config.get("workers", cls.API_WORKERS)
            
            if "database" in api_config:
                cls.DATABASE_URL = api_config["database"].get("url", cls.DATABASE_URL)
            
            if "websocket" in api_config:
                cls.WEBSOCKET_ENABLED = api_config["websocket"].get("enabled", cls.WEBSOCKET_ENABLED)
                cls.WEBSOCKET_PING_INTERVAL = api_config["websocket"].get("ping_interval", cls.WEBSOCKET_PING_INTERVAL)
            
            if "log_ingestion" in api_config:
                cls.LOG_INGESTION_ENABLED = api_config["log_ingestion"].get("enabled", cls.LOG_INGESTION_ENABLED)
                cls.LOG_PATH = api_config["log_ingestion"].get("log_path", cls.LOG_PATH)
        
        return cls()


# Global config instance
config = Config.load_from_yaml()
