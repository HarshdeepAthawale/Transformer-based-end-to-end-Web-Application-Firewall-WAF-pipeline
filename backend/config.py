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
