"""
API Server Configuration
"""
import os
from pathlib import Path
from typing import Optional
import yaml


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
        "http://localhost:3000,http://localhost:3001"
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
    WAF_THRESHOLD: float = float(os.getenv("WAF_THRESHOLD", "0.5"))
    WAF_MODEL_PATH: Optional[str] = os.getenv("WAF_MODEL_PATH", None)
    WAF_VOCAB_PATH: Optional[str] = os.getenv("WAF_VOCAB_PATH", None)
    WAF_TIMEOUT: float = float(os.getenv("WAF_TIMEOUT", "5.0"))  # Timeout in seconds
    
    # Log ingestion
    LOG_INGESTION_ENABLED: bool = os.getenv("LOG_INGESTION_ENABLED", "true").lower() == "true"
    LOG_PATH: Optional[str] = os.getenv("LOG_PATH")
    
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
