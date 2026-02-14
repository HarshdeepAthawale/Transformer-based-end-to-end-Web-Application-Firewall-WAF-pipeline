"""
Ingestion Config Loader

Loads the `ingestion:` section from config/config.yaml into typed dataclasses.
Falls back to sensible defaults if the file is missing or incomplete.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from backend.ingestion.retry import RetryConfig

logger = logging.getLogger(__name__)

# Default config path relative to project root
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"


@dataclass
class BatchConfig:
    """Configuration for batch log reading."""
    chunk_size: int = 1000
    max_lines: Optional[int] = None
    skip_lines: int = 0


@dataclass
class StreamingConfig:
    """Configuration for streaming log tailing."""
    poll_interval: float = 0.1
    follow: bool = True
    buffer_size: int = 10000


@dataclass
class IngestionConfig:
    """Top-level ingestion configuration."""
    batch: BatchConfig
    streaming: StreamingConfig
    retry: RetryConfig

    @staticmethod
    def defaults() -> "IngestionConfig":
        return IngestionConfig(
            batch=BatchConfig(),
            streaming=StreamingConfig(),
            retry=RetryConfig(),
        )


def load_ingestion_config(
    config_path: Optional[str] = None,
) -> IngestionConfig:
    """Load ingestion config from a YAML file.

    Args:
        config_path: Path to config.yaml. Uses default project path if None.

    Returns:
        IngestionConfig populated from YAML, with defaults for missing keys.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.warning("Config file not found at %s, using defaults", path)
        return IngestionConfig.defaults()

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error("Failed to read config %s: %s, using defaults", path, e)
        return IngestionConfig.defaults()

    ingestion_data = data.get("ingestion", {})
    return _parse_ingestion(ingestion_data)


def _parse_ingestion(data: dict[str, Any]) -> IngestionConfig:
    """Parse the ingestion section of the config."""
    batch_data = data.get("batch", {})
    streaming_data = data.get("streaming", {})
    retry_data = data.get("retry", {})

    batch = BatchConfig(
        chunk_size=batch_data.get("chunk_size", 1000),
        max_lines=batch_data.get("max_lines"),
        skip_lines=batch_data.get("skip_lines", 0),
    )

    streaming = StreamingConfig(
        poll_interval=streaming_data.get("poll_interval", 0.1),
        follow=streaming_data.get("follow", True),
        buffer_size=streaming_data.get("buffer_size", 10000),
    )

    retry = RetryConfig(
        max_retries=retry_data.get("max_retries", 3),
        initial_delay=retry_data.get("initial_delay", 1.0),
        max_delay=retry_data.get("max_delay", 60.0),
        exponential_base=retry_data.get("exponential_base", 2.0),
    )

    return IngestionConfig(batch=batch, streaming=streaming, retry=retry)
