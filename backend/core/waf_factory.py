"""
Centralized WAF service creation from config.
Used by middleware and WAF routes; no imports from routes.
"""
from pathlib import Path
from typing import Optional
import json
from loguru import logger

from backend.config import config

_waf_service = None
_waf_service_initialized = False


def _project_root() -> Path:
    return Path(__file__).parent.parent.parent


def _load_optimized_threshold() -> Optional[float]:
    """Load optimized threshold from reports/threshold_optimization.json if present."""
    try:
        reports_dir = _project_root() / "reports"
        threshold_file = reports_dir / "threshold_optimization.json"
        if threshold_file.exists():
            with open(threshold_file, "r") as f:
                data = json.load(f)
                optimal = data.get("optimal_threshold")
                if optimal is not None:
                    logger.info(f"Loaded optimized threshold from {threshold_file}: {optimal}")
                    return float(optimal)
    except Exception as e:
        logger.debug(f"Could not load optimized threshold: {e}")
    return None


def create_waf_service():
    """
    Create and cache a WAFService instance.
    Attempts to load ML models if paths are configured, falls back to placeholder mode.
    """
    global _waf_service, _waf_service_initialized
    if _waf_service is not None:
        return _waf_service
    if _waf_service_initialized:
        return None
    _waf_service_initialized = True
    try:
        from backend.ml.waf_service import WAFService
    except ImportError as e:
        logger.warning(f"WAFService not available: {e}")
        return None

    threshold = _load_optimized_threshold()
    if threshold is None:
        threshold = getattr(config, "WAF_THRESHOLD", 0.5)
    timeout = getattr(config, "WAF_TIMEOUT", 5.0)

    # Try to load model paths from config
    model_path = getattr(config, "WAF_MODEL_PATH", None)
    vocab_path = getattr(config, "WAF_VOCAB_PATH", None)
    
    # If not in config, try default paths
    if not model_path:
        default_model = _project_root() / "models" / "deployed" / "model.pt"
        if default_model.exists():
            model_path = str(default_model)
            logger.info(f"Found default model at {model_path}")
    
    if not vocab_path:
        default_vocab = _project_root() / "models" / "vocabularies" / "vocab.json"
        if default_vocab.exists():
            vocab_path = str(default_vocab)
            logger.info(f"Found default vocabulary at {vocab_path}")

    try:
        # Initialize WAF service (will use ML if paths provided, else placeholder)
        svc = WAFService(
            model_path=model_path,
            vocab_path=vocab_path,
            threshold=threshold,
            timeout=timeout,
        )
        _waf_service = svc
        
        if svc.ml_enabled:
            logger.info("WAF service initialized via core.waf_factory with ML components")
        else:
            logger.info("WAF service initialized via core.waf_factory in placeholder mode")
        
        return _waf_service
    except Exception as e:
        logger.error(f"Failed to create WAF service: {e}", exc_info=True)
        return None


def get_waf_service():
    """Return cached WAF service, creating it if necessary."""
    return create_waf_service()
