"""
Centralized WAF service creation from config.
Uses fine-tuned HuggingFace transformer model for threat detection.
"""
from pathlib import Path
from typing import Optional
from loguru import logger

from backend.config import config

_waf_classifier = None
_waf_initialized = False


def _project_root() -> Path:
    return Path(__file__).parent.parent.parent


def is_model_available(model_path: Optional[str] = None) -> bool:
    """
    Check if WAF model directory exists with required files.
    Requires config.json and tokenizer.json.
    """
    if not model_path:
        model_path = getattr(config, "WAF_MODEL_PATH", None)
    if not model_path:
        model_path = str(_project_root() / "models" / "waf-distilbert")
    p = Path(model_path)
    if not p.is_dir():
        return False
    return (p / "config.json").exists() and (p / "tokenizer.json").exists()


def create_waf_service(model_path: Optional[str] = None, force_reload: bool = False):
    """
    Create and cache a WAFClassifier instance.

    Returns the classifier if model exists, None otherwise.
    If force_reload=True, clear cache and create fresh (for hot swap).
    """
    global _waf_classifier, _waf_initialized

    if force_reload:
        _waf_classifier = None
        _waf_initialized = False

    if _waf_classifier is not None and not force_reload:
        return _waf_classifier

    if _waf_initialized and not force_reload:
        return None

    _waf_initialized = True

    # Get config values
    threshold = getattr(config, "WAF_THRESHOLD", 0.5)
    if not model_path:
        model_path = getattr(config, "WAF_MODEL_PATH", None)
    if not model_path:
        model_path = str(_project_root() / "models" / "waf-distilbert")

    try:
        from backend.ml.waf_classifier import WAFClassifier

        classifier = WAFClassifier(
            model_path=str(model_path),
            threshold=threshold,
        )

        if classifier.is_loaded:
            _waf_classifier = classifier
            logger.info(f"WAF classifier initialized with model at {model_path}")
            return _waf_classifier
        else:
            logger.warning(
                f"WAF model not found at {model_path}. "
                "Run: python scripts/finetune_waf_model.py"
            )
            return None

    except ImportError as e:
        logger.warning(f"WAF classifier not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to create WAF classifier: {e}")
        return None


def get_waf_service():
    """Return cached WAF classifier, creating it if necessary."""
    return create_waf_service()


def reload_waf_service(model_path: str):
    """
    Reload WAF classifier with new model path (hot swap).
    Clears cache and creates fresh classifier.
    """
    return create_waf_service(model_path=model_path, force_reload=True)
