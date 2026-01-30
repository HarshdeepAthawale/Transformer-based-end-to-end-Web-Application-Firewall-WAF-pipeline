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


def create_waf_service():
    """
    Create and cache a WAFClassifier instance.

    Returns the classifier if model exists, None otherwise.
    """
    global _waf_classifier, _waf_initialized

    if _waf_classifier is not None:
        return _waf_classifier

    if _waf_initialized:
        return None

    _waf_initialized = True

    # Get config values
    threshold = getattr(config, "WAF_THRESHOLD", 0.5)
    model_path = getattr(config, "WAF_MODEL_PATH", None)

    # Default model path
    if not model_path:
        model_path = _project_root() / "models" / "waf-distilbert"

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
