"""
Centralized WAF service creation from config.
Uses fine-tuned HuggingFace transformer model for threat detection.
Set WAF_USE_ONNX=true to use the ONNX Runtime backend (3-5x faster).
"""
import os
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

    # ONNX branch — opt-in via WAF_USE_ONNX=true
    use_onnx = os.environ.get("WAF_USE_ONNX", "false").lower() == "true"
    if use_onnx:
        try:
            from backend.ml.onnx_classifier import ONNXWAFClassifier

            onnx_path = str(_project_root() / "models" / "waf-distilbert.onnx")
            classifier = ONNXWAFClassifier(
                model_path=str(model_path),
                onnx_path=onnx_path,
                threshold=threshold,
            )
            if classifier.is_loaded:
                _waf_classifier = classifier
                logger.info(f"ONNX WAF classifier initialized (model={model_path}, onnx={onnx_path})")
                return _waf_classifier
            else:
                logger.warning("ONNX classifier not loaded — falling back to PyTorch")
        except ImportError:
            logger.warning("onnxruntime not available — falling back to PyTorch")
        except Exception as e:
            logger.warning(f"ONNX init failed ({e}) — falling back to PyTorch")

    # Default PyTorch path
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
