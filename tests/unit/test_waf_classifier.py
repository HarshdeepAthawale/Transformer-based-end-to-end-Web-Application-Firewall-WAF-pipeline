"""Unit tests for WAF classifier."""

import pytest
from pathlib import Path

# Import after potential model path check
from backend.ml.waf_classifier import WAFClassifier


@pytest.fixture
def classifier_no_model(tmp_path):
    """Classifier with non-existent model path (graceful fallback)."""
    return WAFClassifier(model_path=str(tmp_path / "nonexistent"), threshold=0.5)


def test_classifier_not_loaded_without_model(classifier_no_model):
    """When model path does not exist, classifier should not be loaded."""
    assert classifier_no_model.is_loaded is False


def test_classify_returns_unknown_when_not_loaded(classifier_no_model):
    """When model not loaded, classify returns unknown/safe result."""
    result = classifier_no_model.classify("GET /api/users HTTP/1.1")
    assert result["label"] == "unknown"
    assert result["is_malicious"] is False
    assert "error" in result
    assert "Model not loaded" in result["error"]


def test_classifier_threshold_configurable(classifier_no_model):
    """Classifier respects threshold configuration."""
    assert classifier_no_model.threshold == 0.5


def test_request_to_text_normalization(classifier_no_model):
    """Test that request serialization works (parser pipeline)."""
    # Parser is initialized even when model not loaded
    text = classifier_no_model._pipeline.process_dict({
        "method": "GET",
        "path": "/api/users",
        "query_params": {"id": "123"},
        "headers": {"User-Agent": "Test"},
        "body": None,
    })
    assert "GET" in text
    assert "/api/users" in text
    assert "id" in text or "123" in text


@pytest.mark.skipif(
    not Path("models/waf-distilbert").exists(),
    reason="Model not available - run finetune_waf_model.py first",
)
def test_classifier_with_real_model():
    """When model exists, classifier loads and can classify (integration)."""
    classifier = WAFClassifier(model_path="models/waf-distilbert", threshold=0.5)
    assert classifier.is_loaded is True
    result = classifier.classify("GET /api/users?id=1 HTTP/1.1")
    assert result["label"] in ("benign", "malicious")
    assert "confidence" in result
