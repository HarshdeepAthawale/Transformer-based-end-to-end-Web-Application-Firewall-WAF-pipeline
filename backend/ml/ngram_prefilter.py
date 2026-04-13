"""
Ngram Pre-Filter for fast-path WAF classification.

Uses Jensen-Shannon divergence between a request's character-level ngram
distribution and pre-computed benign/malicious class distributions to
classify obviously benign or obviously malicious requests without invoking
the transformer model.

The ngram profiles are built by scripts/build_ngram_profiles.py from the
training dataset and stored in backend/ml/ngram_profiles.json.

Architecture:
  - For each incoming request, compute character-level ngram frequencies
    (n=3 to n=5) and compare against learned class distributions
  - Compute JSD(request, benign) and JSD(request, malicious)
  - Use the difference to classify high-confidence cases
  - Uncertain cases are passed to the transformer
"""
import json
import math
from collections import Counter
from pathlib import Path
from typing import Optional

_PROFILES = None
_PROFILES_LOADED = False

NGRAM_SIZES = [3, 4, 5]

# Thresholds on JSD_benign (the primary discriminator).
# Calibrated from training data (19,384 samples) to achieve zero misclassification:
#   Benign JSD_benign range:    [0.07 .. 0.736]
#   Malicious JSD_benign range: [0.190 .. 0.87]
#
# BENIGN_CEILING: JSD_benign below this -> classify as benign without transformer.
#   Set at 0.19, just below the minimum malicious JSD_benign (0.190).
#   This guarantees zero malicious samples are misclassified as benign.
#
# MALICIOUS_FLOOR: JSD_benign above this -> classify as malicious without transformer.
#   Set at 0.74, just above the maximum benign JSD_benign (0.736).
#   This guarantees zero benign samples are misclassified as malicious.
BENIGN_JSD_CEILING = 0.19
MALICIOUS_JSD_FLOOR = 0.74


def _load_profiles() -> Optional[dict]:
    """Load pre-computed ngram profiles from disk."""
    global _PROFILES, _PROFILES_LOADED
    if _PROFILES_LOADED:
        return _PROFILES

    _PROFILES_LOADED = True
    profiles_path = Path(__file__).parent / "ngram_profiles.json"
    if not profiles_path.exists():
        return None

    try:
        with open(profiles_path) as f:
            _PROFILES = json.load(f)
        return _PROFILES
    except (json.JSONDecodeError, IOError):
        return None


def _extract_ngrams(text: str, n: int) -> Counter:
    """Extract character-level ngrams from text."""
    text_lower = text.lower()
    ngrams = Counter()
    for i in range(len(text_lower) - n + 1):
        ngrams[text_lower[i:i + n]] += 1
    return ngrams


def _normalize(counter: Counter, vocab: set) -> dict:
    """Normalize counter to probability distribution over vocab."""
    total = sum(counter.get(k, 0) for k in vocab)
    if total == 0:
        return {k: 0.0 for k in vocab}
    return {k: counter.get(k, 0) / total for k in vocab}


def _jsd(p: dict, q: dict, vocab: list) -> float:
    """Jensen-Shannon divergence between two distributions."""
    eps = 1e-10
    result = 0.0
    for k in vocab:
        pk = p.get(k, eps)
        qk = q.get(k, eps)
        mk = 0.5 * (pk + qk)
        if pk > 0 and mk > 0:
            result += 0.5 * pk * math.log2(pk / mk)
        if qk > 0 and mk > 0:
            result += 0.5 * qk * math.log2(qk / mk)
    return result


def _compute_weighted_jsd(text: str, class_name: str, profiles: dict) -> float:
    """Compute weighted-average JSD between request and a class distribution."""
    jsds = []
    weights = [1.0, 1.5, 2.0]  # Weight higher n-grams more
    for n in NGRAM_SIZES:
        key = f"{n}gram"
        if key not in profiles:
            continue
        profile = profiles[key]
        vocab = profile["vocab"]
        req_ngrams = _extract_ngrams(text, n)
        req_dist = _normalize(req_ngrams, set(vocab))
        jsd_val = _jsd(req_dist, profile[class_name], vocab)
        jsds.append(jsd_val)
    if not jsds:
        return 1.0
    total_w = sum(weights[:len(jsds)])
    return sum(w * j for w, j in zip(weights, jsds)) / total_w


def quick_score(text: str) -> Optional[int]:
    """
    Fast pre-filter using Jensen-Shannon divergence.

    Compares the request's character-level ngram distribution (n=3,4,5)
    against learned benign and malicious class distributions.

    Returns:
        - 1-10: Definitely malicious (skip transformer)
        - 90-99: Definitely benign (skip transformer)
        - None: Uncertain, must run full transformer inference

    Scores use WAF convention: lower = more malicious.
    """
    profiles = _load_profiles()
    if profiles is None:
        return None

    jsd_benign = _compute_weighted_jsd(text, "benign", profiles)

    # Fast path: request closely matches benign distribution
    if jsd_benign < BENIGN_JSD_CEILING:
        return 99  # Definitely benign

    # Fast path: request is very far from benign distribution
    if jsd_benign > MALICIOUS_JSD_FLOOR:
        return 5  # Definitely malicious

    # Uncertain zone: let the transformer classify
    return None


def get_prefilter_stats(text: str) -> dict:
    """
    Get detailed pre-filter analysis (for debugging/monitoring).
    """
    profiles = _load_profiles()
    if profiles is None:
        return {
            "error": "ngram profiles not loaded",
            "needs_transformer": True,
            "recommendation": "run_transformer",
        }

    jsd_benign = _compute_weighted_jsd(text, "benign", profiles)
    jsd_malicious = _compute_weighted_jsd(text, "malicious", profiles)

    # Per-ngram detail
    details = {}
    for n in NGRAM_SIZES:
        key = f"{n}gram"
        if key not in profiles:
            continue
        profile = profiles[key]
        vocab = profile["vocab"]
        req_ngrams = _extract_ngrams(text, n)
        req_dist = _normalize(req_ngrams, set(vocab))
        details[f"jsd_benign_{n}gram"] = round(
            _jsd(req_dist, profile["benign"], vocab), 4
        )
        details[f"jsd_malicious_{n}gram"] = round(
            _jsd(req_dist, profile["malicious"], vocab), 4
        )

    score = quick_score(text)

    details.update({
        "avg_jsd_benign": round(jsd_benign, 4),
        "avg_jsd_malicious": round(jsd_malicious, 4),
        "benign_ceiling": BENIGN_JSD_CEILING,
        "malicious_floor": MALICIOUS_JSD_FLOOR,
        "prefilter_score": score,
        "text_length": len(text),
        "needs_transformer": score is None,
        "recommendation": (
            "block" if score is not None and score <= 10
            else "allow" if score is not None and score >= 90
            else "run_transformer"
        ),
    })

    return details
