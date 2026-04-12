"""
Ngram Pre-Filter for fast-path WAF classification.

Performs a fast check before the expensive transformer inference.
If the request is obviously benign (short, no suspicious patterns) or
obviously malicious (multiple high-signal attack patterns), we can skip
the transformer entirely.

Inspired by Cloudflare's WAF ML optimization blog:
  - Known-malicious ngrams extracted from training data
  - Fast string matching before model inference
  - ~30-50% of requests can be classified without the transformer
"""

from typing import Optional


# High-signal malicious ngrams extracted from common attack payloads.
# These are substrings that almost always indicate an attack.
MALICIOUS_NGRAMS = frozenset({
    # SQL Injection
    "' or ", "' and ", "union select", "union all select",
    "' or '1'='1", "'; drop ", "1=1--", "' or 1=1",
    "select * from", "information_schema", "load_file(",
    "into outfile", "benchmark(", "sleep(",
    "extractvalue(", "updatexml(",

    # XSS
    "<script>", "<script ", "javascript:", "onerror=",
    "onload=", "onmouseover=", "<img src=", "<svg onload",
    "document.cookie", "alert(", "<iframe", "eval(",
    "expression(", "vbscript:",

    # Command Injection / RCE
    "; ls ", "; cat ", "; wget ", "; curl ",
    "| ls ", "| cat ", "| wget ", "| curl ",
    "`ls`", "`cat`", "`id`", "`whoami`",
    "$(/bin/", "$(cat ", "${ifs}", "/etc/passwd",
    "/etc/shadow", "cmd.exe", "powershell",

    # Path Traversal
    "../../", "..\\..\\", "%2e%2e%2f", "%252e%252e",
    "/etc/passwd", "/proc/self", "\\windows\\system32",

    # XXE
    "<!entity", "<!doctype", "system \"file:", "system \"http:",
    "<!element", "<?xml", "xmlns:",

    # SSRF
    "169.254.169.254", "metadata.google", "127.0.0.1:",
    "localhost:", "0.0.0.0:", "[::1]",

    # SSTI
    "{{7*7}}", "${7*7}", "<%=", "<#assign",

    # Log4Shell
    "${jndi:", "${jndi:ldap", "${jndi:rmi",
})

# Minimum hits to classify as definitely malicious without transformer
MALICIOUS_HIT_THRESHOLD = 2

# Maximum request length to classify as definitely benign without transformer
# Short requests with no suspicious patterns are almost always benign
BENIGN_MAX_LENGTH = 150


def quick_score(text: str) -> Optional[int]:
    """
    Fast pre-filter: returns attack_score if confident, None if uncertain.

    Returns:
        - 1-10: Definitely malicious (skip transformer, save time)
        - 95-99: Definitely benign (skip transformer, save time)
        - None: Uncertain, must run full transformer inference

    Scores use Cloudflare convention: lower = more malicious.
    """
    text_lower = text.lower()

    # Count malicious ngram hits
    hits = 0
    matched = []
    for ngram in MALICIOUS_NGRAMS:
        if ngram in text_lower:
            hits += 1
            matched.append(ngram)
            if hits >= MALICIOUS_HIT_THRESHOLD:
                # Definitely malicious: multiple attack indicators
                return 5

    if hits == 0 and len(text) <= BENIGN_MAX_LENGTH:
        # Short request with zero suspicious patterns: definitely benign
        return 99

    if hits == 1 and len(text) > 500:
        # Single hit in a long request: could be a false positive
        # Let the transformer decide
        return None

    if hits >= 1:
        # At least one hit but not enough to be certain
        # Return a moderate score (still suspicious)
        return None

    # No hits, longer request: let transformer decide
    return None


def get_prefilter_stats(text: str) -> dict:
    """
    Get detailed pre-filter analysis (for debugging/monitoring).

    Returns dict with hit count, matched patterns, and recommendation.
    """
    text_lower = text.lower()
    matched = []
    for ngram in MALICIOUS_NGRAMS:
        if ngram in text_lower:
            matched.append(ngram)

    score = quick_score(text)
    return {
        "ngram_hits": len(matched),
        "matched_ngrams": matched[:10],  # Limit output
        "text_length": len(text),
        "prefilter_score": score,
        "needs_transformer": score is None,
        "recommendation": (
            "block" if score is not None and score <= 10
            else "allow" if score is not None and score >= 90
            else "run_transformer"
        ),
    }
