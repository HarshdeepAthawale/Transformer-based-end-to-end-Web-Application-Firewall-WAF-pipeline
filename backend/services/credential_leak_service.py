"""
Credential leak protection: check password against HIBP (or compatible) API using k-anonymity.
Only SHA-1 prefix (5 chars) is sent; full password is never stored or logged.
"""

import hashlib
from typing import Optional

import httpx
from loguru import logger

from backend.config import config


def check_password(password: str) -> bool:
    """
    Return True if password appears in breaches (pwned), False otherwise.
    Uses k-anonymity: send first 5 hex chars of SHA-1, compare full hash suffix locally.
    No password stored or logged.
    """
    if not password:
        return False
    if not getattr(config, "CREDENTIAL_LEAK_ENABLED", False):
        return False

    full_hash = hashlib.sha1(password.encode("utf-8", errors="replace")).hexdigest().upper()
    prefix = full_hash[:5]
    suffix = full_hash[5:]

    api_url = (getattr(config, "CREDENTIAL_LEAK_API_URL", "") or "").rstrip("/")
    if not api_url:
        return False

    url = f"{api_url}{prefix}"
    timeout = max(1.0, getattr(config, "CREDENTIAL_LEAK_TIMEOUT_SECONDS", 5))
    headers = {}
    api_key = getattr(config, "CREDENTIAL_LEAK_API_KEY", None)
    if api_key:
        headers["hibp-api-key"] = api_key

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers=headers or None)
        resp.raise_for_status()
        text = resp.text
    except Exception as e:
        logger.warning(f"Credential leak API request failed: {e}")
        return False

    # Response: lines of "SUFFIX:count" (e.g. "0018A45C4D1DEF81644B54AB7F969B88D4:2")
    for line in text.splitlines():
        line = line.strip()
        if ":" in line:
            candidate_suffix, _ = line.split(":", 1)
            if candidate_suffix.strip().upper() == suffix:
                return True
    return False
