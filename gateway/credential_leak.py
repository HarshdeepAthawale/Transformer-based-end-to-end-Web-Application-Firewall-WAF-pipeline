"""
Credential leak protection: on login paths, extract password from JSON body,
call backend POST /api/credential-leak/check; block or flag on pwned. No password in events.
"""

import json
from typing import Optional

import httpx
from loguru import logger

from gateway.config import gateway_config
from gateway.events import report_event


def is_login_path(path: str, method: str) -> bool:
    """True if path matches CREDENTIAL_LEAK_LOGIN_PATHS and method is POST."""
    if (method or "").upper() != "POST":
        return False
    paths_str = getattr(gateway_config, "CREDENTIAL_LEAK_LOGIN_PATHS", "") or ""
    if not paths_str.strip():
        return False
    for p in paths_str.split(","):
        prefix = p.strip()
        if prefix and (path == prefix or path.startswith(prefix)):
            return True
    return False


def extract_password_and_username(body_bytes: bytes) -> tuple[Optional[str], Optional[str]]:
    """
    Parse JSON body and return (password, username) from configured field names.
    Returns (None, None) if not JSON or missing password field.
    """
    if not body_bytes:
        return None, None
    try:
        data = json.loads(body_bytes.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    pw_field = getattr(gateway_config, "CREDENTIAL_LEAK_PASSWORD_FIELD", "password") or "password"
    user_field = getattr(gateway_config, "CREDENTIAL_LEAK_USERNAME_FIELD", "username") or "username"
    password = data.get(pw_field)
    username = data.get(user_field)
    if password is None:
        return None, None
    return str(password), str(username) if username is not None else None


async def check_password_pwned(password: str) -> bool:
    """Call backend credential-leak check. Returns True if pwned."""
    url = (getattr(gateway_config, "CREDENTIAL_LEAK_BACKEND_URL", "") or "").rstrip("/")
    if not url:
        return False
    check_url = f"{url}/api/credential-leak/check"
    timeout = getattr(gateway_config, "CREDENTIAL_LEAK_TIMEOUT_SECONDS", 5) or 5
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                check_url,
                json={"password": password},
                headers={"Content-Type": "application/json"},
            )
        resp.raise_for_status()
        data = resp.json()
        return bool(data.get("pwned"))
    except Exception as e:
        logger.warning(f"Credential leak check failed: {e}")
        if getattr(gateway_config, "CREDENTIAL_LEAK_FAIL_OPEN", True):
            return False
        return False  # fail open: allow on error


async def process_credential_leak(
    body_bytes: bytes,
    path: str,
    client_ip: str,
    method: str,
) -> tuple[bool, Optional[str]]:
    """
    If request is a login path and body has password: check via backend.
    Returns (should_block, event_type). event_type is credential_leak_block or credential_leak_flag.
    """
    if not getattr(gateway_config, "CREDENTIAL_LEAK_ENABLED", False):
        return False, None
    if not is_login_path(path, method):
        return False, None

    max_bytes = getattr(gateway_config, "CREDENTIAL_LEAK_BODY_MAX_BYTES", 65536) or 65536
    if len(body_bytes or b"") > max_bytes:
        return False, None

    password, username = extract_password_and_username(body_bytes)
    if not password:
        return False, None

    pwned = await check_password_pwned(password)
    if not pwned:
        return False, None

    action = (getattr(gateway_config, "CREDENTIAL_LEAK_ACTION", "block") or "block").strip().lower()
    event_type = "credential_leak_block" if action == "block" else "credential_leak_flag"
    report_event({
        "event_type": event_type,
        "ip": client_ip,
        "method": method,
        "path": path,
        "credential_leak_username": username,
    })
    if action == "block":
        return True, event_type
    return False, event_type
