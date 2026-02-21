"""
Malicious upload scanning: detect multipart/binary uploads, buffer body,
call backend scan API, enforce policy (block / quarantine / log).
"""

import re
import uuid
from typing import Optional

import httpx
from loguru import logger

from gateway.config import gateway_config
from gateway.events import report_event


def is_upload_request(content_type: str, path: str) -> bool:
    """True if request should be subject to upload scan (multipart or path prefix)."""
    if not gateway_config.UPLOAD_SCAN_ENABLED:
        return False
    if (content_type or "").lower().find("multipart/form-data") >= 0:
        return True
    prefixes = gateway_config.UPLOAD_SCAN_PATH_PREFIXES
    if not prefixes:
        return False
    for p in prefixes.split(","):
        if p.strip() and path.startswith(p.strip()):
            return True
    return False


def _parse_boundary(content_type: str) -> Optional[str]:
    m = re.search(r"boundary\s*=\s*[\"']?([^\"'\s;]+)", content_type, re.I)
    return m.group(1).strip() if m else None


def _extract_first_file_from_multipart(body: bytes, content_type: str, max_bytes: int):
    """
    Parse multipart body and return first file part as (filename, data, content_type) or None.
    Respects max_bytes per part.
    """
    boundary = _parse_boundary(content_type or "")
    if not boundary:
        return None
    boundary_b = boundary.encode("utf-8") if isinstance(boundary, str) else boundary
    sep = b"\r\n--" + boundary_b
    parts = body.split(sep)
    for i, part in enumerate(parts):
        if i == 0 and part.strip().startswith(b"--"):
            continue
        if not part.strip():
            continue
        head, _, rest = part.partition(b"\r\n\r\n")
        if not rest:
            continue
        # Check for filename in Content-Disposition
        head_str = head.decode("utf-8", errors="replace")
        if "filename=" not in head_str and "filename*=" not in head_str:
            continue
        m = re.search(r'filename\*?=(?:\s*["\'])?([^"\';]+)', head_str, re.I)
        filename = m.group(1).strip().strip('"') if m else "upload"
        # Content-Type of part
        ct = "application/octet-stream"
        for line in head_str.split("\n"):
            if line.strip().lower().startswith("content-type:"):
                ct = line.split(":", 1)[1].strip().strip()
                break
        data = rest.rstrip(b"\r\n")
        if len(data) > max_bytes:
            return ("oversized", None, ct)  # signal oversized
        return (filename, data, ct)
    return None


async def scan_via_backend(
    file_data: bytes,
    filename: str,
    content_type: str,
) -> dict:
    """
    POST file to backend /api/scan/upload. Returns { "result": "clean"|"infected", "signature": ..., "engine": ... }.
    """
    url = (gateway_config.UPLOAD_SCAN_BACKEND_URL or "").rstrip("/")
    if not url:
        return {"result": "clean", "signature": None, "engine": "none"}
    scan_url = f"{url}/api/scan/upload"
    timeout = gateway_config.UPLOAD_SCAN_TIMEOUT_SECONDS
    try:
        files = {"file": (filename, file_data, content_type)}
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(scan_url, files=files)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Upload scan backend call failed: {e}")
        return {"result": "clean", "signature": None, "engine": "none"}


def quarantine_file(data: bytes, original_filename: str) -> Optional[str]:
    """Write file to quarantine dir with unique name (uuid + ext). Returns path or None."""
    qdir = (gateway_config.UPLOAD_SCAN_QUARANTINE_DIR or "").strip()
    if not qdir:
        return None
    ext = ""
    if original_filename and "." in original_filename:
        ext = "." + original_filename.rsplit(".", 1)[-1].strip()
    name = f"{uuid.uuid4().hex}{ext}"
    path = f"{qdir.rstrip('/')}/{name}"
    try:
        import os
        os.makedirs(qdir, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path
    except Exception as e:
        logger.warning(f"Quarantine write failed: {e}")
        return None


async def process_upload_scan(
    body_bytes: bytes,
    content_type: str,
    path: str,
    client_ip: str,
    method: str,
) -> tuple[bool, Optional[dict], Optional[str]]:
    """
    If upload scan is enabled and request is an upload: parse first file, call backend scan.
    Returns (should_block, scan_result_dict, error_message).
    - should_block: True if infected and policy is block or quarantine.
    - scan_result_dict: { result, signature, engine } or None if no scan was performed.
    - error_message: e.g. "File too large" for 413, or message for 403.
    """
    if not gateway_config.UPLOAD_SCAN_ENABLED or not body_bytes:
        return False, None, None

    max_bytes = gateway_config.UPLOAD_SCAN_MAX_FILE_BYTES
    skip_if_over = gateway_config.UPLOAD_SCAN_SKIP_IF_OVER_MAX

    first = _extract_first_file_from_multipart(body_bytes, content_type or "", max_bytes)
    if first is None:
        return False, None, None
    filename, file_data, ct = first
    if filename == "oversized" and file_data is None:
        if skip_if_over:
            return False, None, None
        return True, None, "File too large"

    if not file_data:
        return False, None, None

    scan_result = await scan_via_backend(file_data, filename, ct)
    result = scan_result.get("result") or "clean"
    signature = scan_result.get("signature")
    engine = scan_result.get("engine") or "unknown"

    policy = (gateway_config.UPLOAD_SCAN_POLICY_INFECTED or "block").strip().lower()
    event_type = "upload_scan_infected" if result == "infected" else "upload_scan_clean"
    event_payload = {
        "event_type": event_type,
        "ip": client_ip,
        "method": method,
        "path": path,
        "upload_scan_result": result,
        "upload_scan_signature": signature,
        "upload_filename": filename,
        "upload_size_bytes": len(file_data),
        "upload_scan_engine": engine,
        "upload_content_type": ct,
    }
    report_event(event_payload)

    if result == "infected":
        if policy == "quarantine":
            quarantine_file(file_data, filename)
        if policy in ("block", "quarantine"):
            return True, scan_result, "Malicious file detected"
        # log: forward request
    return False, scan_result, None
