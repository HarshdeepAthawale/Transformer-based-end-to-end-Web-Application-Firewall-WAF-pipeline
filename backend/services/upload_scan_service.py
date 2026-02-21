"""
Upload scan service: ClamAV (socket INSTREAM) or cloud API.
Returns result (clean | infected), signature, and engine name.
"""

import socket
import struct

import httpx
from loguru import logger

from backend.config import config


def scan_bytes(
    data: bytes,
    filename: str = "",
    content_type: str = "",
) -> dict:
    """
    Scan file bytes with configured engine (ClamAV or cloud).
    Returns: { "result": "clean" | "infected", "signature": str | null, "engine": str }
    """
    engine = (config.UPLOAD_SCAN_ENGINE or "clamav").strip().lower()
    if engine == "clamav":
        return _scan_clamav(data)
    if engine == "cloud":
        return _scan_cloud(data, filename, content_type)
    logger.warning(f"Unknown UPLOAD_SCAN_ENGINE={engine}; treating as clean")
    return {"result": "clean", "signature": None, "engine": engine}


def _scan_clamav(data: bytes) -> dict:
    """Send INSTREAM to ClamAV daemon socket. Ref: ClamAV INSTREAM protocol."""
    socket_path = (config.CLAMAV_SOCKET or "").strip()
    if not socket_path:
        logger.warning("CLAMAV_SOCKET not set; skipping scan (result=clean)")
        return {"result": "clean", "signature": None, "engine": "clamav"}

    timeout = max(1.0, getattr(config, "CLAMAV_TIMEOUT_SECONDS", 30))
    sock = None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(socket_path)

        # INSTREAM: send z\0 (length 4 bytes big-endian) then chunk, then 0-length chunk
        cmd = b"zINSTREAM\0"
        sock.sendall(cmd)

        offset = 0
        chunk_size = 2048
        while offset < len(data):
            chunk = data[offset : offset + chunk_size]
            length_be = struct.pack(b">I", len(chunk))
            sock.sendall(length_be)
            sock.sendall(chunk)
            offset += len(chunk)
        sock.sendall(struct.pack(b">I", 0))

        # Read response: null-terminated string, e.g. "stream: OK\0" or "stream: Eicar-Test-Signature FOUND\0"
        buf = b""
        while True:
            b = sock.recv(1)
            if not b or b == b"\0":
                break
            buf += b
        response = buf.decode("utf-8", errors="replace").strip()
    except (socket.timeout, socket.error, OSError) as e:
        logger.warning(f"ClamAV socket error: {e}")
        return {"result": "clean", "signature": None, "engine": "clamav"}
    finally:
        if sock:
            try:
                sock.close()
            except Exception:
                pass

    # Parse: "stream: OK" -> clean; "stream: <name> FOUND" -> infected
    if response.endswith("OK") or "OK" in response.split():
        return {"result": "clean", "signature": None, "engine": "clamav"}
    if " FOUND" in response:
        sig = response.replace("stream:", "").strip().replace(" FOUND", "").strip()
        return {"result": "infected", "signature": sig or None, "engine": "clamav"}
    # Unknown response: treat as clean to avoid blocking
    logger.debug(f"ClamAV unknown response: {response}")
    return {"result": "clean", "signature": None, "engine": "clamav"}


def _scan_cloud(
    data: bytes,
    filename: str,
    content_type: str,
) -> dict:
    """POST file to cloud scan API. Expects JSON with result (clean/infected) or malicious (bool)."""
    url = (config.UPLOAD_SCAN_CLOUD_URL or "").strip()
    api_key = config.UPLOAD_SCAN_CLOUD_API_KEY or ""
    if not url:
        logger.warning("UPLOAD_SCAN_CLOUD_URL not set; skipping scan (result=clean)")
        return {"result": "clean", "signature": None, "engine": "cloud"}

    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    timeout = max(5.0, getattr(config, "CLAMAV_TIMEOUT_SECONDS", 30))
    try:
        # Multipart file upload
        files = {"file": (filename or "upload", data, content_type or "application/octet-stream")}
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, files=files, headers=headers)
        resp.raise_for_status()
        body = resp.json()
    except Exception as e:
        logger.warning(f"Cloud scan API error: {e}")
        return {"result": "clean", "signature": None, "engine": "cloud"}

    # Support common response shapes: { "result": "clean"|"infected", "signature": "..." }
    # or { "malicious": false } or { "infected": false }
    result = "clean"
    signature = body.get("signature") if isinstance(body, dict) else None
    if isinstance(body, dict):
        if body.get("result") == "infected":
            result = "infected"
        elif body.get("malicious") is True or body.get("infected") is True:
            result = "infected"
            signature = signature or body.get("signature_name") or "unknown"
    return {"result": result, "signature": signature, "engine": "cloud"}
