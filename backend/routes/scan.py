"""Scan API: upload scan endpoint for gateway to call."""

from fastapi import APIRouter, File, UploadFile, HTTPException

from backend.config import config
from backend.services.upload_scan_service import scan_bytes

router = APIRouter()


@router.post("/upload")
async def scan_upload(file: UploadFile = File(...)):
    """
    Scan uploaded file bytes. Used by gateway when upload scan is enabled.
    Returns { "result": "clean" | "infected", "signature": str | null, "engine": str }.
    """
    if not config.UPLOAD_SCAN_ENABLED:
        return {"result": "clean", "signature": None, "engine": "none"}

    max_bytes = config.UPLOAD_SCAN_MAX_FILE_BYTES
    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {max_bytes} bytes)",
        )
    filename = file.filename or "upload"
    content_type = file.content_type or "application/octet-stream"
    result = scan_bytes(data, filename=filename, content_type=content_type)
    return result
