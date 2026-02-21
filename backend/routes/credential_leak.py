"""Credential leak protection: check password against HIBP (k-anonymity)."""

from fastapi import APIRouter
from pydantic import BaseModel

from backend.config import config
from backend.services.credential_leak_service import check_password

router = APIRouter()


class CheckRequest(BaseModel):
    password: str = ""


class CheckResponse(BaseModel):
    pwned: bool


@router.post("/check", response_model=CheckResponse)
async def check_credential_leak(body: CheckRequest):
    """
    Check if password has been seen in breaches (HIBP k-anonymity).
    Used by gateway on login paths. No password stored or logged.
    """
    if not config.CREDENTIAL_LEAK_ENABLED:
        return CheckResponse(pwned=False)
    pwned = check_password(body.password or "")
    return CheckResponse(pwned=pwned)
