"""Firewall-for-AI: evaluate request, LLM endpoints CRUD, list endpoints for gateway."""

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.config import config
from backend.services.llm_endpoint_service import (
    list_endpoints,
    create_endpoint,
    update_endpoint,
    delete_endpoint,
    match_request,
)
from backend.services.firewall_ai_service import (
    check_prompt_injection,
    check_pii,
    check_abuse_rate,
    should_block_prompt_match,
    should_block_pii,
)

router = APIRouter()


class EvaluateRequest(BaseModel):
    path: str
    method: str = "POST"
    body: str = ""
    headers: Optional[dict[str, str]] = None
    client_ip: str = ""


class EvaluateResponse(BaseModel):
    applicable: bool
    block: bool = False
    reason: Optional[str] = None  # prompt_injection | pii | abuse_rate
    pattern: Optional[str] = None  # matched pattern for logging


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest, db: Session = Depends(get_db)):
    """
    Match path to LLM endpoint; if applicable run prompt-injection, PII, abuse rate checks.
    Returns { applicable, block, reason }. No mocks.
    """
    if not config.FIREWALL_AI_ENABLED:
        return EvaluateResponse(applicable=False, block=False, reason=None)

    endpoint = match_request(db, request.path, request.method)
    if not endpoint:
        return EvaluateResponse(applicable=False, block=False, reason=None)

    # Abuse rate limit (per IP per minute)
    if request.client_ip and check_abuse_rate(request.client_ip):
        return EvaluateResponse(
            applicable=True,
            block=True,
            reason="abuse_rate",
            pattern=None,
        )

    # Prompt-injection check
    matched, pattern = check_prompt_injection(
        request.body or "",
        request.headers or {},
        db,
    )
    if matched:
        if should_block_prompt_match():
            return EvaluateResponse(
                applicable=True,
                block=True,
                reason="prompt_injection",
                pattern=pattern,
            )
        return EvaluateResponse(
            applicable=True, block=False, reason="prompt_injection", pattern=pattern
        )

    # PII check
    pii_matched, pii_pattern = check_pii(request.body or "", db)
    if pii_matched and should_block_pii():
        return EvaluateResponse(
            applicable=True, block=True, reason="pii", pattern=pii_pattern
        )

    return EvaluateResponse(applicable=True, block=False, reason=None, pattern=None)


@router.get("/endpoints")
async def list_llm_endpoints(
    active_only: bool = False,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """List LLM endpoints (for gateway and frontend)."""
    endpoints = list_endpoints(db, active_only=active_only)
    return {
        "success": True,
        "data": [e.to_dict() for e in endpoints],
    }


class CreateEndpointRequest(BaseModel):
    path_pattern: str
    methods: str = "POST"
    label: str = "llm"
    is_active: bool = True


class UpdateEndpointRequest(BaseModel):
    path_pattern: Optional[str] = None
    methods: Optional[str] = None
    label: Optional[str] = None
    is_active: Optional[bool] = None


@router.post("/endpoints")
async def create_llm_endpoint(
    body: CreateEndpointRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    ep = create_endpoint(
        db,
        path_pattern=body.path_pattern,
        methods=body.methods,
        label=body.label,
        is_active=body.is_active,
    )
    return {"success": True, "data": ep.to_dict()}


@router.put("/endpoints/{endpoint_id}")
async def update_llm_endpoint(
    endpoint_id: int,
    body: UpdateEndpointRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    ep = update_endpoint(
        db,
        endpoint_id,
        path_pattern=body.path_pattern,
        methods=body.methods,
        label=body.label,
        is_active=body.is_active,
    )
    if not ep:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    return {"success": True, "data": ep.to_dict()}


@router.delete("/endpoints/{endpoint_id}")
async def delete_llm_endpoint(
    endpoint_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if not delete_endpoint(db, endpoint_id):
        raise HTTPException(status_code=404, detail="Endpoint not found")
    return {"success": True}
