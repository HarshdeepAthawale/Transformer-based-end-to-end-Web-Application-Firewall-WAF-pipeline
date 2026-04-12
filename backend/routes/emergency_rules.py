"""Emergency Rules API endpoints."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import get_current_tenant
from backend.services.emergency_rules import EmergencyRuleService, ZERO_DAY_PATTERNS

router = APIRouter()
_service = EmergencyRuleService()


@router.get("")
async def list_emergency_rules(
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """List all emergency rules."""
    return {"success": True, "data": _service.list_rules(db)}


class CreateRuleRequest(BaseModel):
    name: str
    description: Optional[str] = None
    cves: list[str] = []
    patterns: list[dict] = []
    action: str = "block"
    severity: str = "critical"
    source: str = "manual"


@router.post("", status_code=201)
async def create_emergency_rule(
    body: CreateRuleRequest,
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Create a new emergency rule."""
    if not body.patterns:
        raise HTTPException(status_code=400, detail="At least one pattern is required")
    result = _service.create_rule(db, body.model_dump())
    return {"success": True, "data": result}


class ToggleRequest(BaseModel):
    enabled: bool


@router.patch("/{rule_id}/toggle")
async def toggle_emergency_rule(
    rule_id: int,
    body: ToggleRequest,
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Enable or disable an emergency rule."""
    result = _service.toggle_rule(db, rule_id, body.enabled)
    if not result:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"success": True, "data": result}


@router.delete("/{rule_id}", status_code=204)
async def delete_emergency_rule(
    rule_id: int,
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Delete an emergency rule."""
    if not _service.delete_rule(db, rule_id):
        raise HTTPException(status_code=404, detail="Rule not found")
    return None


@router.get("/templates")
async def list_zero_day_templates(
    org_id: int = Depends(get_current_tenant),
):
    """List available zero-day pattern templates that can be quickly deployed."""
    return {
        "success": True,
        "data": [
            {"key": key, **template}
            for key, template in ZERO_DAY_PATTERNS.items()
        ],
    }


@router.post("/deploy/{template_key}", status_code=201)
async def deploy_zero_day_template(
    template_key: str,
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Deploy a pre-built zero-day pattern template as an emergency rule."""
    template = ZERO_DAY_PATTERNS.get(template_key)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template '{template_key}' not found")

    data = {
        **template,
        "source": "zero-day-template",
    }
    result = _service.create_rule(db, data)
    return {"success": True, "data": result}
