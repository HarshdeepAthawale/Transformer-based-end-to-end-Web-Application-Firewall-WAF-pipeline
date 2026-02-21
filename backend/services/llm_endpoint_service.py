"""LLM endpoint service: list, create, update, delete, match_request(path, method)."""

import re
from typing import Optional

from sqlalchemy.orm import Session

from backend.models.llm_endpoint import LLMEndpoint


def list_endpoints(db: Session, active_only: bool = False) -> list[LLMEndpoint]:
    q = db.query(LLMEndpoint)
    if active_only:
        q = q.filter(LLMEndpoint.is_active.is_(True))
    return q.order_by(LLMEndpoint.id).all()


def get_endpoint(db: Session, endpoint_id: int) -> Optional[LLMEndpoint]:
    return db.query(LLMEndpoint).filter(LLMEndpoint.id == endpoint_id).first()


def create_endpoint(
    db: Session,
    path_pattern: str,
    methods: str,
    label: str,
    is_active: bool = True,
) -> LLMEndpoint:
    ep = LLMEndpoint(
        path_pattern=path_pattern.strip(),
        methods=methods.strip() or "POST",
        label=label.strip() or "llm",
        is_active=is_active,
    )
    db.add(ep)
    db.commit()
    db.refresh(ep)
    return ep


def update_endpoint(
    db: Session,
    endpoint_id: int,
    path_pattern: Optional[str] = None,
    methods: Optional[str] = None,
    label: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> Optional[LLMEndpoint]:
    ep = get_endpoint(db, endpoint_id)
    if not ep:
        return None
    if path_pattern is not None:
        ep.path_pattern = path_pattern.strip()
    if methods is not None:
        ep.methods = methods.strip()
    if label is not None:
        ep.label = label.strip()
    if is_active is not None:
        ep.is_active = is_active
    db.commit()
    db.refresh(ep)
    return ep


def delete_endpoint(db: Session, endpoint_id: int) -> bool:
    ep = get_endpoint(db, endpoint_id)
    if not ep:
        return False
    db.delete(ep)
    db.commit()
    return True


def match_request(db: Session, path: str, method: str) -> Optional[LLMEndpoint]:
    """
    Find first active LLM endpoint whose path_pattern matches path and method is in methods.
    path_pattern: prefix match (e.g. /api/chat) or regex if starts with ^ or contains (.
    """
    endpoints = list_endpoints(db, active_only=True)
    method_upper = (method or "").upper()
    for ep in endpoints:
        methods_list = [m.strip().upper() for m in ep.methods.split(",") if m.strip()]
        if methods_list and method_upper not in methods_list:
            continue
        pattern = ep.path_pattern.strip()
        if not pattern:
            continue
        # Regex: if pattern contains ( or starts with ^ treat as regex
        if pattern.startswith("^") or "(" in pattern:
            try:
                if re.search(pattern, path):
                    return ep
            except re.error:
                continue
        # Prefix match
        if path.startswith(pattern) or path == pattern.rstrip("/"):
            return ep
    return None
