"""WAF API request/response schemas."""
from pydantic import BaseModel
from typing import Optional, Dict


class CheckRequest(BaseModel):
    method: str
    path: str
    query_params: Optional[Dict] = {}
    headers: Optional[Dict] = {}
    body: Optional[str] = None


class CheckResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    processing_time_ms: float
    model_version: Optional[str] = None
