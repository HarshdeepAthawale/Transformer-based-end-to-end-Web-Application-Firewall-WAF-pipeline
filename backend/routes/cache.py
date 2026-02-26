"""
Phase 1 — Cache purge and analytics API.
POST /api/v1/cache/purge — purge by URL, tag, prefix, or everything (publishes to Redis for gateways).
GET /api/v1/cache/analytics — hit ratio, bandwidth saved, TTFB reduction (from Redis stats).
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Union

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.config import config
from backend.database import get_db
from backend.models import CachePurgeLog

router = APIRouter()

REDIS_PREFIX = "waf:cache:"
PURGE_CHANNEL = "waf:cache:purge"


class PurgeUrlsItem(BaseModel):
    method: str = "GET"
    url: str


class CachePurgeRequest(BaseModel):
    urls: Optional[list[Union[str, PurgeUrlsItem]]] = None  # ["/path"] or [{"method":"GET","url":"/path"}]
    tags: Optional[list[str]] = None
    prefixes: Optional[list[str]] = None
    purge_everything: bool = False


class CachePurgeResponse(BaseModel):
    success: bool
    message: str
    keys_purged_estimate: Optional[int] = None
    purge_id: Optional[int] = None


def _get_redis():
    try:
        return aioredis.from_url(config.REDIS_URL, decode_responses=True, socket_timeout=5)
    except Exception:
        return None


async def _publish_purge(cmd: dict) -> None:
    r = _get_redis()
    if not r:
        return
    try:
        await r.publish(PURGE_CHANNEL, json.dumps(cmd))
    finally:
        await r.aclose()


@router.post("/purge", response_model=CachePurgeResponse)
async def cache_purge(
    body: CachePurgeRequest,
    db: Session = Depends(get_db),
):
    """Purge edge cache by URL(s), tag(s), prefix(es), or everything. Gateways subscribe to Redis and apply purge."""
    if body.purge_everything:
        cmd = {"type": "all", "values": []}
        await _publish_purge(cmd)
        log = CachePurgeLog(
            purge_type="all",
            purge_value=None,
            keys_purged=0,
            requested_by=None,
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return CachePurgeResponse(
            success=True,
            message="Purge-all requested; all edge caches will clear.",
            keys_purged_estimate=None,
            purge_id=log.id,
        )

    values_url: list[dict[str, str]] = []
    for u in body.urls or []:
        if isinstance(u, PurgeUrlsItem):
            values_url.append({"method": u.method, "url": u.url})
        elif isinstance(u, dict):
            values_url.append({"method": u.get("method", "GET"), "url": u.get("url", "")})
        else:
            values_url.append({"method": "GET", "url": str(u)})

    if values_url:
        cmd = {"type": "url", "values": values_url}
        await _publish_purge(cmd)
        log = CachePurgeLog(
            purge_type="url",
            purge_value=json.dumps(values_url)[:500],
            keys_purged=len(values_url),
            requested_by=None,
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return CachePurgeResponse(
            success=True,
            message=f"Purge by URL requested for {len(values_url)} URL(s).",
            keys_purged_estimate=len(values_url),
            purge_id=log.id,
        )

    if body.tags:
        cmd = {"type": "tag", "values": body.tags}
        await _publish_purge(cmd)
        log = CachePurgeLog(
            purge_type="tag",
            purge_value=",".join(body.tags)[:500],
            keys_purged=len(body.tags),
            requested_by=None,
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return CachePurgeResponse(
            success=True,
            message=f"Purge by tag requested for {len(body.tags)} tag(s).",
            keys_purged_estimate=len(body.tags),
            purge_id=log.id,
        )

    if body.prefixes:
        cmd = {"type": "prefix", "values": body.prefixes}
        await _publish_purge(cmd)
        log = CachePurgeLog(
            purge_type="prefix",
            purge_value=",".join(body.prefixes)[:500],
            keys_purged=len(body.prefixes),
            requested_by=None,
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return CachePurgeResponse(
            success=True,
            message=f"Purge by prefix requested for {len(body.prefixes)} prefix(es).",
            keys_purged_estimate=len(body.prefixes),
            purge_id=log.id,
        )

    raise HTTPException(status_code=400, detail="Provide urls, tags, prefixes, or purge_everything=true.")


@router.get("/analytics")
async def cache_analytics():
    """Return cache analytics: hit ratio, bandwidth saved, per-day stats from Redis (written by gateways)."""
    r = _get_redis()
    if not r:
        return {
            "available": False,
            "message": "Redis not configured or unavailable",
            "hit_ratio": None,
            "bandwidth_saved_bytes": None,
            "by_day": [],
        }

    try:
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        hits_key = f"{REDIS_PREFIX}stats:{today}:hits"
        misses_key = f"{REDIS_PREFIX}stats:{today}:misses"
        bytes_key = f"{REDIS_PREFIX}stats:{today}:bytes_saved"

        hits = int(await r.get(hits_key) or 0)
        misses = int(await r.get(misses_key) or 0)
        bytes_saved = int(await r.get(bytes_key) or 0)
        total = hits + misses
        hit_ratio = round(hits / total, 4) if total else None

        # Optional: last 7 days (scan keys)
        by_day: list[dict[str, Any]] = []
        for i in range(7):
            d = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y%m%d")
            h = int(await r.get(f"{REDIS_PREFIX}stats:{d}:hits") or 0)
            m = int(await r.get(f"{REDIS_PREFIX}stats:{d}:misses") or 0)
            b = int(await r.get(f"{REDIS_PREFIX}stats:{d}:bytes_saved") or 0)
            by_day.append({
                "date": d,
                "hits": h,
                "misses": m,
                "hit_ratio": round(h / (h + m), 4) if (h + m) else None,
                "bandwidth_saved_bytes": b,
            })

        return {
            "available": True,
            "hit_ratio": hit_ratio,
            "bandwidth_saved_bytes": bytes_saved,
            "hits_today": hits,
            "misses_today": misses,
            "by_day": by_day,
        }
    except Exception as e:
        return {
            "available": False,
            "message": str(e),
            "hit_ratio": None,
            "bandwidth_saved_bytes": None,
            "by_day": [],
        }
    finally:
        await r.aclose()
