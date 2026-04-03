"""
Rate limit configuration service (Feature 9).
"""

from typing import List
from sqlalchemy.orm import Session

from backend.models.rate_limit_config import RateLimitConfig


class RateLimitConfigService:
    def __init__(self, db: Session):
        self.db = db

    def list_all(self, org_id: int | None = None, zone_id: str | None = None, active_only: bool = True) -> List[RateLimitConfig]:
        q = self.db.query(RateLimitConfig)
        if org_id is not None:
            q = q.filter(RateLimitConfig.org_id == org_id)
        if zone_id is not None:
            q = q.filter(RateLimitConfig.zone_id == zone_id)
        if active_only:
            q = q.filter(RateLimitConfig.is_active)
        return q.order_by(RateLimitConfig.id).all()

    def get_by_id(self, config_id: int) -> RateLimitConfig | None:
        return self.db.query(RateLimitConfig).filter(RateLimitConfig.id == config_id).first()

    def get_by_org_and_path(self, org_id: int, path: str) -> RateLimitConfig | None:
        """Get rate limit config for org on specific path."""
        return self.db.query(RateLimitConfig).filter(
            RateLimitConfig.org_id == org_id,
            RateLimitConfig.path_prefix == path,
            RateLimitConfig.is_active == True,
        ).first()

    def create(
        self,
        org_id: int,
        path_prefix: str,
        requests_per_minute: int,
        window_seconds: int = 60,
        burst: int | None = None,
        zone_id: str | None = "default",
        is_active: bool = True,
    ) -> RateLimitConfig:
        r = RateLimitConfig(
            org_id=org_id,
            path_prefix=path_prefix,
            requests_per_minute=requests_per_minute,
            window_seconds=window_seconds,
            burst=burst,
            zone_id=zone_id or "default",
            is_active=is_active,
        )
        self.db.add(r)
        self.db.commit()
        self.db.refresh(r)
        return r

    def update(
        self,
        config_id: int,
        *,
        path_prefix: str | None = None,
        requests_per_minute: int | None = None,
        window_seconds: int | None = None,
        burst: int | None = None,
        zone_id: str | None = None,
        is_active: bool | None = None,
    ) -> RateLimitConfig | None:
        r = self.get_by_id(config_id)
        if not r:
            return None
        if path_prefix is not None:
            r.path_prefix = path_prefix
        if requests_per_minute is not None:
            r.requests_per_minute = requests_per_minute
        if window_seconds is not None:
            r.window_seconds = window_seconds
        if burst is not None:
            r.burst = burst
        if zone_id is not None:
            r.zone_id = zone_id
        if is_active is not None:
            r.is_active = is_active
        self.db.commit()
        self.db.refresh(r)
        return r

    def delete(self, config_id: int, org_id: int | None = None) -> bool:
        r = self.get_by_id(config_id)
        if not r:
            return False
        if org_id is not None and r.org_id != org_id:
            return False
        self.db.delete(r)
        self.db.commit()
        return True
