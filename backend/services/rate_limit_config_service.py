"""
Rate limit configuration service (Feature 9).
"""

from typing import List
from sqlalchemy.orm import Session

from backend.models.rate_limit_config import RateLimitConfig


class RateLimitConfigService:
    def __init__(self, db: Session):
        self.db = db

    def list_all(self, zone_id: str | None = None, active_only: bool = True) -> List[RateLimitConfig]:
        q = self.db.query(RateLimitConfig)
        if zone_id is not None:
            q = q.filter(RateLimitConfig.zone_id == zone_id)
        if active_only:
            q = q.filter(RateLimitConfig.is_active)
        return q.order_by(RateLimitConfig.id).all()

    def get_by_id(self, config_id: int) -> RateLimitConfig | None:
        return self.db.query(RateLimitConfig).filter(RateLimitConfig.id == config_id).first()

    def create(
        self,
        path_prefix: str,
        requests_per_minute: int,
        window_seconds: int = 60,
        burst: int | None = None,
        zone_id: str | None = "default",
        is_active: bool = True,
    ) -> RateLimitConfig:
        r = RateLimitConfig(
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

    def delete(self, config_id: int) -> bool:
        r = self.get_by_id(config_id)
        if not r:
            return False
        self.db.delete(r)
        self.db.commit()
        return True
