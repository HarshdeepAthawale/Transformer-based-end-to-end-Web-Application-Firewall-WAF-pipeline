"""
Usage tracking service - Redis-backed per-org request counting.
Uses Redis INCR for O(1) per-request counting.
Background flush task writes Redis counters to PostgreSQL every 30 seconds.
"""
import os
from typing import Optional

from loguru import logger
from sqlalchemy.orm import Session

from backend.models.usage import UsageRecord
from backend.models.subscription import Subscription
from backend.models.billing_plan import BillingPlan
from backend.lib.datetime_utils import utc_now


_redis_client = None
_redis_available = None  # None = not checked yet


def _current_month() -> str:
    """Return current month as YYYY-MM string."""
    return utc_now().strftime("%Y-%m")


def _usage_key(org_id: int, month: str = None) -> str:
    """Redis key for usage counter."""
    return f"usage:{org_id}:{month or _current_month()}"


def _get_redis():
    """Get or create Redis client. Returns None if unavailable."""
    global _redis_client, _redis_available
    if _redis_available is False:
        return None
    if _redis_client is not None:
        return _redis_client

    try:
        import redis
        url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _redis_client = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
        _redis_client.ping()
        _redis_available = True
        logger.info("Usage service: Redis connected")
        return _redis_client
    except Exception as e:
        logger.warning(f"Usage service: Redis unavailable ({e}); falling back to DB writes")
        _redis_available = False
        return None


def increment_usage_redis(org_id: int, count: int = 1) -> int:
    """Increment usage via Redis INCR. O(1), no DB write. Returns new total."""
    r = _get_redis()
    if r is None:
        return -1  # Signal caller to use DB fallback

    key = _usage_key(org_id)
    try:
        new_total = r.incrby(key, count)
        # Set expiry to 35 days (covers current month + buffer)
        r.expire(key, 35 * 86400)
        return new_total
    except Exception as e:
        logger.warning(f"Redis INCR failed for org {org_id}: {e}")
        return -1


def get_usage_redis(org_id: int) -> int:
    """Get current month usage from Redis. Returns -1 if unavailable."""
    r = _get_redis()
    if r is None:
        return -1

    try:
        val = r.get(_usage_key(org_id))
        return int(val) if val else 0
    except Exception:
        return -1


def is_within_limit_redis(org_id: int, limit: int) -> bool:
    """Check usage against limit via Redis. Fast path."""
    if limit < 0:  # Unlimited
        return True

    current = get_usage_redis(org_id)
    if current < 0:  # Redis unavailable
        return True  # Fail open

    return current <= limit


class UsageService:
    """Track and enforce per-org request usage against plan limits."""

    def __init__(self, db: Session):
        self.db = db

    def increment_usage(self, org_id: int, count: int = 1) -> int:
        """Increment usage. Uses Redis if available, DB fallback otherwise."""
        result = increment_usage_redis(org_id, count)
        if result >= 0:
            return result

        # DB fallback
        month = _current_month()
        record = self.db.query(UsageRecord).filter(
            UsageRecord.org_id == org_id,
            UsageRecord.month == month,
        ).first()

        if record:
            record.requests_count += count
            record.last_updated = utc_now()
        else:
            record = UsageRecord(
                org_id=org_id,
                month=month,
                requests_count=count,
            )
            self.db.add(record)

        self.db.commit()
        return record.requests_count

    def get_usage(self, org_id: int, month: Optional[str] = None) -> dict:
        """Get usage stats. Redis first, DB fallback."""
        month = month or _current_month()
        limit = self._get_plan_limit(org_id)

        # Try Redis for current month
        if month == _current_month():
            redis_count = get_usage_redis(org_id)
            if redis_count >= 0:
                return self._build_usage_dict(org_id, month, redis_count, limit)

        # DB fallback
        record = self.db.query(UsageRecord).filter(
            UsageRecord.org_id == org_id,
            UsageRecord.month == month,
        ).first()
        requests_count = record.requests_count if record else 0

        return self._build_usage_dict(org_id, month, requests_count, limit)

    def _build_usage_dict(self, org_id: int, month: str, requests_count: int, limit: int) -> dict:
        return {
            "org_id": org_id,
            "month": month,
            "requests_count": requests_count,
            "requests_limit": limit,
            "usage_percent": round((requests_count / limit) * 100, 1) if limit > 0 else 0,
            "within_limit": limit < 0 or requests_count <= limit,
        }

    def is_within_limit(self, org_id: int) -> bool:
        """Check if org is within plan limit. Redis fast path."""
        limit = self._get_plan_limit(org_id)
        if limit < 0:
            return True

        # Try Redis fast path
        result = is_within_limit_redis(org_id, limit)
        if _redis_available:
            return result

        # DB fallback
        month = _current_month()
        record = self.db.query(UsageRecord).filter(
            UsageRecord.org_id == org_id,
            UsageRecord.month == month,
        ).first()
        current = record.requests_count if record else 0
        return current <= limit

    def _get_plan_limit(self, org_id: int) -> int:
        """Get the request limit for org's current plan. Returns -1 for unlimited."""
        sub = self.db.query(Subscription).filter(
            Subscription.org_id == org_id,
            Subscription.status.in_(["active", "pending"]),
        ).first()

        if not sub:
            free_plan = self.db.query(BillingPlan).filter(BillingPlan.slug == "free").first()
            return free_plan.requests_limit if free_plan else 10000

        plan = self.db.query(BillingPlan).filter(BillingPlan.id == sub.plan_id).first()
        return plan.requests_limit if plan else 10000

    def reset_monthly_usage(self) -> int:
        """Reset usage for previous months (cleanup)."""
        current = _current_month()
        old_records = self.db.query(UsageRecord).filter(
            UsageRecord.month != current
        ).all()
        count = len(old_records)
        for record in old_records:
            self.db.delete(record)
        if count > 0:
            self.db.commit()
            logger.info(f"Cleaned {count} old usage records")
        return count

    @staticmethod
    def flush_redis_to_db(db: Session) -> int:
        """Flush all Redis usage counters to PostgreSQL. Called by background task."""
        r = _get_redis()
        if r is None:
            return 0

        flushed = 0
        month = _current_month()

        try:
            # Scan for all usage keys
            cursor = 0
            keys = []
            while True:
                cursor, batch = r.scan(cursor, match="usage:*", count=100)
                keys.extend(batch)
                if cursor == 0:
                    break

            for key in keys:
                try:
                    parts = key.split(":")
                    if len(parts) != 3:
                        continue
                    org_id = int(parts[1])
                    key_month = parts[2]

                    val = r.get(key)
                    if not val:
                        continue
                    count = int(val)

                    # Upsert to DB
                    record = db.query(UsageRecord).filter(
                        UsageRecord.org_id == org_id,
                        UsageRecord.month == key_month,
                    ).first()

                    if record:
                        record.requests_count = count
                        record.last_updated = utc_now()
                    else:
                        record = UsageRecord(
                            org_id=org_id,
                            month=key_month,
                            requests_count=count,
                        )
                        db.add(record)

                    flushed += 1
                except Exception as e:
                    logger.warning(f"Failed to flush key {key}: {e}")

            if flushed > 0:
                db.commit()
                logger.info(f"Flushed {flushed} usage counters from Redis to DB")

        except Exception as e:
            logger.error(f"Usage flush failed: {e}")

        return flushed
