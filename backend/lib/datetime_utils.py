"""UTC time helper; replaces deprecated datetime.utcnow() for Python 3.12+."""

from datetime import datetime, timezone


def utc_now():
    """Return current UTC time (timezone-aware). Use for default= and onupdate= in models and app code."""
    return datetime.now(timezone.utc)
