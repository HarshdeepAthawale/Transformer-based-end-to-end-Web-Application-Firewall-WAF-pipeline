"""
Database utility helpers - dialect-aware SQL functions.
"""
from sqlalchemy import func

from backend.database import DATABASE_URL


def hour_bucket(column):
    """Dialect-aware hourly time bucket.

    SQLite: func.strftime("%Y-%m-%d %H:00:00", column)
    PostgreSQL: func.date_trunc("hour", column)
    """
    if DATABASE_URL.startswith("sqlite"):
        return func.strftime("%Y-%m-%d %H:00:00", column)
    else:
        return func.date_trunc("hour", column)
