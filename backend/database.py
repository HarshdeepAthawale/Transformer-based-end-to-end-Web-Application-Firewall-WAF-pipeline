"""
Database connection and session management
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import os
from pathlib import Path
from loguru import logger

# Database URL - use SQLite for development, PostgreSQL for production
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{Path(__file__).parent.parent}/data/waf_dashboard.db"
)

# Create engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False
    )
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    # Import all models to ensure they're registered with Base
    from backend.models import (
        metrics, alerts, traffic, threats, activities,
        ip_blacklist, ip_reputation, geo_rules, bot_signatures,
        threat_intel, security_rules, users, audit_log
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully - all tables created")


def close_db():
    """Close database connections"""
    engine.dispose()
    logger.info("Database connections closed")
