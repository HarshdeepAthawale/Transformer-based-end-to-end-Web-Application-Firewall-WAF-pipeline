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
    """Initialize database tables and run migrations"""
    # Import all models to ensure they're registered with Base
    import backend.models  # noqa: F401

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Migration: add any missing security_events columns (bot_score, attack_score, block_duration_seconds)
    _migrate_security_events_columns()

    # Migration: add rule_packs and security_rules managed-rules columns
    _migrate_managed_rules_tables()

    # Seed default bot score bands if empty
    _seed_bot_score_bands()

    # Seed default Firewall-for-AI prompt-injection pattern if empty (no hardcoded patterns in app code)
    _seed_firewall_ai_patterns()

    logger.info("Database initialized successfully - all tables created")


def _migrate_security_events_columns():
    """Add missing columns to security_events (attack_score, block_duration_seconds, bot_score)."""
    from sqlalchemy import text

    required_columns = [
        ("attack_score", "INTEGER"),
        ("block_duration_seconds", "INTEGER"),
        ("bot_score", "INTEGER"),
    ]
    with engine.connect() as conn:
        if DATABASE_URL.startswith("sqlite"):
            result = conn.execute(text("PRAGMA table_info(security_events)"))
            columns = [row[1] for row in result]
            for col_name, col_type in required_columns:
                if col_name not in columns:
                    conn.execute(text(f"ALTER TABLE security_events ADD COLUMN {col_name} {col_type}"))
                    conn.commit()
                    logger.info(f"Migration: added {col_name} column to security_events")
        else:
            # PostgreSQL
            for col_name, col_type in required_columns:
                result = conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'security_events' AND column_name = :name"
                    ),
                    {"name": col_name},
                )
                if result.fetchone() is None:
                    conn.execute(text(f"ALTER TABLE security_events ADD COLUMN {col_name} {col_type}"))
                    conn.commit()
                    logger.info(f"Migration: added {col_name} column to security_events")


def _migrate_managed_rules_tables():
    """Create rule_packs and add rule_pack_id, rule_pack_version, external_id to security_rules."""
    from sqlalchemy import text

    with engine.connect() as conn:
        if DATABASE_URL.startswith("sqlite"):
            result = conn.execute(text("PRAGMA table_info(security_rules)"))
            columns = [row[1] for row in result]
            for col_name, col_type in [
                ("rule_pack_id", "INTEGER"),
                ("rule_pack_version", "VARCHAR(100)"),
                ("external_id", "VARCHAR(200)"),
            ]:
                if col_name not in columns:
                    conn.execute(text(f"ALTER TABLE security_rules ADD COLUMN {col_name} {col_type}"))
                    conn.commit()
                    logger.info(f"Migration: added {col_name} to security_rules")
        else:
            for col_name, col_type in [
                ("rule_pack_id", "INTEGER"),
                ("rule_pack_version", "VARCHAR(100)"),
                ("external_id", "VARCHAR(200)"),
            ]:
                result = conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'security_rules' AND column_name = :name"
                    ),
                    {"name": col_name},
                )
                if result.fetchone() is None:
                    conn.execute(text(f"ALTER TABLE security_rules ADD COLUMN {col_name} {col_type}"))
                    conn.commit()
                    logger.info(f"Migration: added {col_name} to security_rules")


def _seed_firewall_ai_patterns():
    """Seed one default prompt-injection pattern if firewall_ai_patterns table is empty."""
    try:
        from backend.models.firewall_ai_pattern import FirewallAIPattern

        db = SessionLocal()
        try:
            count = db.query(FirewallAIPattern).count()
            if count == 0:
                db.add(
                    FirewallAIPattern(
                        pattern_type="prompt_injection",
                        pattern_value="ignore previous instructions",
                        is_active=True,
                        source="manual",
                    )
                )
                db.commit()
                logger.info("Seeded default Firewall-for-AI prompt-injection pattern")
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"Firewall AI patterns seed skipped: {e}")


def _seed_bot_score_bands():
    """Seed default score bands if bot_score_bands table is empty."""
    from backend.models.bot_score_bands import BotScoreBand

    db = SessionLocal()
    try:
        count = db.query(BotScoreBand).count()
        if count == 0:
            from backend.config import config

            block_max = config.BOT_BAND_BLOCK_MAX
            challenge_max = config.BOT_BAND_CHALLENGE_MAX
            bands = [
                BotScoreBand(min_score=1, max_score=block_max, action="block", priority=1),
                BotScoreBand(
                    min_score=block_max + 1,
                    max_score=challenge_max,
                    action="challenge",
                    priority=2,
                ),
                BotScoreBand(min_score=challenge_max + 1, max_score=99, action="allow", priority=3),
            ]
            for b in bands:
                db.add(b)
            db.commit()
            logger.info("Seeded default bot score bands")
    finally:
        db.close()


def close_db():
    """Close database connections"""
    engine.dispose()
    logger.info("Database connections closed")
