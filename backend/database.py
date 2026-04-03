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

# Ensure SQLite DB directory exists (SQLAlchemy does not create parent dirs)
if DATABASE_URL.startswith("sqlite:///"):
    db_path = Path(DATABASE_URL.replace("sqlite:///", ""))
    if not db_path.is_absolute():
        db_path = Path(__file__).parent.parent / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Seed default organization (for multi-tenancy)
    _seed_default_organization()

    # Seed default bot score bands if empty
    _seed_bot_score_bands()

    # Seed default Firewall-for-AI prompt-injection pattern if empty (no hardcoded patterns in app code)
    _seed_firewall_ai_patterns()

    # Seed first admin user if users table is empty (optional: set SEED_ADMIN=false to disable)
    _seed_admin_user()

    logger.info("Database initialized successfully - all tables created")


# Whitelisted identifiers for dynamic SQL migrations (prevent injection if extended)
_SAFE_COL_NAMES = frozenset({
    "attack_score", "block_duration_seconds", "bot_score",
    "rule_pack_id", "rule_pack_version", "external_id",
})
_SAFE_COL_TYPES = frozenset({
    "INTEGER", "VARCHAR(100)", "VARCHAR(200)",
})


def _validate_migration_column(col_name: str, col_type: str):
    """Ensure column name and type are in the safe whitelist."""
    if col_name not in _SAFE_COL_NAMES:
        raise ValueError(f"Unsafe migration column name: {col_name}")
    if col_type not in _SAFE_COL_TYPES:
        raise ValueError(f"Unsafe migration column type: {col_type}")


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
                _validate_migration_column(col_name, col_type)
                if col_name not in columns:
                    conn.execute(text(f"ALTER TABLE security_events ADD COLUMN {col_name} {col_type}"))
                    conn.commit()
                    logger.info(f"Migration: added {col_name} column to security_events")
        else:
            # PostgreSQL
            for col_name, col_type in required_columns:
                _validate_migration_column(col_name, col_type)
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
                _validate_migration_column(col_name, col_type)
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
                _validate_migration_column(col_name, col_type)
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


def _seed_admin_user():
    """Seed one admin user if users table is empty. Guard with SEED_ADMIN=false to disable in production."""
    import secrets as _secrets

    if os.getenv("SEED_ADMIN", "true").lower() == "false":
        return
    try:
        from backend.models.users import User, UserRole

        db = SessionLocal()
        try:
            if db.query(User).count() > 0:
                return
            email = os.getenv("ADMIN_EMAIL", "admin@waf.example")
            password = os.getenv("ADMIN_PASSWORD", "")
            if not password or password == "admin123":
                # Generate a strong random password instead of using a weak default
                password = _secrets.token_urlsafe(20)
                logger.warning(
                    f"ADMIN_PASSWORD not set or is default. Generated random admin password: {password}  "
                    "Copy this now — it will not be shown again. Set ADMIN_PASSWORD in .env to use your own."
                )
            if len(password) < 10:
                logger.warning("ADMIN_PASSWORD is shorter than 10 characters. Consider using a stronger password.")
            username = email.split("@")[0] if "@" in email else email
            if db.query(User).filter(User.username == username).first() or db.query(User).filter(User.email == email).first():
                return
            user = User(
                username=username,
                email=email,
                role=UserRole.ADMIN,
                full_name="Admin",
                created_by="seed",
            )
            user.set_password(password)
            db.add(user)
            db.commit()
            logger.info("Seeded first admin user (set SEED_ADMIN=false to disable)")
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"Admin user seed skipped: {e}")


def _seed_default_organization():
    """Seed a default organization for existing single-tenant data.

    All existing data needs to belong to some organization. This seeds org_id=1
    as the "Default Organization" for all pre-multi-tenancy data.
    """
    try:
        from backend.models.organization import Organization

        db = SessionLocal()
        try:
            # Check if any org already exists
            existing = db.query(Organization).first()
            if existing:
                logger.info(f"Organization already exists: {existing.name} (id={existing.id})")
                return existing.id

            # Seed default org
            default_org = Organization(
                name="Default Organization",
                slug="default",
                plan="starter",
                is_active=True,
                owner_email=None,
            )
            db.add(default_org)
            db.commit()
            logger.info(f"Seeded default organization: id={default_org.id}, name='{default_org.name}'")
            return default_org.id
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"Default organization seed skipped: {e}")
        return 1


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
