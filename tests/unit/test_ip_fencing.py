"""Unit tests for IP fencing service."""

import pytest
from datetime import timedelta
from backend.lib.datetime_utils import utc_now
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models.ip_blacklist import IPBlacklist, IPListType, IPBlockType
from backend.services.ip_fencing import IPFencingService

# Ensure models are registered with Base
import backend.models  # noqa: F401


@pytest.fixture
def db_session():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def ip_fencing(db_session):
    """IP fencing service with test database."""
    return IPFencingService(db_session)


def test_ip_not_blocked_when_blacklist_empty(ip_fencing):
    """IP should not be blocked when blacklist is empty."""
    is_blocked, reason = ip_fencing.is_ip_blocked("192.168.1.1")
    assert is_blocked is False
    assert reason is None


def test_ip_blocked_exact_match(ip_fencing, db_session):
    """IP should be blocked when it matches blacklist entry."""
    entry = IPBlacklist(
        org_id=1,
        ip="10.0.0.5",
        list_type=IPListType.BLACKLIST,
        block_type=IPBlockType.PERMANENT,
        is_range=False,
        is_active=True,
        source="manual",
    )
    db_session.add(entry)
    db_session.commit()

    is_blocked, reason = ip_fencing.is_ip_blocked("10.0.0.5")
    assert is_blocked is True
    assert reason is not None


def test_ip_blocked_by_cidr_range(ip_fencing, db_session):
    """IP should be blocked when it falls in blacklisted CIDR range."""
    entry = IPBlacklist(
        org_id=1,
        ip="192.168.1.0/24",
        ip_range="192.168.1.0/24",
        is_range=True,
        list_type=IPListType.BLACKLIST,
        block_type=IPBlockType.PERMANENT,
        is_active=True,
        source="manual",
    )
    db_session.add(entry)
    db_session.commit()

    is_blocked, _ = ip_fencing.is_ip_blocked("192.168.1.100")
    assert is_blocked is True


def test_ip_not_blocked_outside_cidr(ip_fencing, db_session):
    """IP outside CIDR range should not be blocked."""
    entry = IPBlacklist(
        org_id=1,
        ip="192.168.1.0/24",
        ip_range="192.168.1.0/24",
        is_range=True,
        list_type=IPListType.BLACKLIST,
        block_type=IPBlockType.PERMANENT,
        is_active=True,
        source="manual",
    )
    db_session.add(entry)
    db_session.commit()

    is_blocked, _ = ip_fencing.is_ip_blocked("192.168.2.1")
    assert is_blocked is False


def test_whitelist_takes_precedence(ip_fencing, db_session):
    """Whitelisted IP should not be reported as blocked (whitelist check)."""
    entry = IPBlacklist(
        org_id=1,
        ip="172.16.0.1",
        list_type=IPListType.WHITELIST,
        block_type=IPBlockType.PERMANENT,
        is_range=False,
        is_active=True,
        source="manual",
    )
    db_session.add(entry)
    db_session.commit()

    is_whitelisted = ip_fencing.is_ip_whitelisted("172.16.0.1")
    assert is_whitelisted is True


def test_invalid_ip_returns_false(ip_fencing):
    """Invalid IP format should return not blocked."""
    is_blocked, _ = ip_fencing.is_ip_blocked("not-an-ip")
    assert is_blocked is False


def test_expired_temporary_block_not_blocked(ip_fencing, db_session):
    """Expired temporary block should not block IP."""
    entry = IPBlacklist(
        org_id=1,
        ip="10.0.0.10",
        list_type=IPListType.BLACKLIST,
        block_type=IPBlockType.TEMPORARY,
        is_range=False,
        is_active=True,
        expires_at=utc_now() - timedelta(hours=1),
        source="manual",
    )
    db_session.add(entry)
    db_session.commit()

    is_blocked, _ = ip_fencing.is_ip_blocked("10.0.0.10")
    assert is_blocked is False
