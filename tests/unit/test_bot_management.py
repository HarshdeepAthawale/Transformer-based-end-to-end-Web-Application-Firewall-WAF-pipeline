"""Unit tests for bot management (bot score, verified bots, score bands)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.database import Base
import backend.models  # noqa: F401 - register all models
from backend.services.bot_detection import BotDetectionService
from backend.services.verified_bots_service import VerifiedBotsService
from backend.services.bot_score_bands_service import BotScoreBandsService
from backend.models.bot_signatures import BotSignature, BotCategory
from backend.models.verified_bots import VerifiedBot
from backend.models.bot_score_bands import BotScoreBand


@pytest.fixture
def db_session():
    from backend.config import config

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Seed score bands from config
    block_max = config.BOT_BAND_BLOCK_MAX
    challenge_max = config.BOT_BAND_CHALLENGE_MAX
    session.add(BotScoreBand(min_score=1, max_score=block_max, action="block", priority=1))
    session.add(
        BotScoreBand(
            min_score=block_max + 1,
            max_score=challenge_max,
            action="challenge",
            priority=2,
        )
    )
    session.add(
        BotScoreBand(min_score=challenge_max + 1, max_score=99, action="allow", priority=3)
    )
    session.commit()

    yield session
    session.close()


def test_bot_score_unknown_user_agent(db_session):
    """Unknown User-Agent gets BOT_DEFAULT_SCORE_UNKNOWN from config and band action."""
    from backend.config import config

    service = BotDetectionService(db_session)
    result = service.detect_bot(
        user_agent="Mozilla/5.0 (compatible; SomeBrowser/1.0)",
        ip="1.2.3.4",
        headers={"Accept-Language": "en", "Accept-Encoding": "gzip"},
    )
    assert "bot_score" in result
    assert 1 <= result["bot_score"] <= 99
    assert result["bot_score"] == config.BOT_DEFAULT_SCORE_UNKNOWN
    assert result["action"] in ("allow", "challenge", "block")


def test_bot_score_missing_user_agent(db_session):
    """Missing User-Agent gets BOT_SCORE_MISSING_UA from config."""
    from backend.config import config

    service = BotDetectionService(db_session)
    result = service.detect_bot(user_agent="", ip="1.2.3.4", headers={})
    assert result["is_bot"] is True
    assert result["bot_score"] == config.BOT_SCORE_MISSING_UA
    assert result["action"] in ("block", "challenge")


def test_verified_bot_high_score(db_session):
    """Verified bot gets BOT_SCORE_VERIFIED from config."""
    from backend.config import config

    db_session.add(
        VerifiedBot(name="Googlebot", user_agent_pattern=r".*Googlebot.*", source="manual")
    )
    db_session.commit()

    service = BotDetectionService(db_session)
    result = service.detect_bot(
        user_agent="Mozilla/5.0 (compatible; Googlebot/2.1)",
        ip="1.2.3.4",
        headers={},
    )
    assert result["is_verified_bot"] is True
    assert result["bot_score"] == config.BOT_SCORE_VERIFIED


def test_score_bands_action(db_session):
    """Score bands return correct action for given score (from config ranges)."""
    from backend.config import config

    svc = BotScoreBandsService(db_session)
    block_max = config.BOT_BAND_BLOCK_MAX
    challenge_max = config.BOT_BAND_CHALLENGE_MAX
    assert svc.get_action_for_score(block_max) == "block"
    assert svc.get_action_for_score((block_max + challenge_max) // 2) == "challenge"
    assert svc.get_action_for_score(challenge_max + 1) == "allow"


def test_score_bands_update(db_session):
    """Update bands replaces all bands."""
    svc = BotScoreBandsService(db_session)
    bands = svc.update_bands([
        {"min_score": 1, "max_score": 39, "action": "block"},
        {"min_score": 40, "max_score": 99, "action": "allow"},
    ])
    assert len(bands) == 2
    assert svc.get_action_for_score(20) == "block"
    assert svc.get_action_for_score(80) == "allow"
