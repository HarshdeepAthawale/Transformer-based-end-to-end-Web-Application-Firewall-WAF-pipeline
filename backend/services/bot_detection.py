"""
Bot Detection Service - bot score (1-99) and verified bots integration.
"""
from sqlalchemy.orm import Session
from typing import Dict, List

import re
from backend.lib.datetime_utils import utc_now

from loguru import logger

from backend.models.bot_signatures import BotSignature, BotCategory
from backend.services.verified_bots_service import VerifiedBotsService
from backend.services.bot_score_bands_service import BotScoreBandsService


def _clamp_score(score: int) -> int:
    return max(1, min(99, score))


class BotDetectionService:
    """Service for bot detection"""
    
    def __init__(self, db: Session):
        self.db = db
        # Signature cache removed for multi-tenancy (fetch per-org on detect)
    
    def detect_bot(self, org_id: int, user_agent: str, ip: str, headers: Dict) -> Dict:
        """
        Detect if request is from a bot. Returns bot_score (1-99), action from bands.
        industry-standard: low score = automated, high score = human.
        Returns: {
            'is_bot': bool,
            'bot_name': str,
            'category': str,
            'is_whitelisted': bool,
            'is_verified_bot': bool,
            'action': str,  # block, allow, challenge, monitor
            'confidence': float,
            'bot_score': int,  # 1-99
            'matched_signature': str | None,
        }
        """
        from backend.config import config

        default_score = _clamp_score(config.BOT_DEFAULT_SCORE_UNKNOWN)
        score_verified = _clamp_score(config.BOT_SCORE_VERIFIED)
        score_missing_ua = _clamp_score(config.BOT_SCORE_MISSING_UA)
        score_behavioral = _clamp_score(config.BOT_SCORE_BEHAVIORAL)
        score_signature_matched = _clamp_score(config.BOT_SCORE_SIGNATURE_MATCHED)

        verified_svc = VerifiedBotsService(self.db)
        bands_svc = BotScoreBandsService(self.db)

        # Check verified bots first - known-good bots get high score from config
        is_verified, verified_name = verified_svc.is_verified(user_agent or "")
        if is_verified:
            score = score_verified
            action = bands_svc.get_action_for_score(score)
            return {
                "is_bot": True,
                "bot_name": verified_name,
                "category": "verified",
                "is_whitelisted": True,
                "is_verified_bot": True,
                "action": action,
                "confidence": 0.95,
                "bot_score": score,
                "matched_signature": verified_name,
            }

        if not user_agent:
            # Missing User-Agent - score from config
            score = score_missing_ua
            action = bands_svc.get_action_for_score(score)
            return {
                "is_bot": True,
                "bot_name": "Missing User-Agent",
                "category": "unknown",
                "is_whitelisted": False,
                "is_verified_bot": False,
                "action": action,
                "confidence": 0.7,
                "bot_score": score,
                "matched_signature": None,
            }

        # Check against org-specific signatures (no caching for multi-tenancy)
        signatures = self.db.query(BotSignature).filter(BotSignature.is_active).all()
        for signature in signatures:
            try:
                if re.search(signature.user_agent_pattern, user_agent, re.IGNORECASE):
                    signature.detection_count += 1
                    signature.last_detected = utc_now()
                    self.db.commit()

                    if signature.is_whitelisted:
                        score = score_verified
                    else:
                        score = score_signature_matched
                    action = bands_svc.get_action_for_score(score)
                    if signature.is_whitelisted:
                        action = "allow"

                    return {
                        "is_bot": True,
                        "bot_name": signature.name,
                        "category": signature.category.value if signature.category else "unknown",
                        "is_whitelisted": signature.is_whitelisted,
                        "is_verified_bot": False,
                        "action": action,
                        "confidence": 0.9,
                        "bot_score": score,
                        "matched_signature": signature.name,
                    }
            except re.error:
                logger.warning(f"Invalid regex pattern in signature {signature.id}: {signature.user_agent_pattern}")
                continue

        # Behavioral checks
        behavioral_result = self._check_behavioral_patterns(
            user_agent, headers, bands_svc, score_behavioral
        )
        if behavioral_result["is_bot"]:
            return behavioral_result

        # No match - use default score
        action = bands_svc.get_action_for_score(default_score)
        return {
            "is_bot": False,
            "bot_name": None,
            "category": None,
            "is_whitelisted": False,
            "is_verified_bot": False,
            "action": action,
            "confidence": 0.0,
            "bot_score": default_score,
            "matched_signature": None,
        }
    
    def _check_behavioral_patterns(
        self,
        user_agent: str,
        headers: Dict,
        bands_svc: BotScoreBandsService,
        score_behavioral: int,
    ) -> Dict:
        """Check behavioral patterns that indicate bots."""
        suspicious_indicators = 0

        # Missing common headers
        if "Accept-Language" not in headers:
            suspicious_indicators += 1
        if "Accept-Encoding" not in headers:
            suspicious_indicators += 1

        # Suspicious User-Agent patterns
        suspicious_patterns = [
            r"bot",
            r"crawler",
            r"spider",
            r"scraper",
            r"curl",
            r"wget",
            r"python",
            r"java",
            r"^$",
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, user_agent, re.IGNORECASE):
                suspicious_indicators += 1
                break

        if suspicious_indicators >= 2:
            score = score_behavioral
            action = bands_svc.get_action_for_score(score)
            return {
                "is_bot": True,
                "bot_name": "Suspicious Behavior",
                "category": "unknown",
                "is_whitelisted": False,
                "is_verified_bot": False,
                "action": action,
                "confidence": 0.6,
                "bot_score": score,
                "matched_signature": None,
            }

        return {
            "is_bot": False,
            "bot_name": None,
            "category": None,
            "is_whitelisted": False,
            "action": "allow",
            "confidence": 0.0,
        }
    
    def add_signature(
        self,
        org_id: int,
        user_agent_pattern: str,
        name: str,
        category: BotCategory,
        action: str = "block",
        is_whitelisted: bool = False
    ) -> BotSignature:
        """Add bot signature (global, but org_id used for audit logging)"""
        signature = BotSignature(
            user_agent_pattern=user_agent_pattern,
            name=name,
            category=category,
            action=action,
            is_whitelisted=is_whitelisted
        )

        self.db.add(signature)
        self.db.commit()
        self.db.refresh(signature)

        return signature

    def get_signatures(self, org_id: int, active_only: bool = True) -> List[BotSignature]:
        """Get bot signatures (global list, org_id used for audit)"""
        query = self.db.query(BotSignature)
        if active_only:
            query = query.filter(BotSignature.is_active)
        return query.order_by(BotSignature.timestamp.desc()).all()

    def get_signature_by_id(self, signature_id: int) -> BotSignature | None:
        """Get a single signature by id."""
        return self.db.query(BotSignature).filter(BotSignature.id == signature_id).first()

    def update_signature(
        self,
        signature_id: int,
        *,
        user_agent_pattern: str | None = None,
        name: str | None = None,
        category: BotCategory | None = None,
        action: str | None = None,
        is_whitelisted: bool | None = None,
        is_active: bool | None = None,
    ) -> BotSignature | None:
        """Update a signature by id. Returns None if not found."""
        sig = self.get_signature_by_id(signature_id)
        if not sig:
            return None
        if user_agent_pattern is not None:
            sig.user_agent_pattern = user_agent_pattern
        if name is not None:
            sig.name = name
        if category is not None:
            sig.category = category
        if action is not None:
            sig.action = action
        if is_whitelisted is not None:
            sig.is_whitelisted = is_whitelisted
        if is_active is not None:
            sig.is_active = is_active
        self.db.commit()
        self.db.refresh(sig)
        return sig

    def delete_signature(self, signature_id: int) -> bool:
        """Delete a signature by id. Returns False if not found."""
        sig = self.get_signature_by_id(signature_id)
        if not sig:
            return False
        self.db.delete(sig)
        self.db.commit()
        return True

    def get_bot_stats(self, org_id: int, db: Session, range_str: str) -> dict:
        """Per-org bot score distribution from SecurityEvent table."""
        from backend.core.time_range import parse_time_range
        from backend.models.security_event import SecurityEvent

        start_time, _ = parse_time_range(range_str)

        # Count bot events by action
        bot_events = db.query(SecurityEvent).filter(
            SecurityEvent.org_id == org_id,
            SecurityEvent.event_type.in_(["bot_block", "bot_challenge"]),
            SecurityEvent.timestamp >= start_time
        ).all()

        block_count = sum(1 for e in bot_events if e.event_type == "bot_block")
        challenge_count = sum(1 for e in bot_events if e.event_type == "bot_challenge")

        # Bot score distribution (if scores are logged)
        bot_scores = [e.bot_score for e in bot_events if hasattr(e, 'bot_score') and e.bot_score is not None]
        avg_score = sum(bot_scores) / len(bot_scores) if bot_scores else None

        return {
            "total_bot_events": len(bot_events),
            "blocked": block_count,
            "challenged": challenge_count,
            "avg_bot_score": round(avg_score, 1) if avg_score else None,
            "time_range": range_str,
        }
