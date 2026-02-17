"""
Bot Detection Service - bot score (1-99) and verified bots integration.
"""
from sqlalchemy.orm import Session
from typing import Dict, List

import re
from datetime import datetime

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
        self._signature_cache = None
        self._load_signatures()
    
    def _load_signatures(self):
        """Load bot signatures into cache"""
        self._signature_cache = self.db.query(BotSignature)\
            .filter(BotSignature.is_active)\
            .all()
        logger.info(f"Loaded {len(self._signature_cache)} bot signatures")
    
    def detect_bot(self, user_agent: str, ip: str, headers: Dict) -> Dict:
        """
        Detect if request is from a bot. Returns bot_score (1-99), action from bands.
        Cloudflare-style: low score = automated, high score = human.
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

        # Check against signatures
        for signature in self._signature_cache:
            try:
                if re.search(signature.user_agent_pattern, user_agent, re.IGNORECASE):
                    signature.detection_count += 1
                    signature.last_detected = datetime.utcnow()
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
        user_agent_pattern: str,
        name: str,
        category: BotCategory,
        action: str = "block",
        is_whitelisted: bool = False
    ) -> BotSignature:
        """Add bot signature"""
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
        
        # Reload cache
        self._load_signatures()
        
        return signature
    
    def get_signatures(self, active_only: bool = True) -> List[BotSignature]:
        """Get bot signatures"""
        query = self.db.query(BotSignature)
        if active_only:
            query = query.filter(BotSignature.is_active)
        return query.order_by(BotSignature.timestamp.desc()).all()
