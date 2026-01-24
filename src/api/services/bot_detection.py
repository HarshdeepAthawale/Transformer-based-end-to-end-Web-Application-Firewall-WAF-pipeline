"""
Bot Detection Service
"""
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
import re
from datetime import datetime
from loguru import logger

from src.api.models.bot_signatures import BotSignature, BotCategory


class BotDetectionService:
    """Service for bot detection"""
    
    def __init__(self, db: Session):
        self.db = db
        self._signature_cache = None
        self._load_signatures()
    
    def _load_signatures(self):
        """Load bot signatures into cache"""
        self._signature_cache = self.db.query(BotSignature)\
            .filter(BotSignature.is_active == True)\
            .all()
        logger.info(f"Loaded {len(self._signature_cache)} bot signatures")
    
    def detect_bot(
        self,
        user_agent: str,
        ip: str,
        headers: Dict
    ) -> Dict:
        """
        Detect if request is from a bot
        Returns: {
            'is_bot': bool,
            'bot_name': str,
            'category': str,
            'is_whitelisted': bool,
            'action': str,  # block, allow, challenge, monitor
            'confidence': float
        }
        """
        if not user_agent:
            # Missing User-Agent is suspicious
            return {
                'is_bot': True,
                'bot_name': 'Missing User-Agent',
                'category': 'unknown',
                'is_whitelisted': False,
                'action': 'challenge',
                'confidence': 0.7
            }
        
        # Check against signatures
        for signature in self._signature_cache:
            try:
                if re.search(signature.user_agent_pattern, user_agent, re.IGNORECASE):
                    # Update statistics
                    signature.detection_count += 1
                    signature.last_detected = datetime.utcnow()
                    self.db.commit()
                    
                    return {
                        'is_bot': True,
                        'bot_name': signature.name,
                        'category': signature.category.value if signature.category else 'unknown',
                        'is_whitelisted': signature.is_whitelisted,
                        'action': signature.action if not signature.is_whitelisted else 'allow',
                        'confidence': 0.9
                    }
            except re.error:
                logger.warning(f"Invalid regex pattern in signature {signature.id}: {signature.user_agent_pattern}")
                continue
        
        # Behavioral checks
        behavioral_result = self._check_behavioral_patterns(user_agent, headers)
        if behavioral_result['is_bot']:
            return behavioral_result
        
        return {
            'is_bot': False,
            'bot_name': None,
            'category': None,
            'is_whitelisted': False,
            'action': 'allow',
            'confidence': 0.0
        }
    
    def _check_behavioral_patterns(self, user_agent: str, headers: Dict) -> Dict:
        """Check behavioral patterns that indicate bots"""
        suspicious_indicators = 0
        
        # Missing common headers
        if 'Accept-Language' not in headers:
            suspicious_indicators += 1
        if 'Accept-Encoding' not in headers:
            suspicious_indicators += 1
        
        # Suspicious User-Agent patterns
        suspicious_patterns = [
            r'bot', r'crawler', r'spider', r'scraper',
            r'curl', r'wget', r'python', r'java',
            r'^$'  # Empty user agent
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, user_agent, re.IGNORECASE):
                suspicious_indicators += 1
                break
        
        if suspicious_indicators >= 2:
            return {
                'is_bot': True,
                'bot_name': 'Suspicious Behavior',
                'category': 'unknown',
                'is_whitelisted': False,
                'action': 'challenge',
                'confidence': 0.6
            }
        
        return {
            'is_bot': False,
            'bot_name': None,
            'category': None,
            'is_whitelisted': False,
            'action': 'allow',
            'confidence': 0.0
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
            query = query.filter(BotSignature.is_active == True)
        return query.order_by(BotSignature.timestamp.desc()).all()
