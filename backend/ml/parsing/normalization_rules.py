"""
Normalization Rules Module

Defines rules for normalizing dynamic values in HTTP requests
"""
import re
from typing import Dict, List, Callable, Any
from datetime import datetime
from loguru import logger


class NormalizationRule:
    """Base class for normalization rules"""
    
    def __init__(self, name: str, pattern: str, replacement: str, flags: int = 0):
        self.name = name
        self.pattern = re.compile(pattern, flags)
        self.replacement = replacement
    
    def apply(self, text: str) -> str:
        """Apply normalization rule"""
        return self.pattern.sub(self.replacement, text)


class NormalizationRules:
    """Collection of normalization rules"""
    
    def __init__(self):
        self.rules: List[NormalizationRule] = []
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default normalization rules"""
        # UUIDs (8-4-4-4-12 format)
        self.add_rule(
            "uuid",
            r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',
            '<UUID>'
        )
        
        # Timestamps (ISO format)
        self.add_rule(
            "iso_timestamp",
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?',
            '<TIMESTAMP>'
        )
        
        # Unix timestamps (10 digits)
        self.add_rule(
            "unix_timestamp",
            r'\b\d{10}\b',
            '<TIMESTAMP>'
        )
        
        # Session IDs (common patterns)
        self.add_rule(
            "session_id",
            r'(session|sess|sid|jsessionid|phpsessid)=[a-zA-Z0-9]{20,}',
            r'\1=<SESSION_ID>'
        )
        
        # Numeric IDs (long numbers likely to be IDs)
        self.add_rule(
            "numeric_id",
            r'\b\d{6,}\b',
            '<NUMERIC_ID>'
        )
        
        # Email addresses
        self.add_rule(
            "email",
            r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            '<EMAIL>'
        )
        
        # IP addresses (but keep structure)
        self.add_rule(
            "ip_address",
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            '<IP_ADDRESS>'
        )
        
        # Credit card numbers (basic pattern)
        self.add_rule(
            "credit_card",
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            '<CARD_NUMBER>'
        )
        
        # Base64 encoded strings (long base64)
        self.add_rule(
            "base64",
            r'[A-Za-z0-9+/]{50,}={0,2}',
            '<BASE64>'
        )
        
        # JWT tokens
        self.add_rule(
            "jwt",
            r'eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]*',
            '<JWT_TOKEN>'
        )
        
        # API keys (common patterns)
        self.add_rule(
            "api_key",
            r'(api[_-]?key|apikey|access[_-]?token|auth[_-]?token)=[a-zA-Z0-9]{20,}',
            r'\1=<API_KEY>'
        )
    
    def add_rule(self, name: str, pattern: str, replacement: str, flags: int = 0):
        """Add a normalization rule"""
        rule = NormalizationRule(name, pattern, replacement, flags)
        self.rules.append(rule)
        logger.debug(f"Added normalization rule: {name}")
    
    def normalize(self, text: str) -> str:
        """Apply all normalization rules"""
        if not text:
            return text
        
        normalized = text
        for rule in self.rules:
            normalized = rule.apply(normalized)
        
        return normalized
