"""
Data Leakage Prevention (DLP) Service
"""
from typing import Dict, List, Optional
import re
from loguru import logger


class DLPService:
    """Service for data leakage prevention"""
    
    def __init__(self):
        self.sensitive_patterns = self._load_default_patterns()
    
    def _load_default_patterns(self) -> List[Dict]:
        """Load default sensitive data patterns"""
        return [
            {
                'type': 'credit_card',
                'pattern': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                'name': 'Credit Card Number'
            },
            {
                'type': 'ssn',
                'pattern': r'\b\d{3}-\d{2}-\d{4}\b',
                'name': 'Social Security Number'
            },
            {
                'type': 'email',
                'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'name': 'Email Address'
            },
            {
                'type': 'api_key',
                'pattern': r'\b(api[_-]?key|apikey|access[_-]?token|secret[_-]?key)\s*[:=]\s*[\w\-]{20,}\b',
                'name': 'API Key'
            },
            {
                'type': 'password',
                'pattern': r'\b(password|passwd|pwd)\s*[:=]\s*[\w\-!@#$%^&*()]{8,}\b',
                'name': 'Password'
            },
            {
                'type': 'ip_address',
                'pattern': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
                'name': 'IP Address'
            }
        ]
    
    def inspect_response(self, response_body: str, response_headers: Dict = None) -> Dict:
        """
        Inspect response for sensitive data
        Returns: {
            'has_sensitive_data': bool,
            'detections': List[Dict],
            'action': str  # block, redact, alert
        }
        """
        detections = []
        
        for pattern_info in self.sensitive_patterns:
            matches = re.finditer(pattern_info['pattern'], response_body, re.IGNORECASE)
            for match in matches:
                detections.append({
                    'type': pattern_info['type'],
                    'name': pattern_info['name'],
                    'position': match.start(),
                    'length': len(match.group()),
                    'value': match.group()[:20] + '...' if len(match.group()) > 20 else match.group()
                })
        
        if detections:
            return {
                'has_sensitive_data': True,
                'detections': detections,
                'action': 'alert'  # Default action
            }
        
        return {
            'has_sensitive_data': False,
            'detections': [],
            'action': 'allow'
        }
    
    def redact_sensitive_data(self, text: str, pattern_type: str = None) -> str:
        """Redact sensitive data from text"""
        patterns_to_use = self.sensitive_patterns
        if pattern_type:
            patterns_to_use = [p for p in self.sensitive_patterns if p['type'] == pattern_type]
        
        redacted = text
        for pattern_info in patterns_to_use:
            redacted = re.sub(
                pattern_info['pattern'],
                f"[{pattern_info['name']} REDACTED]",
                redacted,
                flags=re.IGNORECASE
            )
        
        return redacted
    
    def add_custom_pattern(self, pattern_type: str, pattern: str, name: str):
        """Add custom sensitive data pattern"""
        self.sensitive_patterns.append({
            'type': pattern_type,
            'pattern': pattern,
            'name': name
        })
