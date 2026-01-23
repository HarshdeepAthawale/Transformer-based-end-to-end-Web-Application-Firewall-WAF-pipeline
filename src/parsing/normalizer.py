"""
Request Normalizer Module

Normalizes HTTP requests by removing/replacing dynamic values
"""
import re
import json
from typing import Dict, List, Any
from .log_parser import HTTPRequest
from .normalization_rules import NormalizationRules
from loguru import logger


class RequestNormalizer:
    """Normalize HTTP requests by removing/replacing dynamic values"""
    
    def __init__(self, config_path: str = None):
        self.rules = NormalizationRules()
        if config_path:
            self._load_config(config_path)
    
    def _load_config(self, config_path: str):
        """Load normalization configuration from file"""
        # TODO: Implement config loading if needed
        pass
    
    def normalize(self, request: HTTPRequest) -> HTTPRequest:
        """Normalize HTTP request"""
        # Normalize path
        normalized_path = self._normalize_path(request.path)
        
        # Normalize query parameters
        normalized_query = self._normalize_query_params(request.query_params)
        
        # Normalize headers
        normalized_headers = self._normalize_headers(request.headers)
        
        # Normalize body
        normalized_body = self._normalize_body(request.body) if request.body else None
        
        # Normalize cookies
        normalized_cookies = self._normalize_cookies(request.cookies)
        
        # Create normalized request
        normalized = HTTPRequest(
            method=request.method,
            path=normalized_path,
            query_params=normalized_query,
            headers=normalized_headers,
            body=normalized_body,
            remote_addr='<IP_ADDRESS>',  # Always normalize IPs
            timestamp=None,  # Remove timestamps
            user_agent=self._normalize_user_agent(request.user_agent),
            referer=self._normalize_referer(request.referer),
            cookies=normalized_cookies,
            content_type=request.content_type,
            content_length=request.content_length
        )
        
        return normalized
    
    def _normalize_path(self, path: str) -> str:
        """Normalize URL path"""
        if not path:
            return '/'
        
        # Normalize dynamic segments
        normalized = self.rules.normalize(path)
        
        # Replace numeric path segments
        normalized = re.sub(r'/\d+/', '/<ID>/', normalized)
        normalized = re.sub(r'/\d+$', '/<ID>', normalized)
        
        return normalized
    
    def _normalize_query_params(self, params: Dict) -> Dict:
        """Normalize query parameters"""
        normalized = {}
        for key, value in params.items():
            # Normalize key
            norm_key = self.rules.normalize(key)
            
            # Normalize value
            if isinstance(value, list):
                norm_value = [self.rules.normalize(str(v)) for v in value]
            else:
                norm_value = self.rules.normalize(str(value))
            
            normalized[norm_key] = norm_value
        
        return normalized
    
    def _normalize_headers(self, headers: Dict) -> Dict:
        """Normalize HTTP headers"""
        normalized = {}
        sensitive_headers = ['authorization', 'cookie', 'x-api-key', 'x-auth-token']
        
        for key, value in headers.items():
            key_lower = key.lower()
            
            # Remove or mask sensitive headers
            if key_lower in sensitive_headers:
                normalized[key] = '<SENSITIVE>'
            else:
                normalized[key] = self.rules.normalize(str(value))
        
        return normalized
    
    def _normalize_body(self, body: str) -> str:
        """Normalize request body"""
        if not body:
            return body
        
        # Normalize JSON bodies
        try:
            data = json.loads(body)
            normalized_data = self._normalize_json(data)
            return json.dumps(normalized_data, separators=(',', ':'))
        except:
            # Not JSON, apply general normalization
            return self.rules.normalize(body)
    
    def _normalize_json(self, data: Any) -> Any:
        """Recursively normalize JSON data"""
        if isinstance(data, dict):
            return {k: self._normalize_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._normalize_json(item) for item in data]
        elif isinstance(data, str):
            return self.rules.normalize(data)
        elif isinstance(data, (int, float)):
            # Normalize large numbers (likely IDs)
            if data > 100000:
                return '<NUMERIC_ID>'
            return data
        else:
            return data
    
    def _normalize_cookies(self, cookies: Dict) -> Dict:
        """Normalize cookies"""
        normalized = {}
        for key, value in cookies.items():
            norm_key = self.rules.normalize(key)
            norm_value = self.rules.normalize(value)
            normalized[norm_key] = norm_value
        
        return normalized
    
    def _normalize_user_agent(self, user_agent: str) -> Optional[str]:
        """Normalize user agent (keep structure, remove versions)"""
        if not user_agent:
            return None
        
        # Remove version numbers but keep browser/OS info
        normalized = re.sub(r'/\d+\.\d+', '', user_agent)
        normalized = re.sub(r'\d+\.\d+\.\d+', '<VERSION>', normalized)
        
        return normalized
    
    def _normalize_referer(self, referer: str) -> Optional[str]:
        """Normalize referer URL"""
        if not referer:
            return None
        
        # Normalize domain and path
        normalized = self.rules.normalize(referer)
        # Replace domain with placeholder
        normalized = re.sub(r'https?://[^/]+', '<DOMAIN>', normalized)
        
        return normalized
