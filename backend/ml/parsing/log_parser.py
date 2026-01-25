"""
Log Parser Module

Parses various log formats (Apache, Nginx) into structured HTTPRequest objects
"""
import re
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from urllib.parse import urlparse, parse_qs, unquote
from loguru import logger


@dataclass
class HTTPRequest:
    """Structured representation of HTTP request"""
    method: str
    path: str
    query_params: Dict[str, List[str]] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[str] = None
    remote_addr: Optional[str] = None
    timestamp: Optional[str] = None
    user_agent: Optional[str] = None
    referer: Optional[str] = None
    cookies: Dict[str, str] = field(default_factory=dict)
    content_type: Optional[str] = None
    content_length: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'method': self.method,
            'path': self.path,
            'query_params': self.query_params,
            'headers': self.headers,
            'body': self.body,
            'remote_addr': self.remote_addr,
            'timestamp': self.timestamp,
            'user_agent': self.user_agent,
            'referer': self.referer,
            'cookies': self.cookies,
            'content_type': self.content_type,
            'content_length': self.content_length
        }


class LogParser:
    """Parse various log formats into structured HTTPRequest objects"""
    
    # Apache Combined Log Format
    APACHE_COMBINED_PATTERN = re.compile(
        r'^(\S+) (\S+) (\S+) \[([^\]]+)\] "(\S+) ([^"]+)" (\d+) (\d+|-) '
        r'"([^"]*)" "([^"]*)"'
    )
    
    # Apache Detailed Format (with additional fields)
    APACHE_DETAILED_PATTERN = re.compile(
        r'^(\S+) (\S+) (\S+) \[([^\]]+)\] "(\S+) ([^"]+)" (\d+) (\d+|-) '
        r'"([^"]*)" "([^"]*)" "([^"]*)" "([^"]*)" "([^"]*)" "([^"]*)"'
    )
    
    # Nginx Combined Format
    NGINX_COMBINED_PATTERN = re.compile(
        r'^(\S+) - (\S+) \[([^\]]+)\] "(\S+) ([^"]+)" (\d+) (\d+|-) '
        r'"([^"]*)" "([^"]*)"'
    )
    
    # Nginx Combined Format (alternative - request line as single group)
    NGINX_COMBINED_PATTERN_ALT = re.compile(
        r'^(\S+) - (\S+) \[([^\]]+)\] "([^"]+)" (\d+) (\d+|-) '
        r'"([^"]*)" "([^"]*)"'
    )
    
    # Nginx Detailed Format (with additional fields)
    NGINX_DETAILED_PATTERN = re.compile(
        r'^(\S+) - (\S+) \[([^\]]+)\] "(\S+) ([^"]+)" (\d+) (\d+|-) '
        r'"([^"]*)" "([^"]*)" "([^"]*)" "([^"]*)" "([^"]*)"'
    )
    
    def parse(self, log_line: str) -> Optional[HTTPRequest]:
        """Parse log line into HTTPRequest object"""
        if not log_line or not log_line.strip():
            return None
        
        # Try Nginx Detailed first (most specific)
        match = self.NGINX_DETAILED_PATTERN.match(log_line)
        if match:
            return self._parse_nginx_detailed(match, log_line)
        
        # Try Nginx Combined
        match = self.NGINX_COMBINED_PATTERN.match(log_line)
        if match:
            return self._parse_nginx_combined(match, log_line)
        
        # Try Apache Detailed
        match = self.APACHE_DETAILED_PATTERN.match(log_line)
        if match:
            return self._parse_apache_detailed(match, log_line)
        
        # Try Apache Combined
        match = self.APACHE_COMBINED_PATTERN.match(log_line)
        if match:
            return self._parse_apache_combined(match, log_line)
        
        logger.warning(f"Could not parse log line: {log_line[:100]}")
        return None
    
    def _parse_nginx_combined(self, match, log_line: str) -> HTTPRequest:
        """Parse Nginx Combined format"""
        groups = match.groups()
        
        remote_addr = groups[0]
        remote_user = groups[1] if groups[1] != '-' else None
        timestamp = groups[2]
        
        # Nginx format: "METHOD PATH HTTP/VERSION" is in groups[4]
        # groups[3] is METHOD, groups[4] is "PATH HTTP/VERSION"
        # Actually, the regex captures: groups[3] = METHOD, groups[4] = "PATH HTTP/VERSION"
        # So we need to combine them or parse groups[4] correctly
        method_str = groups[3]  # This is the HTTP method
        request_path_part = groups[4]  # This is "PATH HTTP/VERSION" or full request line
        
        # Parse the request line - it might be just the path+version or full "METHOD PATH HTTP/VERSION"
        # Check if request_path_part starts with HTTP method
        if request_path_part.startswith(('GET ', 'POST ', 'PUT ', 'DELETE ', 'PATCH ', 'HEAD ', 'OPTIONS ')):
            # Full request line
            method, path, query_params = self._parse_request_line(request_path_part)
        else:
            # Just path part, method is separate
            method = method_str
            # Parse just the path part (remove HTTP/VERSION if present)
            path_part = request_path_part.split(' HTTP/')[0] if ' HTTP/' in request_path_part else request_path_part
            parsed = urlparse(path_part)
            path = unquote(parsed.path) if parsed.path else '/'
            query_params = parse_qs(parsed.query, keep_blank_values=True)
            query_params = {k: v[0] if len(v) == 1 else v for k, v in query_params.items()}
        
        status = int(groups[5])
        bytes_sent = int(groups[6]) if groups[6] != '-' else 0
        referer = groups[7] if groups[7] != '-' else None
        user_agent = groups[8] if groups[8] != '-' else None
        
        return HTTPRequest(
            method=method,
            path=path,
            query_params=query_params,
            headers={},
            body=None,
            remote_addr=remote_addr,
            timestamp=timestamp,
            user_agent=user_agent,
            referer=referer
        )
    
    def _parse_nginx_detailed(self, match, log_line: str) -> HTTPRequest:
        """Parse Nginx Detailed format"""
        groups = match.groups()
        
        remote_addr = groups[0]
        remote_user = groups[1] if groups[1] != '-' else None
        timestamp = groups[2]
        
        # Nginx format: groups[3] is METHOD, groups[4] is "PATH HTTP/VERSION"
        method_str = groups[3]
        request_path_part = groups[4]
        
        # Parse the request line
        if request_path_part.startswith(('GET ', 'POST ', 'PUT ', 'DELETE ', 'PATCH ', 'HEAD ', 'OPTIONS ')):
            # Full request line
            method, path, query_params = self._parse_request_line(request_path_part)
        else:
            # Just path part, method is separate
            method = method_str
            # Parse just the path part (remove HTTP/VERSION if present)
            path_part = request_path_part.split(' HTTP/')[0] if ' HTTP/' in request_path_part else request_path_part
            parsed = urlparse(path_part)
            path = unquote(parsed.path) if parsed.path else '/'
            query_params = parse_qs(parsed.query, keep_blank_values=True)
            query_params = {k: v[0] if len(v) == 1 else v for k, v in query_params.items()}
        
        status = int(groups[5])
        bytes_sent = int(groups[6]) if groups[6] != '-' else 0
        referer = groups[7] if groups[7] != '-' else None
        user_agent = groups[8] if groups[8] != '-' else None
        
        # Extract additional fields if present
        cookies = {}
        content_type = None
        content_length = None
        body = None
        
        if len(groups) > 9:
            x_forwarded_for = groups[9] if len(groups) > 9 and groups[9] != '-' else None
            cookie_str = groups[10] if len(groups) > 10 and groups[10] != '-' else None
            if cookie_str:
                cookies = self._parse_cookies(cookie_str)
            content_type = groups[11] if len(groups) > 11 and groups[11] != '-' else None
            content_length = int(groups[12]) if len(groups) > 12 and groups[12] != '-' else None
            body = groups[13] if len(groups) > 13 and groups[13] != '-' else None
        
        return HTTPRequest(
            method=method,
            path=path,
            query_params=query_params,
            headers={},
            body=body,
            remote_addr=remote_addr,
            timestamp=timestamp,
            user_agent=user_agent,
            referer=referer,
            cookies=cookies,
            content_type=content_type,
            content_length=content_length
        )
    
    def _parse_apache_combined(self, match, log_line: str) -> HTTPRequest:
        """Parse Apache Combined format"""
        groups = match.groups()
        
        remote_addr = groups[0]
        remote_user = groups[1] if groups[1] != '-' else None
        timestamp = groups[3]
        
        # Apache format: groups[4] is METHOD, groups[5] is "PATH HTTP/VERSION"
        method_str = groups[4]
        request_path_part = groups[5]
        
        # Parse the request line
        if request_path_part.startswith(('GET ', 'POST ', 'PUT ', 'DELETE ', 'PATCH ', 'HEAD ', 'OPTIONS ')):
            # Full request line
            method, path, query_params = self._parse_request_line(request_path_part)
        else:
            # Just path part, method is separate
            method = method_str
            # Parse just the path part (remove HTTP/VERSION if present)
            path_part = request_path_part.split(' HTTP/')[0] if ' HTTP/' in request_path_part else request_path_part
            parsed = urlparse(path_part)
            path = unquote(parsed.path) if parsed.path else '/'
            query_params = parse_qs(parsed.query, keep_blank_values=True)
            query_params = {k: v[0] if len(v) == 1 else v for k, v in query_params.items()}
        
        status = int(groups[6])
        bytes_sent = int(groups[7]) if groups[7] != '-' else 0
        referer = groups[8] if groups[8] != '-' else None
        user_agent = groups[9] if groups[9] != '-' else None
        
        return HTTPRequest(
            method=method,
            path=path,
            query_params=query_params,
            headers={},
            body=None,
            remote_addr=remote_addr,
            timestamp=timestamp,
            user_agent=user_agent,
            referer=referer
        )
    
    def _parse_apache_detailed(self, match, log_line: str) -> HTTPRequest:
        """Parse Apache Detailed format (with POST data)"""
        groups = match.groups()
        
        remote_addr = groups[0]
        remote_user = groups[1] if groups[1] != '-' else None
        timestamp = groups[3]
        
        # Apache format: groups[4] is METHOD, groups[5] is "PATH HTTP/VERSION"
        method_str = groups[4]
        request_path_part = groups[5]
        
        # Parse the request line
        if request_path_part.startswith(('GET ', 'POST ', 'PUT ', 'DELETE ', 'PATCH ', 'HEAD ', 'OPTIONS ')):
            # Full request line
            method, path, query_params = self._parse_request_line(request_path_part)
        else:
            # Just path part, method is separate
            method = method_str
            # Parse just the path part (remove HTTP/VERSION if present)
            path_part = request_path_part.split(' HTTP/')[0] if ' HTTP/' in request_path_part else request_path_part
            parsed = urlparse(path_part)
            path = unquote(parsed.path) if parsed.path else '/'
            query_params = parse_qs(parsed.query, keep_blank_values=True)
            query_params = {k: v[0] if len(v) == 1 else v for k, v in query_params.items()}
        
        status = int(groups[6])
        bytes_sent = int(groups[7]) if groups[7] != '-' else 0
        referer = groups[8] if groups[8] != '-' else None
        user_agent = groups[9] if groups[9] != '-' else None
        
        # Extract additional fields if present
        cookies = {}
        content_type = None
        content_length = None
        body = None
        
        if len(groups) > 10:
            x_forwarded_for = groups[10] if len(groups) > 10 and groups[10] != '-' else None
            cookie_str = groups[11] if len(groups) > 11 and groups[11] != '-' else None
            if cookie_str:
                cookies = self._parse_cookies(cookie_str)
            content_type = groups[12] if len(groups) > 12 and groups[12] != '-' else None
            content_length = int(groups[13]) if len(groups) > 13 and groups[13] != '-' else None
            body = groups[14] if len(groups) > 14 and groups[14] != '-' else None
        
        return HTTPRequest(
            method=method,
            path=path,
            query_params=query_params,
            headers={},
            body=body,
            remote_addr=remote_addr,
            timestamp=timestamp,
            user_agent=user_agent,
            referer=referer,
            cookies=cookies,
            content_type=content_type,
            content_length=content_length
        )
    
    def _parse_request_line(self, request_line: str) -> tuple:
        """Parse HTTP request line: METHOD PATH?QUERY HTTP/VERSION"""
        # Request line format: "METHOD PATH?QUERY HTTP/VERSION"
        # Example: "GET /api/data?foo=bar HTTP/1.1"
        parts = request_line.split(' ', 2)
        if len(parts) < 2:
            return 'GET', '/', {}
        
        method = parts[0]
        url_part = parts[1] if len(parts) > 1 else '/'
        
        # Parse URL
        parsed = urlparse(url_part)
        path = unquote(parsed.path) if parsed.path else '/'
        query_params = parse_qs(parsed.query, keep_blank_values=True)
        
        # Convert lists to single values where appropriate
        query_params = {k: v[0] if len(v) == 1 else v for k, v in query_params.items()}
        
        return method, path, query_params
    
    def _parse_cookies(self, cookie_str: str) -> Dict[str, str]:
        """Parse cookie string into dictionary"""
        cookies = {}
        for item in cookie_str.split(';'):
            item = item.strip()
            if '=' in item:
                key, value = item.split('=', 1)
                cookies[key.strip()] = value.strip()
        return cookies
