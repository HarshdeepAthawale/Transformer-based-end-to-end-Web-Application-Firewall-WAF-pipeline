"""
Request Serializer Module

Serializes HTTPRequest to string format for tokenization
"""
from typing import Dict
from .log_parser import HTTPRequest


class RequestSerializer:
    """Serialize HTTPRequest to string format for tokenization"""
    
    @staticmethod
    def to_string(request: HTTPRequest, format: str = "compact") -> str:
        """Convert request to string representation"""
        if format == "compact":
            return RequestSerializer._to_compact_string(request)
        elif format == "detailed":
            return RequestSerializer._to_detailed_string(request)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @staticmethod
    def _to_compact_string(request: HTTPRequest) -> str:
        """Compact string representation"""
        parts = []
        
        # Method and path
        parts.append(f"{request.method} {request.path}")
        
        # Query parameters
        if request.query_params:
            query_items = []
            for k, v in request.query_params.items():
                if isinstance(v, list):
                    for item in v:
                        query_items.append(f"{k}={item}")
                else:
                    query_items.append(f"{k}={v}")
            query_str = "&".join(query_items)
            parts.append(f"?{query_str}")
        
        # Headers (selected)
        important_headers = ['content-type', 'user-agent', 'accept']
        header_parts = []
        for h in important_headers:
            if h in request.headers:
                header_parts.append(f"{h}:{request.headers[h]}")
        if header_parts:
            parts.append(" ".join(header_parts))
        
        # Body (truncated)
        if request.body:
            body_preview = request.body[:200] if len(request.body) > 200 else request.body
            parts.append(f"BODY:{body_preview}")
        
        return " | ".join(parts)
    
    @staticmethod
    def _to_detailed_string(request: HTTPRequest) -> str:
        """Detailed string representation"""
        lines = []
        
        # Request line
        query_str = ""
        if request.query_params:
            query_items = []
            for k, v in request.query_params.items():
                if isinstance(v, list):
                    for item in v:
                        query_items.append(f"{k}={item}")
                else:
                    query_items.append(f"{k}={v}")
            query_str = "?" + "&".join(query_items)
        lines.append(f"{request.method} {request.path}{query_str}")
        
        # Headers
        for key, value in request.headers.items():
            lines.append(f"{key}: {value}")
        
        # Body
        if request.body:
            lines.append("")
            lines.append(request.body)
        
        return "\n".join(lines)
