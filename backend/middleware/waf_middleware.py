"""
WAF Middleware

Intercepts all incoming HTTP requests and checks them with the WAF service.
Blocks requests that are detected as anomalies.
"""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Optional
import time
import json
from loguru import logger
import threading

from backend.config import config


class WAFMiddleware(BaseHTTPMiddleware):
    """Middleware for WAF request interception and blocking"""
    
    _instance: Optional["WAFMiddleware"] = None  # Set in __init__ for metrics endpoint

    def __init__(self, app, waf_service=None):
        super().__init__(app)
        WAFMiddleware._instance = self
        self.waf_service = waf_service
        self.metrics = {
            'total_requests': 0,
            'anomalies_detected': 0,
            'requests_blocked': 0,
            'waf_errors': 0,
            'processing_times': [],
        }
        self.metrics_lock = threading.Lock()
        
        # Paths to skip WAF checking
        self.skip_paths = {
            '/health',
            '/docs',
            '/openapi.json',
            '/api/debug/routes',
            '/api/waf/check',  # Don't check WAF check endpoint itself
            '/api/waf/check/batch',  # Batch check endpoint
            '/api/waf/stats',
            '/api/waf/config',
            '/api/waf/model-info',
            '/api/waf/middleware-metrics',
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request through WAF"""
        # Skip WAF for certain paths
        if request.url.path in self.skip_paths:
            return await call_next(request)
        
        # Skip if WAF is disabled
        if not config.WAF_ENABLED:
            return await call_next(request)
        
        # Get WAF service from app state (set at startup) or instance
        waf_svc = self.waf_service
        if waf_svc is None and hasattr(request.app, "state"):
            waf_svc = getattr(request.app.state, "waf_service", None)
        if waf_svc is None:
            try:
                from backend.core.waf_factory import get_waf_service
                waf_svc = get_waf_service()
            except Exception as e:
                logger.debug(f"Could not get WAF service: {e}")
        self.waf_service = waf_svc

        # Skip if WAF service is not available
        if self.waf_service is None:
            if config.WAF_FAIL_OPEN:
                logger.warning("WAF service not available, allowing request (fail-open)")
                return await call_next(request)
            else:
                logger.error("WAF service not available, blocking request (fail-closed)")
                return JSONResponse(
                    status_code=503,
                    content={
                        "success": False,
                        "message": "WAF service unavailable",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                    }
                )
        
        start_time = time.time()
        
        try:
            # Extract request details
            method = request.method
            path = str(request.url.path)
            query_params = dict(request.query_params)
            headers = dict(request.headers)
            
            # Get request body if present (preserve for downstream)
            body = None
            body_bytes = None
            if request.method in ['POST', 'PUT', 'PATCH']:
                try:
                    body_bytes = await request.body()
                    if body_bytes:
                        # Try to decode as JSON, fallback to string
                        try:
                            body = json.loads(body_bytes.decode('utf-8'))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            body = body_bytes.decode('utf-8', errors='ignore')
                except Exception as e:
                    logger.debug(f"Error reading request body: {e}")
            
            # Store body bytes for recreating request
            request._body = body_bytes
            
            # Check request with WAF
            try:
                result = await self.waf_service.check_request_async(
                    method=method,
                    path=path,
                    query_params=query_params,
                    headers=headers,
                    body=body
                )
                
                processing_time = (time.time() - start_time) * 1000  # ms
                
                # Update metrics
                with self.metrics_lock:
                    self.metrics['total_requests'] += 1
                    self.metrics['processing_times'].append(processing_time)
                    if len(self.metrics['processing_times']) > 1000:
                        self.metrics['processing_times'] = self.metrics['processing_times'][-1000:]
                
                # Check if anomaly detected
                if result.get('is_anomaly', False):
                    anomaly_score = result.get('anomaly_score', 0.0)
                    
                    # Update metrics
                    with self.metrics_lock:
                        self.metrics['anomalies_detected'] += 1
                        self.metrics['requests_blocked'] += 1
                    
                    # Log the blocked request
                    logger.warning(
                        f"WAF BLOCKED: {method} {path} | "
                        f"Score: {anomaly_score:.4f} | "
                        f"Threshold: {result.get('threshold', 0.5):.4f} | "
                        f"IP: {request.client.host if request.client else 'unknown'}"
                    )
                    
                    # Block the request
                    blocked_response = JSONResponse(
                        status_code=403,
                        content={
                            "success": False,
                            "message": "Request blocked by WAF",
                            "reason": "Anomaly detected",
                            "anomaly_score": anomaly_score,
                            "threshold": result.get('threshold', 0.5),
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                        }
                    )
                    
                    # Store traffic log in background (don't block request)
                    self._store_traffic_log_async(
                        request=request,
                        method=method,
                        path=path,
                        query_params=query_params,
                        headers=headers,
                        body=body,
                        response=blocked_response,
                        result=result,
                        processing_time=processing_time
                    )
                    
                    return blocked_response
                
                # Request is normal, allow it to proceed
                # Recreate request with body since it was consumed
                if hasattr(request, '_body') and request._body:
                    # Recreate the receive function to restore body
                    async def receive():
                        return {'type': 'http.request', 'body': request._body}
                    request._receive = receive
                
                response = await call_next(request)
                
                # Store traffic log in background (don't block request)
                self._store_traffic_log_async(
                    request=request,
                    method=method,
                    path=path,
                    query_params=query_params,
                    headers=headers,
                    body=body,
                    response=response,
                    result=result,
                    processing_time=processing_time
                )
                
                return response
                
            except Exception as e:
                # WAF check failed
                processing_time = (time.time() - start_time) * 1000
                
                with self.metrics_lock:
                    self.metrics['waf_errors'] += 1
                
                logger.error(f"WAF check failed: {e}", exc_info=True)
                
                # Handle based on fail-open/fail-closed setting
                if config.WAF_FAIL_OPEN:
                    logger.warning("WAF check failed, allowing request (fail-open)")
                    # Recreate request with body if needed
                    if hasattr(request, '_body') and request._body:
                        async def receive():
                            return {'type': 'http.request', 'body': request._body}
                        request._receive = receive
                    return await call_next(request)
                else:
                    logger.error("WAF check failed, blocking request (fail-closed)")
                    return JSONResponse(
                        status_code=503,
                        content={
                            "success": False,
                            "message": "WAF check failed",
                            "error": str(e),
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                        }
                    )
        
        except Exception as e:
            # Unexpected error in middleware
            logger.error(f"WAF middleware error: {e}", exc_info=True)
            
            if config.WAF_FAIL_OPEN:
                return await call_next(request)
            else:
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "message": "Internal WAF error",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                    }
                )
    
    def _store_traffic_log_async(
        self,
        request: Request,
        method: str,
        path: str,
        query_params: dict,
        headers: dict,
        body,
        response,
        result: dict,
        processing_time: float
    ):
        """Store traffic log in background thread"""
        try:
            from backend.database import SessionLocal
            from backend.services.traffic_service import TrafficService
            from backend.services.websocket_service import broadcast_update_sync
            
            # Get request metadata
            client_ip = request.client.host if request.client else "unknown"
            user_agent = headers.get("user-agent", "")
            
            # Get response size
            response_size = 0
            try:
                # Try to get size from response body
                if hasattr(response, 'body'):
                    if isinstance(response.body, bytes):
                        response_size = len(response.body)
                    elif isinstance(response.body, str):
                        response_size = len(response.body.encode('utf-8'))
                # For JSONResponse, try to get from content
                elif hasattr(response, 'body_iterator'):
                    # Response body is an iterator, can't easily get size
                    # Estimate based on status code (will be updated if needed)
                    response_size = 0
                # Try to estimate from content if available
                elif hasattr(response, 'content'):
                    if isinstance(response.content, bytes):
                        response_size = len(response.content)
                    elif isinstance(response.content, str):
                        response_size = len(response.content.encode('utf-8'))
            except Exception:
                # If we can't determine size, default to 0
                response_size = 0
            
            # Store in background thread to avoid blocking
            def store_traffic():
                db = SessionLocal()
                try:
                    traffic_service = TrafficService(db)
                    traffic_log = traffic_service.create_traffic_log(
                        ip=client_ip,
                        method=method,
                        endpoint=path,
                        status_code=response.status_code,
                        response_size=response_size,
                        user_agent=user_agent,
                        query_string=str(query_params) if query_params else None,
                        request_body=str(body)[:1000] if body else None,
                        processing_time_ms=int(processing_time),
                        was_blocked=result.get('is_anomaly', False),
                        anomaly_score=result.get('anomaly_score'),
                        threat_type=None  # Can add classification later
                    )
                    # Broadcast update
                    broadcast_update_sync("traffic", traffic_log.to_dict())
                except Exception as e:
                    logger.error(f"Failed to store traffic log: {e}")
                finally:
                    db.close()
            
            # Run in background
            threading.Thread(target=store_traffic, daemon=True).start()
        except Exception as e:
            logger.debug(f"Traffic logging not available: {e}")
    
    def get_metrics(self) -> Dict:
        """Get WAF middleware metrics"""
        with self.metrics_lock:
            processing_times = self.metrics['processing_times']
            return {
                'total_requests': self.metrics['total_requests'],
                'anomalies_detected': self.metrics['anomalies_detected'],
                'requests_blocked': self.metrics['requests_blocked'],
                'waf_errors': self.metrics['waf_errors'],
                'avg_processing_time_ms': sum(processing_times) / max(1, len(processing_times)),
                'max_processing_time_ms': max(processing_times) if processing_times else 0,
                'min_processing_time_ms': min(processing_times) if processing_times else 0,
                'block_rate': self.metrics['requests_blocked'] / max(1, self.metrics['total_requests']),
            }
