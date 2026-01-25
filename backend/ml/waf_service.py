"""
WAF Service Module with ML Inference

Real-time anomaly detection using Transformer-based ML models.
"""
from typing import Dict, Optional
from loguru import logger
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import psutil
import torch

# ML components
try:
    from backend.ml.model.anomaly_detector import AnomalyDetector
    from backend.ml.model.scoring import AnomalyScorer
    from backend.ml.tokenization.tokenizer import HTTPTokenizer
    from backend.ml.tokenization.sequence_prep import SequencePreparator
    from backend.ml.parsing.pipeline import ParsingPipeline
    ML_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML components not available: {e}")
    ML_AVAILABLE = False


class WAFService:
    """WAF Service with ML-based anomaly detection"""
    
    def __init__(
        self,
        model_path: str = None,
        vocab_path: str = None,
        threshold: float = 0.5,
        device: str = None,
        max_batch_size: int = 32,
        timeout: float = 5.0
    ):
        """
        Initialize WAF service
        
        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to vocabulary file
            threshold: Anomaly detection threshold
            device: Device for inference (auto-detect if None)
            max_batch_size: Maximum batch size for inference
            timeout: Request timeout in seconds
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.threshold = threshold
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="waf-service")
        
        # ML components
        self.model = None
        self.scorer = None
        self.tokenizer = None
        self.preparator = None
        self.parser = ParsingPipeline()
        self.ml_enabled = False
        
        # Load ML components if paths provided
        if model_path and vocab_path and ML_AVAILABLE:
            try:
                self._load_ml_components(model_path, vocab_path)
                self.ml_enabled = True
                logger.info("WAF Service initialized with ML components")
            except Exception as e:
                logger.error(f"Failed to load ML components: {e}")
                logger.warning("Falling back to placeholder mode")
        else:
            if not ML_AVAILABLE:
                logger.warning("ML components not available, using placeholder mode")
            else:
                logger.info("WAF Service initialized in placeholder mode (no model/vocab provided)")
        
        # Metrics
        self.metrics = {
            'total_requests': 0,
            'anomalies_detected': 0,
            'processing_times': [],
            'last_health_check': time.time(),
            'uptime': time.time()
        }
        self.metrics_lock = threading.Lock()
    
    def _load_ml_components(self, model_path: str, vocab_path: str):
        """Load ML model and tokenizer"""
        logger.info(f"Loading model from {model_path}...")
        self.model = AnomalyDetector.load_checkpoint(model_path, device=self.device)
        
        logger.info(f"Loading vocabulary from {vocab_path}...")
        self.tokenizer = HTTPTokenizer()
        self.tokenizer.load_vocab(vocab_path)
        
        self.preparator = SequencePreparator(self.tokenizer)
        self.scorer = AnomalyScorer(self.model, threshold=self.threshold, device=self.device)
        
        logger.info("ML components loaded successfully")
    
    def check_request(
        self,
        method: str,
        path: str,
        query_params: Dict = None,
        headers: Dict = None,
        body: Optional[str] = None
    ) -> Dict:
        """Check if request is anomalous using ML model"""
        start_time = time.time()
        
        try:
            if not self.ml_enabled:
                # Placeholder mode: return default values
                result = {
                    'anomaly_score': 0.0,
                    'is_anomaly': False,
                    'threshold': self.threshold,
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'mode': 'placeholder'
                }
            else:
                # ML inference mode
                # Reconstruct request string
                request_text = self._reconstruct_request(
                    method, path, query_params, headers, body
                )
                
                # Parse and normalize
                normalized = self.parser.process_log_line(request_text)
                if not normalized:
                    # Parsing failed - treat as potentially suspicious
                    result = {
                        'anomaly_score': 0.8,
                        'is_anomaly': True,
                        'threshold': self.threshold,
                        'reason': 'parsing_failed',
                        'processing_time_ms': (time.time() - start_time) * 1000
                    }
                else:
                    # Tokenize and prepare sequence
                    token_ids, attention_mask = self.preparator.prepare_sequence(
                        normalized,
                        max_length=512,
                        padding=True,
                        truncation=True
                    )
                    
                    # Convert to tensors
                    input_ids_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
                    attn_mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
                    
                    # Score request
                    score_result = self.scorer.score(input_ids_tensor, attn_mask_tensor)
                    
                    result = {
                        'anomaly_score': score_result['anomaly_score'],
                        'is_anomaly': score_result['is_anomaly'],
                        'threshold': self.threshold,
                        'processing_time_ms': (time.time() - start_time) * 1000,
                        'normalized_request': normalized[:200]  # Preview
                    }
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update metrics
            with self.metrics_lock:
                self.metrics['total_requests'] += 1
                if result.get('is_anomaly', False):
                    self.metrics['anomalies_detected'] += 1
                self.metrics['processing_times'].append(processing_time)
                if len(self.metrics['processing_times']) > 1000:
                    self.metrics['processing_times'] = self.metrics['processing_times'][-1000:]
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking request: {e}", exc_info=True)
            processing_time = (time.time() - start_time) * 1000
            
            # On error, allow request but log (fail open)
            return {
                'anomaly_score': 0.0,
                'is_anomaly': False,
                'threshold': self.threshold,
                'error': str(e),
                'processing_time_ms': processing_time
            }
    
    def _reconstruct_request(
        self,
        method: str,
        path: str,
        query_params: Dict = None,
        headers: Dict = None,
        body: Optional[str] = None
    ) -> str:
        """Reconstruct request string from components"""
        # Build request line
        query_str = ""
        if query_params:
            query_items = []
            for k, v in query_params.items():
                if isinstance(v, list):
                    for item in v:
                        query_items.append(f"{k}={item}")
                else:
                    query_items.append(f"{k}={v}")
            query_str = "?" + "&".join(query_items)
        
        request_line = f'{method} {path}{query_str} HTTP/1.1'
        
        # Add headers
        header_lines = []
        if headers:
            for key, value in headers.items():
                header_lines.append(f"{key}: {value}")
        
        # Add body if present
        body_line = ""
        if body:
            body_preview = body[:200] if len(body) > 200 else body
            body_line = f"BODY:{body_preview}"
        
        # Combine into log-like format
        parts = [request_line]
        if header_lines:
            parts.extend(header_lines[:5])  # Limit headers
        if body_line:
            parts.append(body_line)
        
        return " | ".join(parts)
    
    async def check_request_async(
        self,
        method: str,
        path: str,
        query_params: Dict = None,
        headers: Dict = None,
        body: Optional[str] = None
    ) -> Dict:
        """Async wrapper for request checking"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.check_request,
            method, path, query_params, headers, body
        )
    
    def get_metrics(self) -> Dict:
        """Get current WAF metrics"""
        with self.metrics_lock:
            processing_times = self.metrics['processing_times']
            return {
                'total_requests': self.metrics['total_requests'],
                'anomalies_detected': self.metrics['anomalies_detected'],
                'anomaly_rate': self.metrics['anomalies_detected'] / max(1, self.metrics['total_requests']),
                'avg_processing_time_ms': sum(processing_times) / max(1, len(processing_times)),
                'max_processing_time_ms': max(processing_times) if processing_times else 0,
                'min_processing_time_ms': min(processing_times) if processing_times else 0,
                'uptime_seconds': time.time() - self.metrics['uptime'],
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'cpu_percent': psutil.cpu_percent(interval=0.1)
            }
    
    def update_threshold(self, new_threshold: float):
        """Update anomaly detection threshold"""
        self.threshold = new_threshold
        logger.info(f"WAF threshold updated to {new_threshold}")


# Global WAF service instance
waf_service: Optional[WAFService] = None


def initialize_waf_service(
    model_path: str = None,
    vocab_path: str = None,
    threshold: float = None,
    device: str = None
):
    """Initialize WAF service with ML components"""
    global waf_service
    
    if threshold is None:
        threshold = 0.5  # Default
        logger.info(f"Using default threshold: {threshold}")
    
    waf_service = WAFService(
        model_path=model_path,
        vocab_path=vocab_path,
        threshold=threshold,
        device=device
    )
    
    if waf_service.ml_enabled:
        logger.info("WAF service initialized with ML components")
    else:
        logger.info("WAF service initialized in placeholder mode")


# FastAPI Models (for compatibility)
from pydantic import BaseModel


class CheckRequest(BaseModel):
    method: str
    path: str
    query_params: Optional[Dict] = {}
    headers: Optional[Dict] = {}
    body: Optional[str] = None


class CheckResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    processing_time_ms: float
    normalized_request: Optional[str] = None
    error: Optional[str] = None


class MetricsResponse(BaseModel):
    total_requests: int
    anomalies_detected: int
    anomaly_rate: float
    avg_processing_time_ms: float
    max_processing_time_ms: float
    min_processing_time_ms: float
    uptime_seconds: float
    memory_usage_mb: float
    cpu_percent: float


class UpdateThresholdRequest(BaseModel):
    threshold: float


# FastAPI App (for standalone service)
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    
    app = FastAPI(
        title="WAF Service",
        version="1.0.0",
        description="Web Application Firewall service (ML components removed - placeholder mode)"
    )
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize WAF service on startup"""
        logger.info("WAF Service starting up...")
        initialize_waf_service()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        global waf_service
        if waf_service and hasattr(waf_service, 'executor'):
            waf_service.executor.shutdown(wait=True)
        logger.info("WAF Service shutting down...")
    
    @app.post("/check", response_model=CheckResponse)
    async def check_request(request: CheckRequest):
        """Check if request is anomalous"""
        if waf_service is None:
            raise HTTPException(
                status_code=503,
                detail="WAF service not initialized"
            )
        
        try:
            result = await waf_service.check_request_async(
                method=request.method,
                path=request.path,
                query_params=request.query_params or {},
                headers=request.headers or {},
                body=request.body
            )
            return CheckResponse(**result)
        except Exception as e:
            logger.error(f"API error in /check: {e}")
            raise HTTPException(status_code=500, detail=f"Internal service error: {str(e)}")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        global waf_service
        
        health_status = {
            "status": "healthy" if waf_service else "unhealthy",
            "service": "waf",
            "model_loaded": waf_service.ml_enabled if waf_service else False,
            "mode": "ml" if (waf_service and waf_service.ml_enabled) else "placeholder",
            "timestamp": time.time()
        }
        
        if waf_service:
            health_status.update({
                "device": waf_service.device,
                "threshold": waf_service.threshold
            })
        
        status_code = 200 if waf_service else 503
        return JSONResponse(content=health_status, status_code=status_code)
    
    @app.get("/metrics", response_model=MetricsResponse)
    async def get_metrics():
        """Get WAF service metrics"""
        if waf_service is None:
            raise HTTPException(status_code=503, detail="WAF service not initialized")
        
        try:
            metrics = waf_service.get_metrics()
            return MetricsResponse(**metrics)
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")
    
    @app.post("/update-threshold")
    async def update_threshold(request: UpdateThresholdRequest):
        """Update anomaly detection threshold"""
        if waf_service is None:
            raise HTTPException(status_code=503, detail="WAF service not initialized")
        
        try:
            waf_service.update_threshold(request.threshold)
            return {"status": "success", "new_threshold": request.threshold}
        except Exception as e:
            logger.error(f"Error updating threshold: {e}")
            raise HTTPException(status_code=500, detail=f"Threshold update error: {str(e)}")
    
    @app.get("/config")
    async def get_config():
        """Get current WAF configuration"""
        if waf_service is None:
            raise HTTPException(status_code=503, detail="WAF service not initialized")
        
        return {
            "threshold": waf_service.threshold,
            "device": waf_service.device,
            "max_batch_size": waf_service.max_batch_size,
            "timeout": waf_service.timeout,
            "mode": "placeholder"
        }
    
except ImportError:
    # FastAPI not available, skip app creation
    app = None
    logger.debug("FastAPI not available, skipping app creation")
