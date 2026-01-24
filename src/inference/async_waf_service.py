"""
Async WAF Service Module

Non-blocking, concurrent request processing with async/await and thread pool execution
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
from loguru import logger
import time
from queue import Queue
from threading import Lock

from src.model.anomaly_detector import AnomalyDetector
from src.model.scoring import AnomalyScorer
from src.tokenization.tokenizer import HTTPTokenizer
from src.tokenization.sequence_prep import SequencePreparator
from src.parsing.pipeline import ParsingPipeline
from src.inference.optimization import optimize_model
from src.inference.rate_limiter import RateLimiter, PerIPRateLimiter
from src.inference.queue_manager import RequestQueueManager


class AsyncWAFService:
    """Async WAF service with non-blocking inference"""
    
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        threshold: float = 0.5,
        device: str = "cpu",
        max_workers: int = 4,
        batch_size: int = 32,
        timeout: float = 5.0,
        optimization: Optional[str] = None,
        use_queue_manager: bool = False,
        queue_max_size: int = 1000,
        queue_batch_timeout: float = 0.1,
        anomaly_log_file: Optional[str] = None
    ):
        self.device = device
        self.threshold = threshold
        self.timeout = timeout
        self.batch_size = batch_size
        self.anomaly_log_file = anomaly_log_file
        
        # Load model and tokenizer
        self._load_model(model_path, vocab_path, optimization)
        
        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="waf-async")
        
        # Request queue for batching (legacy, kept for compatibility)
        self.request_queue: Queue = Queue()
        self.batch_lock = Lock()
        self.processing = False
        
        # Optional queue manager for advanced queuing
        self.queue_manager: Optional[RequestQueueManager] = None
        if use_queue_manager:
            self.queue_manager = RequestQueueManager(
                max_size=queue_max_size,
                batch_timeout=queue_batch_timeout,
                batch_size=batch_size
            )
            # Start queue processing
            self.queue_manager.start_processing(self._queue_processor)
        
        # Metrics
        self.total_requests = 0
        self.anomalies_detected = 0
        self.avg_processing_time = 0.0
        self.metrics_lock = Lock()
        
        logger.info(f"Async WAF Service initialized: device={device}, max_workers={max_workers}, batch_size={batch_size}, timeout={timeout}, optimization={optimization}, queue_manager={use_queue_manager}")
    
    def _load_model(self, model_path: str, vocab_path: str, optimization: Optional[str] = None):
        """Load model and tokenizer with optional optimization"""
        logger.info(f"Loading tokenizer from {vocab_path}")
        self.tokenizer = HTTPTokenizer()
        self.tokenizer.load_vocab(vocab_path)
        
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        vocab_size = checkpoint.get('vocab_size', len(self.tokenizer.word_to_id))
        
        # Infer architecture from checkpoint
        state_dict = checkpoint['model_state_dict']
        
        # Get hidden size from word embeddings
        if 'transformer.embeddings.word_embeddings.weight' in state_dict:
            hidden_size = state_dict['transformer.embeddings.word_embeddings.weight'].shape[1]
        else:
            hidden_size = 768  # fallback
        
        # Get max length from position embeddings
        if 'transformer.embeddings.position_embeddings.weight' in state_dict:
            max_length = state_dict['transformer.embeddings.position_embeddings.weight'].shape[0]
        else:
            max_length = 512  # fallback
        
        # Infer number of layers
        num_layers = 0
        while f'transformer.transformer.layer.{num_layers}.attention.q_lin.weight' in state_dict:
            num_layers += 1
        
        # Infer number of heads
        possible_heads = [8, 12, 16]
        num_heads = 12  # default
        for heads in possible_heads:
            if hidden_size % heads == 0:
                num_heads = heads
                break
        
        logger.info(f"Inferred model architecture: vocab_size={vocab_size}, hidden_size={hidden_size}, "
                   f"num_layers={num_layers}, num_heads={num_heads}, max_length={max_length}")
        
        self.model = AnomalyDetector(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_length=max_length
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Apply optimization if specified
        if optimization:
            logger.info(f"Applying model optimization: {optimization}")
            self.model = optimize_model(self.model, method=optimization)
        
        self.scorer = AnomalyScorer(self.model, threshold=self.threshold, device=self.device)
        self.preparator = SequencePreparator(self.tokenizer)
        self.pipeline = ParsingPipeline()
        
        logger.info("Model and tokenizer loaded successfully")
    
    async def check_request_async(
        self,
        method: str,
        path: str,
        query_params: Dict = None,
        headers: Dict = None,
        body: Optional[str] = None
    ) -> Dict:
        """Check request asynchronously"""
        start_time = time.time()
        
        with self.metrics_lock:
            self.total_requests += 1
        
        try:
            # Run inference in thread pool
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.executor,
                    self._check_request_sync,
                    method, path, query_params, headers, body
                ),
                timeout=self.timeout
            )
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Log anomaly if detected
            if result.get('is_anomaly', False):
                self._log_anomaly(
                    method=method,
                    path=path,
                    query_params=query_params,
                    headers=headers,
                    body=body,
                    anomaly_score=result.get('anomaly_score', 0.0),
                    reason=result.get('reason')
                )
            
            # Update metrics
            self._update_metrics(processing_time, result['is_anomaly'])
            
            result['processing_time_ms'] = processing_time
            return result
            
        except asyncio.TimeoutError:
            logger.warning("Request check timed out")
            return {
                'anomaly_score': 0.0,
                'is_anomaly': False,
                'threshold': self.threshold,
                'error': 'timeout',
                'processing_time_ms': self.timeout * 1000
            }
        except Exception as e:
            logger.error(f"Error checking request: {e}")
            return {
                'anomaly_score': 0.0,
                'is_anomaly': False,
                'threshold': self.threshold,
                'error': str(e),
                'processing_time_ms': (time.time() - start_time) * 1000
            }
    
    def _check_request_sync(
        self,
        method: str,
        path: str,
        query_params: Dict = None,
        headers: Dict = None,
        body: Optional[str] = None
    ) -> Dict:
        """Synchronous request check (runs in thread pool)"""
        # Reconstruct request
        request_text = self._reconstruct_request(
            method, path, query_params, headers, body
        )
        
        # Normalize
        normalized = self.pipeline.process_log_line(request_text)
        if not normalized:
            # Fallback normalization
            normalized = self._create_normalized_text(method, path, query_params, headers, body)
            if not normalized:
                return {
                    'anomaly_score': 0.8,
                    'is_anomaly': True,
                    'threshold': self.threshold,
                    'reason': 'parsing_failed'
                }
        
        # Get max length from model
        max_length = self.model.transformer.embeddings.position_embeddings.num_embeddings
        
        # Tokenize
        token_ids, attention_mask = self.preparator.prepare_sequence(
            normalized,
            max_length=max_length,
            padding=True,
            truncation=True
        )
        
        # Convert to tensors
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        attn_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        # Score
        result = self.scorer.score(input_ids, attn_mask)
        return result
    
    async def check_batch_async(
        self,
        requests: List[Dict]
    ) -> List[Dict]:
        """Check batch of requests asynchronously"""
        # Process in batches
        results = []
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
        
        return results
    
    async def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process batch of requests"""
        loop = asyncio.get_event_loop()
        
        # Prepare batch inputs
        input_ids_list = []
        attention_masks_list = []
        valid_indices = []
        
        for idx, req in enumerate(batch):
            request_text = self._reconstruct_request(
                req['method'],
                req['path'],
                req.get('query_params', {}),
                req.get('headers', {}),
                req.get('body')
            )
            
            normalized = self.pipeline.process_log_line(request_text)
            if not normalized:
                normalized = self._create_normalized_text(
                    req['method'],
                    req['path'],
                    req.get('query_params', {}),
                    req.get('headers', {}),
                    req.get('body')
                )
            
            if normalized:
                max_length = self.model.transformer.embeddings.position_embeddings.num_embeddings
                token_ids, attn_mask = self.preparator.prepare_sequence(
                    normalized, max_length=max_length, padding=True, truncation=True
                )
                input_ids_list.append(token_ids)
                attention_masks_list.append(attn_mask)
                valid_indices.append(idx)
        
        # Batch inference
        if input_ids_list:
            input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(self.device)
            attention_masks = torch.tensor(attention_masks_list, dtype=torch.long).to(self.device)
            
            # Run inference
            batch_results = await loop.run_in_executor(
                self.executor,
                self._batch_inference,
                input_ids,
                attention_masks
            )
            
            # Map results back to original batch order and log anomalies
            results = []
            result_idx = 0
            for idx in range(len(batch)):
                if idx in valid_indices:
                    result = batch_results[result_idx]
                    # Log anomaly if detected
                    if result.get('is_anomaly', False):
                        req = batch[idx]
                        self._log_anomaly(
                            method=req['method'],
                            path=req['path'],
                            query_params=req.get('query_params', {}),
                            headers=req.get('headers', {}),
                            body=req.get('body'),
                            anomaly_score=result.get('anomaly_score', 0.0),
                            reason=result.get('reason')
                        )
                    results.append(result)
                    result_idx += 1
                else:
                    # Parsing failed - log as anomaly
                    req = batch[idx]
                    self._log_anomaly(
                        method=req['method'],
                        path=req['path'],
                        query_params=req.get('query_params', {}),
                        headers=req.get('headers', {}),
                        body=req.get('body'),
                        anomaly_score=0.8,
                        reason='parsing_failed'
                    )
                    results.append({
                        'anomaly_score': 0.8,
                        'is_anomaly': True,
                        'threshold': self.threshold,
                        'reason': 'parsing_failed'
                    })
            
            return results
        
        return [{'anomaly_score': 0.0, 'is_anomaly': False, 'threshold': self.threshold}] * len(batch)
    
    def _batch_inference(
        self,
        input_ids: torch.Tensor,
        attention_masks: torch.Tensor
    ) -> List[Dict]:
        """Perform batch inference"""
        with torch.no_grad():
            outputs = self.model(input_ids, attention_masks)
            scores = outputs['anomaly_score'].cpu().numpy()
        
        results = []
        for score in scores:
            results.append({
                'anomaly_score': float(score),
                'is_anomaly': float(score) > self.threshold,
                'threshold': self.threshold
            })
        
        return results
    
    def _reconstruct_request(
        self,
        method: str,
        path: str,
        query_params: Dict = None,
        headers: Dict = None,
        body: Optional[str] = None
    ) -> str:
        """Reconstruct request string"""
        query_str = ""
        if query_params:
            query_parts = []
            for k, v in query_params.items():
                if isinstance(v, list):
                    for val in v:
                        query_parts.append(f"{k}={val}")
                else:
                    query_parts.append(f"{k}={v}")
            query_str = "?" + "&".join(query_parts)
        
        request_line = f'{method} {path}{query_str} HTTP/1.1'
        
        # Add headers
        if headers:
            header_lines = []
            for name, value in headers.items():
                header_lines.append(f"{name}: {value}")
            request_line += "\n" + "\n".join(header_lines)
        
        # Add body
        if body:
            request_line += "\n\n" + body
        
        return request_line
    
    def _create_normalized_text(
        self,
        method: str,
        path: str,
        query_params: Dict = None,
        headers: Dict = None,
        body: Optional[str] = None
    ) -> str:
        """Create normalized text representation directly from request components"""
        normalized_parts = [method, path]
        
        # Add query parameters
        if query_params:
            query_parts = []
            for key, value in query_params.items():
                if isinstance(value, list):
                    for val in value:
                        query_parts.append(f"{key}={val}")
                else:
                    query_parts.append(f"{key}={value}")
            if query_parts:
                normalized_parts.append("?" + "&".join(query_parts))
        
        # Add important headers
        important_headers = ['user-agent', 'accept', 'content-type', 'referer']
        if headers:
            for header_name in important_headers:
                header_value = headers.get(header_name.lower())
                if header_value:
                    normalized_parts.append(f"{header_name}:{header_value}")
        
        # Add body if present (truncated)
        if body and len(body) > 0:
            truncated_body = body[:200] if len(body) > 200 else body
            normalized_parts.append(f"body:{truncated_body}")
        
        return " ".join(normalized_parts)
    
    def _log_anomaly(
        self,
        method: str,
        path: str,
        query_params: Dict = None,
        headers: Dict = None,
        body: Optional[str] = None,
        anomaly_score: float = 0.0,
        reason: Optional[str] = None
    ):
        """Log detected anomaly with request details"""
        # Build request summary
        request_summary = {
            'method': method,
            'path': path,
            'query_params': query_params or {},
            'anomaly_score': anomaly_score,
            'threshold': self.threshold,
            'reason': reason,
            'timestamp': time.time()
        }
        
        # Add headers (sanitized)
        if headers:
            sanitized_headers = {}
            for k, v in headers.items():
                # Don't log sensitive headers
                if k.lower() not in ['authorization', 'cookie', 'x-api-key']:
                    sanitized_headers[k] = v[:100] if isinstance(v, str) and len(v) > 100 else v
            request_summary['headers'] = sanitized_headers
        
        # Add body (truncated)
        if body:
            request_summary['body_preview'] = body[:200] if len(body) > 200 else body
        
        # Log to logger
        logger.warning(
            f"ANOMALY DETECTED: {method} {path} | "
            f"Score: {anomaly_score:.4f} | "
            f"Threshold: {self.threshold} | "
            f"Reason: {reason or 'model_detection'}"
        )
        logger.debug(f"Anomaly details: {request_summary}")
        
        # Log to file if specified
        if self.anomaly_log_file:
            try:
                import json
                from pathlib import Path
                log_path = Path(self.anomaly_log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(log_path, 'a') as f:
                    f.write(json.dumps(request_summary) + '\n')
            except Exception as e:
                logger.error(f"Failed to write anomaly log: {e}")
    
    def _update_metrics(self, processing_time: float, is_anomaly: bool):
        """Update service metrics"""
        with self.metrics_lock:
            if is_anomaly:
                self.anomalies_detected += 1
            
            # Exponential moving average
            alpha = 0.1
            self.avg_processing_time = (
                alpha * processing_time + (1 - alpha) * self.avg_processing_time
            )
    
    async def _queue_processor(self, requests: List[Dict]) -> List[Dict]:
        """Process requests from queue manager"""
        return await self.check_batch_async(requests)
    
    def get_metrics(self) -> Dict:
        """Get service metrics"""
        with self.metrics_lock:
            return {
                'total_requests': self.total_requests,
                'anomalies_detected': self.anomalies_detected,
                'anomaly_rate': (
                    self.anomalies_detected / self.total_requests
                    if self.total_requests > 0 else 0.0
                ),
                'avg_processing_time_ms': self.avg_processing_time,
                'device': self.device,
                'threshold': self.threshold,
                'batch_size': self.batch_size
            }
    
    async def check_request_via_queue(
        self,
        method: str,
        path: str,
        query_params: Dict = None,
        headers: Dict = None,
        body: Optional[str] = None
    ) -> Dict:
        """Check request via queue manager (if enabled)"""
        if not self.queue_manager:
            # Fallback to direct async check
            return await self.check_request_async(method, path, query_params, headers, body)
        
        request = {
            'method': method,
            'path': path,
            'query_params': query_params or {},
            'headers': headers or {},
            'body': body
        }
        
        return await self.queue_manager.enqueue(request)
    
    async def shutdown_async(self):
        """Shutdown service and cleanup resources (async)"""
        if self.queue_manager:
            # Stop queue manager
            await self.queue_manager.stop()
        self.executor.shutdown(wait=True)
        logger.info("Async WAF Service shutdown complete")
    
    def shutdown(self):
        """Shutdown service and cleanup resources (sync wrapper)"""
        if self.queue_manager:
            # Create event loop if needed
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule shutdown
                    asyncio.create_task(self.queue_manager.stop())
                else:
                    loop.run_until_complete(self.queue_manager.stop())
            except RuntimeError:
                # No event loop, create one
                asyncio.run(self.queue_manager.stop())
        self.executor.shutdown(wait=True)
        logger.info("Async WAF Service shutdown complete")


# Global service instance
waf_service: Optional[AsyncWAFService] = None

def initialize_service(
    model_path: str,
    vocab_path: str,
    threshold: float = 0.5,
    max_workers: int = 4,
    batch_size: int = 32,
    timeout: float = 5.0,
    device: str = None,
    optimization: Optional[str] = None,
    use_queue_manager: bool = False,
    queue_max_size: int = 1000,
    queue_batch_timeout: float = 0.1,
    anomaly_log_file: Optional[str] = None
):
    """Initialize async WAF service"""
    global waf_service
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    waf_service = AsyncWAFService(
        model_path=model_path,
        vocab_path=vocab_path,
        threshold=threshold,
        device=device,
        max_workers=max_workers,
        batch_size=batch_size,
        timeout=timeout,
        optimization=optimization,
        use_queue_manager=use_queue_manager,
        queue_max_size=queue_max_size,
        queue_batch_timeout=queue_batch_timeout,
        anomaly_log_file=anomaly_log_file
    )

# Request/Response models
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
    error: Optional[str] = None
    reason: Optional[str] = None

# Global rate limiter (optional)
rate_limiter: Optional[RateLimiter] = None
per_ip_rate_limiter: Optional[PerIPRateLimiter] = None

def initialize_rate_limiting(
    enabled: bool = False,
    max_requests_per_second: int = 100,
    per_ip: bool = False,
    max_ips: int = 10000
):
    """Initialize rate limiting"""
    global rate_limiter, per_ip_rate_limiter
    
    if not enabled:
        rate_limiter = None
        per_ip_rate_limiter = None
        return
    
    if per_ip:
        per_ip_rate_limiter = PerIPRateLimiter(
            max_requests=max_requests_per_second,
            window_seconds=1,
            max_ips=max_ips
        )
        logger.info(f"Per-IP rate limiting enabled: {max_requests_per_second} req/s per IP")
    else:
        rate_limiter = RateLimiter(
            max_requests=max_requests_per_second,
            window_seconds=1
        )
        logger.info(f"Global rate limiting enabled: {max_requests_per_second} req/s")

# FastAPI App
app = FastAPI(title="Async WAF Service", version="1.0.0")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global waf_service
    if waf_service:
        waf_service.shutdown()

def get_client_ip(request: Request = None) -> str:
    """Get client IP address from request"""
    if request is None:
        return "unknown"
    
    # Check for forwarded IP
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    # Check for real IP
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to client host
    if hasattr(request, 'client') and request.client:
        return request.client.host
    
    return "unknown"

@app.post("/check", response_model=CheckResponse)
async def check_request(request: CheckRequest, http_request: Request = None):
    """Check single request"""
    global waf_service, rate_limiter, per_ip_rate_limiter
    
    if waf_service is None:
        raise HTTPException(status_code=503, detail="WAF service not initialized")
    
    # Rate limiting check
    if per_ip_rate_limiter:
        client_ip = get_client_ip(http_request)
        if not per_ip_rate_limiter.is_allowed(client_ip):
            wait_time = per_ip_rate_limiter.get_wait_time(client_ip)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Please wait {wait_time:.2f}s"
            )
    elif rate_limiter:
        if not rate_limiter.is_allowed():
            wait_time = rate_limiter.get_wait_time()
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Please wait {wait_time:.2f}s"
            )
    
    # Use queue manager if enabled, otherwise direct async
    if waf_service.queue_manager:
        result = await waf_service.check_request_via_queue(
            method=request.method,
            path=request.path,
            query_params=request.query_params or {},
            headers=request.headers or {},
            body=request.body
        )
    else:
        result = await waf_service.check_request_async(
            method=request.method,
            path=request.path,
            query_params=request.query_params or {},
            headers=request.headers or {},
            body=request.body
        )
    
    return CheckResponse(
        anomaly_score=result['anomaly_score'],
        is_anomaly=result['is_anomaly'],
        threshold=result['threshold'],
        processing_time_ms=result.get('processing_time_ms', 0.0),
        error=result.get('error'),
        reason=result.get('reason')
    )

@app.post("/check/batch")
async def check_batch(requests: List[CheckRequest], http_request: Request = None):
    """Check batch of requests"""
    global waf_service, rate_limiter, per_ip_rate_limiter
    
    if waf_service is None:
        raise HTTPException(status_code=503, detail="WAF service not initialized")
    
    # Rate limiting check (check once per batch)
    if per_ip_rate_limiter:
        client_ip = get_client_ip(http_request)
        if not per_ip_rate_limiter.is_allowed(client_ip):
            wait_time = per_ip_rate_limiter.get_wait_time(client_ip)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Please wait {wait_time:.2f}s"
            )
    elif rate_limiter:
        if not rate_limiter.is_allowed():
            wait_time = rate_limiter.get_wait_time()
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Please wait {wait_time:.2f}s"
            )
    
    # For batch requests, if queue manager is enabled, process individually via queue
    # Otherwise use direct batch processing
    if waf_service.queue_manager:
        # Process each request through queue (they'll be batched by queue manager)
        tasks = [
            waf_service.check_request_via_queue(
                method=req.method,
                path=req.path,
                query_params=req.query_params or {},
                headers=req.headers or {},
                body=req.body
            )
            for req in requests
        ]
        results = await asyncio.gather(*tasks)
    else:
        request_dicts = [req.dict() for req in requests]
        results = await waf_service.check_batch_async(request_dicts)
    
    return results

@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    if waf_service is None:
        raise HTTPException(status_code=503, detail="WAF service not initialized")
    
    return waf_service.get_metrics()

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "service": "async_waf",
        "model_loaded": waf_service is not None
    }
