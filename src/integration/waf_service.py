"""
WAF Service Module

FastAPI service for real-time WAF integration with trained Transformer model
"""
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional, List
import torch
from loguru import logger
import time
import json
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import psutil
import os
import re

from src.model.anomaly_detector import AnomalyDetector
from src.model.scoring import AnomalyScorer
from src.tokenization.tokenizer import HTTPTokenizer
from src.tokenization.sequence_prep import SequencePreparator
from src.parsing.pipeline import ParsingPipeline


class WAFService:
    """WAF Service for real-time anomaly detection"""

    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        threshold: float = 0.5,
        device: str = "cpu",
        max_batch_size: int = 32,
        timeout: float = 5.0
    ):
        self.device = device
        self.threshold = threshold
        self.max_batch_size = max_batch_size
        self.timeout = timeout

        # Thread pool for CPU-bound model inference
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="waf-inference")

        # Metrics
        self.metrics = {
            'total_requests': 0,
            'anomalies_detected': 0,
            'processing_times': [],
            'last_health_check': time.time(),
            'uptime': time.time()
        }
        self.metrics_lock = threading.Lock()

        # Load tokenizer
        logger.info(f"Loading tokenizer from {vocab_path}")
        self.tokenizer = HTTPTokenizer()
        self.tokenizer.load_vocab(vocab_path)

        # Load model
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)

        # Extract model architecture from checkpoint
        state_dict = checkpoint['model_state_dict']

        # Infer architecture from state dict shapes
        vocab_size = checkpoint.get('vocab_size', len(self.tokenizer.word_to_id))

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

        # Infer number of layers from state dict
        num_layers = 0
        while f'transformer.transformer.layer.{num_layers}.attention.q_lin.weight' in state_dict:
            num_layers += 1

        # Infer number of heads (assuming standard transformer setup)
        # hidden_size should be divisible by num_heads
        possible_heads = [8, 12, 16]
        num_heads = 12  # default
        for heads in possible_heads:
            if hidden_size % heads == 0:
                num_heads = heads
                break

        logger.info(f"Inferred model architecture: vocab_size={vocab_size}, hidden_size={hidden_size}, "
                   f"num_layers={num_layers}, num_heads={num_heads}, max_length={max_length}")

        # Create model with correct architecture
        self.model = AnomalyDetector(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_length=max_length
        )

        # Load state dict
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        # Initialize scorer
        self.scorer = AnomalyScorer(self.model, threshold=threshold, device=device)
        self.preparator = SequencePreparator(self.tokenizer)
        self.pipeline = ParsingPipeline()

        logger.info("WAF Service initialized with real model")

    def check_request(
        self,
        method: str,
        path: str,
        query_params: Dict = None,
        headers: Dict = None,
        body: Optional[str] = None
    ) -> Dict:
        """Check if request is anomalous using real model inference"""
        start_time = time.time()

        try:
            # Reconstruct request string
            request_text = self._reconstruct_request(
                method, path, query_params, headers, body
            )

            # Try parsing with pipeline first
            normalized = self.pipeline.process_log_line(request_text)

            # If parsing fails, create normalized text directly from components
            if not normalized:
                logger.debug(f"Log parsing failed, using direct normalization for: {request_text[:100]}...")
                normalized = self._create_normalized_text(method, path, query_params, headers, body)

            # Tokenize and prepare
            token_ids, attention_mask = self.preparator.prepare_sequence(
                normalized,
                max_length=self.model.transformer.embeddings.position_embeddings.num_embeddings,
                padding=True,
                truncation=True
            )

            # Convert to tensors
            input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
            attn_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)

            # Score using real model
            result = self.scorer.score(input_ids, attn_mask)

            processing_time = (time.time() - start_time) * 1000

            # Update metrics
            with self.metrics_lock:
                self.metrics['total_requests'] += 1
                if result['is_anomaly']:
                    self.metrics['anomalies_detected'] += 1
                self.metrics['processing_times'].append(processing_time)
                # Keep only last 1000 processing times
                if len(self.metrics['processing_times']) > 1000:
                    self.metrics['processing_times'] = self.metrics['processing_times'][-1000:]

            return {
                'anomaly_score': result['anomaly_score'],
                'is_anomaly': result['is_anomaly'],
                'threshold': result['threshold'],
                'processing_time_ms': processing_time,
                'normalized_request': normalized[:200]  # First 200 chars for logging
            }

        except Exception as e:
            logger.error(f"Error checking request: {e}")
            processing_time = (time.time() - start_time) * 1000

            # On error, allow request but log (fail open)
            return {
                'anomaly_score': 0.0,
                'is_anomaly': False,
                'threshold': self.threshold,
                'error': str(e),
                'processing_time_ms': processing_time
            }

    async def check_request_async(
        self,
        method: str,
        path: str,
        query_params: Dict = None,
        headers: Dict = None,
        body: Optional[str] = None
    ) -> Dict:
        """Async wrapper for request checking using thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.check_request,
            method, path, query_params, headers, body
        )

    def _reconstruct_request(
        self,
        method: str,
        path: str,
        query_params: Dict = None,
        headers: Dict = None,
        body: Optional[str] = None
    ) -> str:
        """Reconstruct request string from components for parsing"""
        # Build request line
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
        header_lines = []
        if headers:
            for name, value in headers.items():
                # Normalize header name
                header_lines.append(f"{name}: {value}")

        # Combine into full request
        full_request = request_line
        if header_lines:
            full_request += "\n" + "\n".join(header_lines)

        # Add body if present
        if body:
            full_request += "\n\n" + body

        return full_request

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
        self.scorer.set_threshold(new_threshold)
        logger.info(f"WAF threshold updated to {new_threshold}")

    def _create_normalized_text(
        self,
        method: str,
        path: str,
        query_params: Dict = None,
        headers: Dict = None,
        body: Optional[str] = None
    ) -> str:
        """Create normalized text representation directly from request components"""
        # Start with method and path
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
                    # Normalize header values (remove sensitive info)
                    normalized_value = self._normalize_header_value(header_value)
                    normalized_parts.append(f"{header_name}:{normalized_value}")

        # Add body if present (truncated)
        if body and len(body) > 0:
            truncated_body = body[:200] if len(body) > 200 else body
            normalized_parts.append(f"body:{truncated_body}")

        return " ".join(normalized_parts)

    def _normalize_header_value(self, value: str) -> str:
        """Normalize header values to remove sensitive information"""
        # Simple normalization - in production this would be more sophisticated
        # Remove potential sensitive patterns
        value = re.sub(r'\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b', '[CARD_NUMBER]', value)  # Credit cards
        value = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', value)  # Emails
        value = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP_ADDRESS]', value)  # IPs
        value = re.sub(r'\b[A-Za-z0-9+/=]{20,}\b', '[BASE64]', value)  # Base64 strings
        return value


# Global WAF service instance
waf_service: Optional[WAFService] = None

def initialize_waf_service(
    model_path: str,
    vocab_path: str,
    threshold: float = 0.5,
    device: str = None
):
    """Initialize WAF service with real model"""
    global waf_service
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    waf_service = WAFService(
        model_path=model_path,
        vocab_path=vocab_path,
        threshold=threshold,
        device=device
    )

# FastAPI Models
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

# FastAPI App
app = FastAPI(
    title="WAF Service",
    version="1.0.0",
    description="Real-time Web Application Firewall using Transformer-based anomaly detection"
)

@app.on_event("startup")
async def startup_event():
    """Initialize WAF service on startup"""
    logger.info("WAF Service starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global waf_service
    if waf_service and hasattr(waf_service, 'executor'):
        waf_service.executor.shutdown(wait=True)
    logger.info("WAF Service shutting down...")

@app.post("/check", response_model=CheckResponse)
async def check_request(request: CheckRequest):
    """Check if request is anomalous using real model inference"""
    if waf_service is None:
        raise HTTPException(
            status_code=503,
            detail="WAF service not initialized. Please ensure model and vocab paths are configured."
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
        "model_loaded": waf_service is not None,
        "timestamp": time.time()
    }

    if waf_service:
        health_status.update({
            "device": waf_service.device,
            "threshold": waf_service.threshold,
            "vocab_size": len(waf_service.tokenizer.word_to_id)
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
        "vocab_size": len(waf_service.tokenizer.word_to_id)
    }