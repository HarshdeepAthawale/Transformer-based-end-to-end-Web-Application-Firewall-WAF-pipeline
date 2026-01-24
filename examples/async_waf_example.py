#!/usr/bin/env python3
"""
Example: Using Async WAF Service

Demonstrates how to use the async WAF service for real-time request checking
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.async_waf_service import AsyncWAFService
from src.inference.rate_limiter import RateLimiter, PerIPRateLimiter
from src.inference.queue_manager import RequestQueueManager
from loguru import logger


async def example_single_request():
    """Example: Check a single request"""
    logger.info("=== Example: Single Request ===")
    
    # Initialize service
    service = AsyncWAFService(
        model_path="models/checkpoints/best_model.pt",
        vocab_path="models/vocabularies/http_vocab.json",
        threshold=0.5,
        device="cpu",
        max_workers=2,
        batch_size=16,
        timeout=5.0
    )
    
    # Check request
    result = await service.check_request_async(
        method="GET",
        path="/api/users",
        query_params={"id": "123", "page": "1"},
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        },
        body=None
    )
    
    logger.info(f"Result: {result}")
    logger.info(f"Anomaly Score: {result['anomaly_score']:.4f}")
    logger.info(f"Is Anomaly: {result['is_anomaly']}")
    logger.info(f"Processing Time: {result['processing_time_ms']:.2f}ms")
    
    # Cleanup
    service.shutdown()


async def example_batch_requests():
    """Example: Check batch of requests"""
    logger.info("=== Example: Batch Requests ===")
    
    # Initialize service
    service = AsyncWAFService(
        model_path="models/checkpoints/best_model.pt",
        vocab_path="models/vocabularies/http_vocab.json",
        threshold=0.5,
        device="cpu",
        max_workers=4,
        batch_size=32,
        timeout=5.0
    )
    
    # Create batch of requests
    requests = [
        {
            'method': 'GET',
            'path': f'/api/users/{i}',
            'query_params': {},
            'headers': {'User-Agent': 'test-agent'},
            'body': None
        }
        for i in range(50)
    ]
    
    # Process batch
    results = await service.check_batch_async(requests)
    
    logger.info(f"Processed {len(results)} requests")
    anomalies = sum(1 for r in results if r['is_anomaly'])
    logger.info(f"Anomalies detected: {anomalies}")
    
    # Show metrics
    metrics = service.get_metrics()
    logger.info(f"Metrics: {metrics}")
    
    # Cleanup
    service.shutdown()


async def example_concurrent_requests():
    """Example: Process requests concurrently"""
    logger.info("=== Example: Concurrent Requests ===")
    
    # Initialize service
    service = AsyncWAFService(
        model_path="models/checkpoints/best_model.pt",
        vocab_path="models/vocabularies/http_vocab.json",
        threshold=0.5,
        device="cpu",
        max_workers=4,
        batch_size=32,
        timeout=5.0
    )
    
    # Create concurrent tasks
    tasks = [
        service.check_request_async(
            method="GET",
            path=f"/api/test{i}",
            query_params={"id": str(i)},
            headers={},
            body=None
        )
        for i in range(20)
    ]
    
    # Process concurrently
    results = await asyncio.gather(*tasks)
    
    logger.info(f"Processed {len(results)} concurrent requests")
    anomalies = sum(1 for r in results if r['is_anomaly'])
    logger.info(f"Anomalies detected: {anomalies}")
    
    # Cleanup
    service.shutdown()


async def example_rate_limiting():
    """Example: Rate limiting"""
    logger.info("=== Example: Rate Limiting ===")
    
    # Global rate limiter
    limiter = RateLimiter(max_requests=10, window_seconds=1)
    
    allowed = 0
    blocked = 0
    
    for i in range(15):
        if limiter.is_allowed():
            allowed += 1
            logger.info(f"Request {i}: Allowed")
        else:
            blocked += 1
            wait_time = limiter.get_wait_time()
            logger.info(f"Request {i}: Blocked (wait {wait_time:.2f}s)")
    
    logger.info(f"Allowed: {allowed}, Blocked: {blocked}")
    
    # Per-IP rate limiter
    ip_limiter = PerIPRateLimiter(max_requests=5, window_seconds=1)
    
    ip1_allowed = sum(1 for _ in range(10) if ip_limiter.is_allowed("192.168.1.1"))
    ip2_allowed = sum(1 for _ in range(10) if ip_limiter.is_allowed("192.168.1.2"))
    
    logger.info(f"IP 192.168.1.1: {ip1_allowed} allowed")
    logger.info(f"IP 192.168.1.2: {ip2_allowed} allowed")


async def example_queue_manager():
    """Example: Queue manager"""
    logger.info("=== Example: Queue Manager ===")
    
    queue_manager = RequestQueueManager(
        max_size=100,
        batch_timeout=0.1,
        batch_size=10
    )
    
    # Define processor
    async def processor(requests):
        # Simulate processing
        await asyncio.sleep(0.01)
        return [
            {
                'anomaly_score': 0.1,
                'is_anomaly': False,
                'threshold': 0.5
            }
            for _ in requests
        ]
    
    # Start processing
    queue_manager.start_processing(processor)
    
    # Enqueue requests
    tasks = [
        queue_manager.enqueue({
            'method': 'GET',
            'path': f'/api/test{i}',
            'query_params': {},
            'headers': {},
            'body': None
        })
        for i in range(25)
    ]
    
    results = await asyncio.gather(*tasks)
    
    logger.info(f"Processed {len(results)} requests via queue")
    logger.info(f"Queue size: {queue_manager.get_queue_size()}")
    
    # Stop processing
    await queue_manager.stop()


async def main():
    """Run all examples"""
    logger.info("Starting Async WAF Service Examples")
    
    # Check if model exists
    model_path = project_root / "models" / "checkpoints" / "best_model.pt"
    vocab_path = project_root / "models" / "vocabularies" / "http_vocab.json"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Please train a model first")
        return
    
    if not vocab_path.exists():
        logger.error(f"Vocabulary not found: {vocab_path}")
        logger.error("Please generate vocabulary first")
        return
    
    try:
        # Run examples that require model
        await example_single_request()
        await example_batch_requests()
        await example_concurrent_requests()
    except Exception as e:
        logger.error(f"Error in model examples: {e}")
        import traceback
        traceback.print_exc()
    
    # Run examples that don't require model
    await example_rate_limiting()
    await example_queue_manager()
    
    logger.info("Examples completed")


if __name__ == "__main__":
    asyncio.run(main())
