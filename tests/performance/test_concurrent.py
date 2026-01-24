"""
Performance Tests for Concurrent Request Processing

Tests for Phase 7: Real-Time Non-Blocking Detection
"""
import pytest
import asyncio
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.async_waf_service import AsyncWAFService, initialize_service
from src.inference.rate_limiter import RateLimiter, PerIPRateLimiter
from src.inference.queue_manager import RequestQueueManager


@pytest.fixture
def model_path():
    """Get model path from config or use default"""
    config_path = project_root / "config" / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            model_path = config.get('waf_service', {}).get('model_path', 'models/checkpoints/best_model.pt')
            model_path = project_root / model_path
            if model_path.exists():
                return str(model_path)
    
    # Fallback
    return str(project_root / "models" / "checkpoints" / "best_model.pt")


@pytest.fixture
def vocab_path():
    """Get vocab path from config or use default"""
    config_path = project_root / "config" / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            vocab_path = config.get('waf_service', {}).get('vocab_path', 'models/vocabularies/http_vocab.json')
            vocab_path = project_root / vocab_path
            if vocab_path.exists():
                return str(vocab_path)
    
    # Fallback
    return str(project_root / "models" / "vocabularies" / "http_vocab.json")


@pytest.fixture
async def waf_service(model_path, vocab_path):
    """Create WAF service instance for testing"""
    # Check if model exists
    if not Path(model_path).exists():
        pytest.skip(f"Model not found: {model_path}. Please train a model first.")
    
    if not Path(vocab_path).exists():
        pytest.skip(f"Vocabulary not found: {vocab_path}. Please generate vocabulary first.")
    
    service = AsyncWAFService(
        model_path=model_path,
        vocab_path=vocab_path,
        threshold=0.5,
        device="cpu",  # Use CPU for tests
        max_workers=4,
        batch_size=32,
        timeout=5.0
    )
    
    yield service
    
    # Cleanup
    service.shutdown()


@pytest.mark.asyncio
async def test_concurrent_requests(waf_service):
    """Test concurrent request processing"""
    # Create 100 concurrent requests
    requests = [
        {
            'method': 'GET',
            'path': f'/api/test{i}',
            'query_params': {'id': str(i)},
            'headers': {'User-Agent': 'test-agent'},
            'body': None
        }
        for i in range(100)
    ]
    
    start_time = time.time()
    
    # Process all requests concurrently
    tasks = [
        waf_service.check_request_async(
            method=req['method'],
            path=req['path'],
            query_params=req['query_params'],
            headers=req['headers'],
            body=req['body']
        )
        for req in requests
    ]
    
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time
    
    # Verify results
    assert len(results) == 100
    assert all('anomaly_score' in r for r in results)
    assert all('is_anomaly' in r for r in results)
    assert all('processing_time_ms' in r for r in results)
    
    # Performance assertions
    assert elapsed < 30.0, f"Processing took too long: {elapsed:.2f}s"
    
    print(f"\nProcessed 100 requests in {elapsed:.2f}s")
    print(f"Throughput: {100/elapsed:.2f} req/s")
    print(f"Average processing time: {sum(r['processing_time_ms'] for r in results)/100:.2f}ms")


@pytest.mark.asyncio
async def test_batch_processing(waf_service):
    """Test batch request processing"""
    # Create batch of requests
    requests = [
        {
            'method': 'GET',
            'path': f'/api/batch/{i}',
            'query_params': {},
            'headers': {},
            'body': None
        }
        for i in range(50)
    ]
    
    start_time = time.time()
    results = await waf_service.check_batch_async(requests)
    elapsed = time.time() - start_time
    
    # Verify results
    assert len(results) == 50
    assert all('anomaly_score' in r for r in results)
    
    print(f"\nProcessed batch of 50 requests in {elapsed:.2f}s")
    print(f"Throughput: {50/elapsed:.2f} req/s")


@pytest.mark.asyncio
async def test_timeout_handling(waf_service):
    """Test timeout handling"""
    # Set very short timeout
    waf_service.timeout = 0.001  # 1ms - should timeout
    
    result = await waf_service.check_request_async(
        method='GET',
        path='/api/test',
        query_params={},
        headers={},
        body=None
    )
    
    # Should get timeout error
    assert 'error' in result
    assert result['error'] == 'timeout' or 'timeout' in result.get('error', '').lower()


@pytest.mark.asyncio
async def test_rate_limiter():
    """Test rate limiter"""
    limiter = RateLimiter(max_requests=10, window_seconds=1)
    
    # First 10 requests should be allowed
    for i in range(10):
        assert limiter.is_allowed(), f"Request {i} should be allowed"
    
    # 11th request should be blocked
    assert not limiter.is_allowed(), "11th request should be blocked"
    
    # Wait for window to expire
    await asyncio.sleep(1.1)
    
    # Should be allowed again
    assert limiter.is_allowed(), "Request after window should be allowed"


@pytest.mark.asyncio
async def test_per_ip_rate_limiter():
    """Test per-IP rate limiter"""
    limiter = PerIPRateLimiter(max_requests=5, window_seconds=1)
    
    # Different IPs should have separate limits
    assert limiter.is_allowed("192.168.1.1")
    assert limiter.is_allowed("192.168.1.2")
    
    # Same IP should hit limit
    for i in range(4):
        assert limiter.is_allowed("192.168.1.1")
    
    assert not limiter.is_allowed("192.168.1.1"), "IP should be rate limited"


@pytest.mark.asyncio
async def test_queue_manager():
    """Test request queue manager"""
    queue_manager = RequestQueueManager(max_size=100, batch_timeout=0.1, batch_size=10)
    
    # Create processor function
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
        for i in range(20)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Verify results
    assert len(results) == 20
    assert all('anomaly_score' in r for r in results)
    
    # Stop processing
    await queue_manager.stop()


@pytest.mark.asyncio
async def test_metrics_collection(waf_service):
    """Test metrics collection"""
    # Process some requests
    for i in range(10):
        await waf_service.check_request_async(
            method='GET',
            path=f'/api/test{i}',
            query_params={},
            headers={},
            body=None
        )
    
    # Get metrics
    metrics = waf_service.get_metrics()
    
    assert 'total_requests' in metrics
    assert 'anomalies_detected' in metrics
    assert 'anomaly_rate' in metrics
    assert 'avg_processing_time_ms' in metrics
    
    assert metrics['total_requests'] == 10
    assert 0 <= metrics['anomaly_rate'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
