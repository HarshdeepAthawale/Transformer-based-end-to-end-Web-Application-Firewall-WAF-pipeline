"""
Performance Tests for WAF Service

Tests latency, throughput, and concurrent request handling
"""
import pytest
import asyncio
import time
import statistics
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.async_waf_service import AsyncWAFService


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
    
    return str(project_root / "models" / "vocabularies" / "http_vocab.json")


@pytest.fixture
async def waf_service(model_path, vocab_path):
    """Create async WAF service for testing"""
    if not Path(model_path).exists():
        pytest.skip(f"Model not found: {model_path}. Please train a model first.")
    
    if not Path(vocab_path).exists():
        pytest.skip(f"Vocabulary not found: {vocab_path}. Please generate vocabulary first.")
    
    service = AsyncWAFService(
        model_path=model_path,
        vocab_path=vocab_path,
        threshold=0.5,
        device="cpu",
        max_workers=4,
        batch_size=32,
        timeout=5.0
    )
    
    yield service
    
    # Cleanup
    service.shutdown()


@pytest.mark.asyncio
async def test_latency(waf_service):
    """Test request latency"""
    latencies = []
    num_requests = 100
    
    print(f"\nTesting latency with {num_requests} requests...")
    
    for i in range(num_requests):
        start = time.time()
        await waf_service.check_request_async(
            method="GET",
            path=f"/api/test{i}",
            query_params={"id": str(i)}
        )
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
    
    avg_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    # Calculate percentiles
    sorted_latencies = sorted(latencies)
    p95_idx = int(len(sorted_latencies) * 0.95)
    p99_idx = int(len(sorted_latencies) * 0.99)
    
    p95_latency = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else sorted_latencies[-1]
    p99_latency = sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else sorted_latencies[-1]
    
    print(f"Average Latency: {avg_latency:.2f}ms")
    print(f"Median Latency: {median_latency:.2f}ms")
    print(f"Min Latency: {min_latency:.2f}ms")
    print(f"Max Latency: {max_latency:.2f}ms")
    print(f"P95 Latency: {p95_latency:.2f}ms")
    print(f"P99 Latency: {p99_latency:.2f}ms")
    
    # Store for report
    return {
        'avg_latency': avg_latency,
        'median_latency': median_latency,
        'min_latency': min_latency,
        'max_latency': max_latency,
        'p95_latency': p95_latency,
        'p99_latency': p99_latency
    }


@pytest.mark.asyncio
async def test_throughput(waf_service):
    """Test request throughput"""
    num_requests = 1000
    concurrent = 100
    
    print(f"\nTesting throughput: {num_requests} requests, {concurrent} concurrent...")
    
    start_time = time.time()
    
    # Create concurrent requests in batches
    tasks = []
    for i in range(num_requests):
        task = waf_service.check_request_async(
            method="GET",
            path=f"/api/test{i}",
            query_params={"id": str(i)}
        )
        tasks.append(task)
        
        # Batch concurrent requests
        if len(tasks) >= concurrent:
            await asyncio.gather(*tasks)
            tasks = []
    
    # Process remaining
    if tasks:
        await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    throughput = num_requests / elapsed
    
    print(f"Total requests: {num_requests}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.2f} requests/second")
    print(f"Average time per request: {(elapsed/num_requests)*1000:.2f}ms")
    
    return {
        'throughput': throughput,
        'total_time': elapsed,
        'num_requests': num_requests
    }


@pytest.mark.asyncio
async def test_concurrent_requests(waf_service):
    """Test concurrent request handling"""
    num_concurrent = 200
    
    print(f"\nTesting {num_concurrent} concurrent requests...")
    
    start_time = time.time()
    
    tasks = [
        waf_service.check_request_async(
            method="GET",
            path=f"/api/test{i}",
            query_params={"id": str(i)}
        )
        for i in range(num_concurrent)
    ]
    
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time
    
    print(f"Processed {num_concurrent} concurrent requests in {elapsed:.2f}s")
    print(f"Average: {(elapsed/num_concurrent)*1000:.2f}ms per request")
    print(f"Throughput: {num_concurrent/elapsed:.2f} req/s")
    
    assert len(results) == num_concurrent
    assert elapsed < 30.0, f"Concurrent processing too slow: {elapsed:.2f}s"
    
    return {
        'num_concurrent': num_concurrent,
        'elapsed': elapsed,
        'avg_time_per_request': (elapsed/num_concurrent)*1000
    }


@pytest.mark.asyncio
async def test_batch_processing_performance(waf_service):
    """Test batch processing performance"""
    num_requests = 500
    batch_size = 50
    
    print(f"\nTesting batch processing: {num_requests} requests in batches of {batch_size}...")
    
    # Create batch of requests
    requests = [
        {
            'method': 'GET',
            'path': f'/api/test{i}',
            'query_params': {'id': str(i)},
            'headers': {},
            'body': None
        }
        for i in range(num_requests)
    ]
    
    start_time = time.time()
    results = await waf_service.check_batch_async(requests)
    elapsed = time.time() - start_time
    
    print(f"Processed {num_requests} requests in {elapsed:.2f}s")
    print(f"Throughput: {num_requests/elapsed:.2f} req/s")
    print(f"Average: {(elapsed/num_requests)*1000:.2f}ms per request")
    
    assert len(results) == num_requests
    
    return {
        'num_requests': num_requests,
        'elapsed': elapsed,
        'throughput': num_requests/elapsed
    }


@pytest.mark.asyncio
async def test_under_load(waf_service):
    """Test performance under sustained load"""
    num_requests = 2000
    concurrent = 50
    duration_seconds = 30
    
    print(f"\nTesting under load: {num_requests} requests, {concurrent} concurrent, {duration_seconds}s duration...")
    
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    request_count = 0
    latencies = []
    
    while time.time() < end_time and request_count < num_requests:
        batch_start = time.time()
        
        # Create batch of concurrent requests
        tasks = [
            waf_service.check_request_async(
                method="GET",
                path=f"/api/test{request_count + i}",
                query_params={"id": str(request_count + i)}
            )
            for i in range(min(concurrent, num_requests - request_count))
        ]
        
        await asyncio.gather(*tasks)
        
        batch_latency = (time.time() - batch_start) * 1000
        latencies.append(batch_latency)
        request_count += len(tasks)
    
    elapsed = time.time() - start_time
    throughput = request_count / elapsed
    
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    
    print(f"Processed {request_count} requests in {elapsed:.2f}s")
    print(f"Sustained throughput: {throughput:.2f} req/s")
    print(f"Average batch latency: {avg_latency:.2f}ms")
    
    return {
        'request_count': request_count,
        'elapsed': elapsed,
        'throughput': throughput,
        'avg_latency': avg_latency
    }
