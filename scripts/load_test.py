#!/usr/bin/env python3
"""
Load Testing Script

Load test the WAF service with concurrent requests
"""
import asyncio
import aiohttp
import time
import statistics
import sys
from pathlib import Path
from loguru import logger
from typing import List, Dict
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def send_request(session: aiohttp.ClientSession, url: str, request_data: Dict) -> Dict:
    """Send single request"""
    start = time.time()
    try:
        async with session.post(url, json=request_data, timeout=aiohttp.ClientTimeout(total=10.0)) as response:
            result = await response.json()
            latency = (time.time() - start) * 1000  # ms
            return {
                'status': response.status,
                'latency': latency,
                'anomaly_detected': result.get('is_anomaly', False),
                'anomaly_score': result.get('anomaly_score', 0.0),
                'success': True
            }
    except asyncio.TimeoutError:
        return {
            'status': 'timeout',
            'latency': (time.time() - start) * 1000,
            'success': False,
            'error': 'timeout'
        }
    except Exception as e:
        return {
            'status': 'error',
            'latency': (time.time() - start) * 1000,
            'success': False,
            'error': str(e)
        }


async def load_test(
    url: str,
    num_requests: int,
    concurrent: int,
    request_template: Dict = None
) -> Dict:
    """
    Run load test
    
    Args:
        url: WAF service URL
        num_requests: Total number of requests
        concurrent: Number of concurrent requests
        request_template: Template for request data
    
    Returns:
        Dictionary with test results
    """
    logger.info(f"Starting load test: {num_requests} requests, {concurrent} concurrent")
    logger.info(f"Target URL: {url}")
    
    if request_template is None:
        request_template = {
            'method': 'GET',
            'path': '/api/test',
            'query_params': {},
            'headers': {},
            'body': None
        }
    
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        start_time = time.time()
        
        for i in range(num_requests):
            # Create request data
            request_data = request_template.copy()
            request_data['path'] = f"/api/test{i}"
            if 'query_params' in request_data:
                request_data['query_params']['id'] = str(i)
            
            task = send_request(session, url, request_data)
            tasks.append(task)
            
            # Control concurrency
            if len(tasks) >= concurrent:
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                tasks = []
        
        # Process remaining
        if tasks:
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        
        elapsed = time.time() - start_time
    
    # Calculate metrics
    successful_results = [r for r in results if r.get('success', False)]
    latencies = [r['latency'] for r in successful_results if 'latency' in r]
    
    metrics = {
        'total_requests': num_requests,
        'successful_requests': len(successful_results),
        'failed_requests': len(results) - len(successful_results),
        'total_time': elapsed,
        'throughput': num_requests / elapsed if elapsed > 0 else 0.0
    }
    
    if latencies:
        metrics.update({
            'avg_latency_ms': statistics.mean(latencies),
            'median_latency_ms': statistics.median(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
        })
        
        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        p50_idx = int(len(sorted_latencies) * 0.50)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)
        
        metrics['p50_latency_ms'] = sorted_latencies[p50_idx] if p50_idx < len(sorted_latencies) else sorted_latencies[-1]
        metrics['p95_latency_ms'] = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else sorted_latencies[-1]
        metrics['p99_latency_ms'] = sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else sorted_latencies[-1]
    else:
        metrics.update({
            'avg_latency_ms': 0.0,
            'median_latency_ms': 0.0,
            'min_latency_ms': 0.0,
            'max_latency_ms': 0.0,
            'p50_latency_ms': 0.0,
            'p95_latency_ms': 0.0,
            'p99_latency_ms': 0.0
        })
    
    # Count anomalies
    anomalies_detected = sum(1 for r in successful_results if r.get('anomaly_detected', False))
    metrics['anomalies_detected'] = anomalies_detected
    metrics['anomaly_rate'] = anomalies_detected / len(successful_results) if successful_results else 0.0
    
    # Error breakdown
    errors = {}
    for r in results:
        if not r.get('success', False):
            error_type = r.get('error', 'unknown')
            errors[error_type] = errors.get(error_type, 0) + 1
    metrics['errors'] = errors
    
    return metrics


def print_results(metrics: Dict):
    """Print load test results"""
    print("\n" + "=" * 60)
    print("LOAD TEST RESULTS")
    print("=" * 60)
    print(f"Total Requests: {metrics['total_requests']}")
    print(f"Successful: {metrics['successful_requests']}")
    print(f"Failed: {metrics['failed_requests']}")
    print(f"Total Time: {metrics['total_time']:.2f}s")
    print(f"\nThroughput: {metrics['throughput']:.2f} requests/second")
    print(f"\nLatency Metrics:")
    print(f"  Average: {metrics['avg_latency_ms']:.2f}ms")
    print(f"  Median (P50): {metrics['p50_latency_ms']:.2f}ms")
    print(f"  P95: {metrics['p95_latency_ms']:.2f}ms")
    print(f"  P99: {metrics['p99_latency_ms']:.2f}ms")
    print(f"  Min: {metrics['min_latency_ms']:.2f}ms")
    print(f"  Max: {metrics['max_latency_ms']:.2f}ms")
    print(f"\nAnomalies Detected: {metrics['anomalies_detected']} ({metrics['anomaly_rate']:.2%})")
    
    if metrics.get('errors'):
        print(f"\nErrors:")
        for error_type, count in metrics['errors'].items():
            print(f"  {error_type}: {count}")
    
    print("=" * 60)


def save_results(metrics: Dict, output_path: str):
    """Save results to JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load test WAF service")
    parser.add_argument("--url", default="http://localhost:8000/check", help="WAF service URL")
    parser.add_argument("--requests", type=int, default=1000, help="Number of requests")
    parser.add_argument("--concurrent", type=int, default=100, help="Concurrent requests")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    logger.info("Starting load test...")
    
    try:
        metrics = await load_test(
            url=args.url,
            num_requests=args.requests,
            concurrent=args.concurrent
        )
        
        print_results(metrics)
        
        if args.output:
            save_results(metrics, args.output)
        
        logger.info("Load test completed")
        
    except KeyboardInterrupt:
        logger.info("Load test interrupted by user")
    except Exception as e:
        logger.error(f"Load test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
