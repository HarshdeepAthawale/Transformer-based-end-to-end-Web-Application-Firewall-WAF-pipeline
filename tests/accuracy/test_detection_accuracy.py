"""
Accuracy Tests for WAF Detection

Tests detection accuracy (TPR, FPR, Precision, Recall) on malicious and benign traffic
"""
import pytest
import sys
from pathlib import Path
from typing import List, Dict
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.async_waf_service import AsyncWAFService
from tests.payloads.malicious_payloads import get_all_malicious_payloads


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
    """Create WAF service instance for testing"""
    if not Path(model_path).exists():
        pytest.skip(f"Model not found: {model_path}. Please train a model first.")
    
    if not Path(vocab_path).exists():
        pytest.skip(f"Vocabulary not found: {vocab_path}. Please generate vocabulary first.")
    
    service = AsyncWAFService(
        model_path=model_path,
        vocab_path=vocab_path,
        threshold=0.5,
        device="cpu",  # Use CPU for tests
        max_workers=2,
        batch_size=16,
        timeout=5.0
    )
    
    yield service
    
    # Cleanup
    service.shutdown()


def load_benign_requests() -> List[Dict]:
    """Load benign request samples"""
    # Try to load from file
    benign_file = project_root / "data" / "validation" / "benign_samples.json"
    if benign_file.exists():
        try:
            with open(benign_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'data' in data:
                    return data['data']
        except:
            pass
    
    # Fallback: Generate basic benign requests
    return [
        {"method": "GET", "path": "/api/users", "query_params": {"page": "1"}},
        {"method": "GET", "path": "/api/products", "query_params": {"category": "electronics"}},
        {"method": "POST", "path": "/api/login", "body": '{"username": "user", "password": "pass"}'},
        {"method": "GET", "path": "/api/search", "query_params": {"q": "laptop"}},
        {"method": "GET", "path": "/api/cart", "query_params": {}},
        {"method": "POST", "path": "/api/register", "body": '{"email": "user@example.com", "name": "John"}'},
        {"method": "GET", "path": "/api/profile", "query_params": {"id": "123"}},
        {"method": "PUT", "path": "/api/settings", "body": '{"theme": "dark"}'},
        {"method": "GET", "path": "/api/orders", "query_params": {"status": "completed"}},
        {"method": "DELETE", "path": "/api/cart/item", "query_params": {"item_id": "456"}},
    ]


@pytest.mark.asyncio
async def test_sql_injection_detection(waf_service):
    """Test SQL injection detection"""
    from tests.payloads.malicious_payloads import SQL_INJECTION_PAYLOADS
    
    detected = 0
    total = len(SQL_INJECTION_PAYLOADS)
    
    for payload in SQL_INJECTION_PAYLOADS:
        result = await waf_service.check_request_async(
            method="GET",
            path="/api/users",
            query_params={"id": payload}
        )
        
        if result.get('is_anomaly', False):
            detected += 1
    
    detection_rate = detected / total if total > 0 else 0.0
    print(f"\nSQL Injection Detection Rate: {detection_rate:.2%} ({detected}/{total})")
    
    # Note: Assertion threshold may need adjustment based on model performance
    assert detection_rate >= 0.70, f"Detection rate {detection_rate:.2%} below 70%"


@pytest.mark.asyncio
async def test_xss_detection(waf_service):
    """Test XSS detection"""
    from tests.payloads.malicious_payloads import XSS_PAYLOADS
    
    detected = 0
    total = len(XSS_PAYLOADS)
    
    for payload in XSS_PAYLOADS:
        result = await waf_service.check_request_async(
            method="GET",
            path="/search",
            query_params={"q": payload}
        )
        
        if result.get('is_anomaly', False):
            detected += 1
    
    detection_rate = detected / total if total > 0 else 0.0
    print(f"\nXSS Detection Rate: {detection_rate:.2%} ({detected}/{total})")
    
    assert detection_rate >= 0.70, f"Detection rate {detection_rate:.2%} below 70%"


@pytest.mark.asyncio
async def test_command_injection_detection(waf_service):
    """Test command injection detection"""
    from tests.payloads.malicious_payloads import COMMAND_INJECTION_PAYLOADS
    
    detected = 0
    total = len(COMMAND_INJECTION_PAYLOADS)
    
    for payload in COMMAND_INJECTION_PAYLOADS:
        result = await waf_service.check_request_async(
            method="POST",
            path="/api/execute",
            body=payload
        )
        
        if result.get('is_anomaly', False):
            detected += 1
    
    detection_rate = detected / total if total > 0 else 0.0
    print(f"\nCommand Injection Detection Rate: {detection_rate:.2%} ({detected}/{total})")
    
    assert detection_rate >= 0.70, f"Detection rate {detection_rate:.2%} below 70%"


@pytest.mark.asyncio
async def test_path_traversal_detection(waf_service):
    """Test path traversal detection"""
    from tests.payloads.malicious_payloads import PATH_TRAVERSAL_PAYLOADS
    
    detected = 0
    total = len(PATH_TRAVERSAL_PAYLOADS)
    
    for payload in PATH_TRAVERSAL_PAYLOADS:
        result = await waf_service.check_request_async(
            method="GET",
            path=f"/api/files/{payload}",
            query_params={}
        )
        
        if result.get('is_anomaly', False):
            detected += 1
    
    detection_rate = detected / total if total > 0 else 0.0
    print(f"\nPath Traversal Detection Rate: {detection_rate:.2%} ({detected}/{total})")
    
    assert detection_rate >= 0.70, f"Detection rate {detection_rate:.2%} below 70%"


@pytest.mark.asyncio
async def test_false_positive_rate(waf_service):
    """Test false positive rate on benign traffic"""
    benign_requests = load_benign_requests()
    
    false_positives = 0
    total = len(benign_requests)
    
    for request in benign_requests:
        result = await waf_service.check_request_async(
            method=request['method'],
            path=request['path'],
            query_params=request.get('query_params', {}),
            headers=request.get('headers', {}),
            body=request.get('body')
        )
        
        if result.get('is_anomaly', False):
            false_positives += 1
    
    false_positive_rate = false_positives / total if total > 0 else 0.0
    print(f"\nFalse Positive Rate: {false_positive_rate:.2%} ({false_positives}/{total})")
    
    # FPR should be low (ideally < 5%)
    assert false_positive_rate <= 0.10, f"FPR {false_positive_rate:.2%} above 10%"


@pytest.mark.asyncio
async def test_all_malicious_categories(waf_service):
    """Test detection across all malicious payload categories"""
    all_payloads = get_all_malicious_payloads()
    
    results_by_category = {}
    
    for category, payloads in all_payloads.items():
        detected = 0
        total = len(payloads)
        
        for payload in payloads[:10]:  # Test first 10 of each category for speed
            result = await waf_service.check_request_async(
                method="GET",
                path="/api/data",
                query_params={"id": payload}
            )
            
            if result.get('is_anomaly', False):
                detected += 1
        
        detection_rate = detected / total if total > 0 else 0.0
        results_by_category[category] = {
            'detected': detected,
            'total': total,
            'rate': detection_rate
        }
        
        print(f"{category}: {detection_rate:.2%} ({detected}/{total})")
    
    # Calculate overall detection rate
    total_detected = sum(r['detected'] for r in results_by_category.values())
    total_tested = sum(r['total'] for r in results_by_category.values())
    overall_rate = total_detected / total_tested if total_tested > 0 else 0.0
    
    print(f"\nOverall Detection Rate: {overall_rate:.2%} ({total_detected}/{total_tested})")
    
    assert overall_rate >= 0.70, f"Overall detection rate {overall_rate:.2%} below 70%"


@pytest.mark.asyncio
async def test_accuracy_metrics(waf_service):
    """Calculate comprehensive accuracy metrics (TPR, FPR, Precision, Recall)"""
    from tests.payloads.malicious_payloads import SQL_INJECTION_PAYLOADS, XSS_PAYLOADS
    
    # Test on malicious payloads (should be detected = True Positive)
    malicious_payloads = SQL_INJECTION_PAYLOADS[:20] + XSS_PAYLOADS[:20]
    
    true_positives = 0
    false_negatives = 0
    
    for payload in malicious_payloads:
        result = await waf_service.check_request_async(
            method="GET",
            path="/api/data",
            query_params={"id": payload}
        )
        
        if result.get('is_anomaly', False):
            true_positives += 1
        else:
            false_negatives += 1
    
    # Test on benign requests (should not be detected = True Negative)
    benign_requests = load_benign_requests()
    
    true_negatives = 0
    false_positives = 0
    
    for request in benign_requests:
        result = await waf_service.check_request_async(
            method=request['method'],
            path=request['path'],
            query_params=request.get('query_params', {})
        )
        
        if result.get('is_anomaly', False):
            false_positives += 1
        else:
            true_negatives += 1
    
    # Calculate metrics
    total_malicious = len(malicious_payloads)
    total_benign = len(benign_requests)
    
    tpr = true_positives / total_malicious if total_malicious > 0 else 0.0  # Recall
    fpr = false_positives / total_benign if total_benign > 0 else 0.0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = tpr  # Same as TPR
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\n" + "=" * 60)
    print("ACCURACY METRICS")
    print("=" * 60)
    print(f"True Positives (TP): {true_positives}")
    print(f"False Negatives (FN): {false_negatives}")
    print(f"True Negatives (TN): {true_negatives}")
    print(f"False Positives (FP): {false_positives}")
    print(f"\nTrue Positive Rate (TPR/Recall): {tpr:.2%}")
    print(f"False Positive Rate (FPR): {fpr:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1_score:.2%}")
    print("=" * 60)
    
    # Store results for report generation
    return {
        'tpr': tpr,
        'fpr': fpr,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': true_positives,
        'fn': false_negatives,
        'tn': true_negatives,
        'fp': false_positives
    }
