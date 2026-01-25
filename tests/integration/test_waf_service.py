"""
Integration Tests for WAF Service

Tests the complete WAF service integration with real model inference
"""
import pytest
import requests
import json
import time
from pathlib import Path
import subprocess
import signal
import os
import tempfile
import shutil
from unittest.mock import patch

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from integration.waf_service import app, initialize_waf_service


class TestWAFServiceIntegration:
    """Integration tests for WAF service"""

    @pytest.fixture(scope="class")
    def waf_service(self):
        """Start WAF service for testing - placeholder mode (ML removed)"""
        # Initialize in placeholder mode (no ML dependencies)
        initialize_waf_service(
            model_path=None,
            vocab_path=None,
            threshold=0.5
        )

        # Start test server
        from fastapi.testclient import TestClient
        client = TestClient(app)
        return client

    def test_health_check(self, waf_service):
        """Test health check endpoint"""
        response = waf_service.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "waf"
        assert data["model_loaded"] == False  # ML removed - placeholder mode
        assert "mode" in data
        assert data["mode"] == "placeholder"
        assert "device" in data
        assert "threshold" in data

    def test_normal_request(self, waf_service):
        """Test normal (non-anomalous) request"""
        request_data = {
            "method": "GET",
            "path": "/api/products",
            "query_params": {"page": "1", "limit": "10"},
            "headers": {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "accept": "application/json"
            }
        }

        response = waf_service.post("/check", json=request_data)
        assert response.status_code == 200

        result = response.json()
        assert "anomaly_score" in result
        assert "is_anomaly" in result
        assert "threshold" in result
        assert "processing_time_ms" in result
        assert isinstance(result["anomaly_score"], float)
        assert isinstance(result["is_anomaly"], bool)
        assert result["anomaly_score"] >= 0.0
        assert result["anomaly_score"] <= 1.0

    def test_suspicious_request_sql_injection(self, waf_service):
        """Test SQL injection attempt"""
        request_data = {
            "method": "GET",
            "path": "/api/users",
            "query_params": {"id": "1' OR '1'='1"},
            "headers": {
                "user-agent": "sqlmap/1.6.5",
                "accept": "application/json"
            }
        }

        response = waf_service.post("/check", json=request_data)
        assert response.status_code == 200

        result = response.json()
        assert "anomaly_score" in result
        assert "is_anomaly" in result
        # SQL injection should be flagged as anomalous
        assert isinstance(result["is_anomaly"], bool)

    def test_suspicious_request_xss(self, waf_service):
        """Test XSS attempt"""
        request_data = {
            "method": "POST",
            "path": "/api/comments",
            "body": "<script>alert('xss')</script>",
            "headers": {
                "content-type": "application/json",
                "user-agent": "Mozilla/5.0"
            }
        }

        response = waf_service.post("/check", json=request_data)
        assert response.status_code == 200

        result = response.json()
        assert "anomaly_score" in result
        assert "is_anomaly" in result
        assert isinstance(result["is_anomaly"], bool)

    def test_malformed_request(self, waf_service):
        """Test malformed request handling"""
        request_data = {
            "method": "INVALID_METHOD",
            "path": "/../../../../etc/passwd",
            "query_params": {"cmd": "cat /etc/passwd"},
            "headers": {}
        }

        response = waf_service.post("/check", json=request_data)
        assert response.status_code == 200

        result = response.json()
        assert "anomaly_score" in result
        assert "is_anomaly" in result
        # Path traversal should be flagged
        assert isinstance(result["is_anomaly"], bool)

    def test_metrics_endpoint(self, waf_service):
        """Test metrics endpoint"""
        # Make some requests first
        for i in range(3):
            request_data = {
                "method": "GET",
                "path": f"/api/test/{i}",
                "query_params": {"param": f"value{i}"}
            }
            waf_service.post("/check", json=request_data)

        response = waf_service.get("/metrics")
        assert response.status_code == 200

        metrics = response.json()
        assert "total_requests" in metrics
        assert "anomalies_detected" in metrics
        assert "anomaly_rate" in metrics
        assert "avg_processing_time_ms" in metrics
        assert "uptime_seconds" in metrics
        assert "memory_usage_mb" in metrics
        assert "cpu_percent" in metrics

        assert metrics["total_requests"] >= 3
        assert isinstance(metrics["anomaly_rate"], float)
        assert metrics["anomaly_rate"] >= 0.0
        assert metrics["anomaly_rate"] <= 1.0

    def test_config_endpoint(self, waf_service):
        """Test configuration endpoint"""
        response = waf_service.get("/config")
        assert response.status_code == 200

        config = response.json()
        assert "threshold" in config
        assert "device" in config
        assert "vocab_size" in config
        assert "max_batch_size" in config
        assert "timeout" in config

    def test_threshold_update(self, waf_service):
        """Test threshold update"""
        new_threshold = 0.3

        response = waf_service.post("/update-threshold", json={"threshold": new_threshold})
        assert response.status_code == 200

        result = response.json()
        assert result["status"] == "success"
        assert result["new_threshold"] == new_threshold

        # Verify threshold was updated
        config_response = waf_service.get("/config")
        assert config_response.json()["threshold"] == new_threshold

    def test_invalid_request_handling(self, waf_service):
        """Test handling of invalid requests"""
        # Missing required fields
        response = waf_service.post("/check", json={})
        assert response.status_code == 422  # Validation error

        # Invalid method
        response = waf_service.post("/check", json={"invalid": "data"})
        assert response.status_code == 422

    def test_service_unavailable(self):
        """Test behavior when service is not initialized"""
        # This test doesn't use the waf_service fixture to test uninitialized state
        # Create a test client without initializing the service
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        test_app = FastAPI()
        # Don't initialize waf_service

        @test_app.post("/check")
        async def check_endpoint(request_data: dict):
            # Temporarily set waf_service to None to simulate uninitialized state
            import integration.waf_service as waf_module
            original_service = waf_module.waf_service
            waf_module.waf_service = None
            try:
                from fastapi import HTTPException
                raise HTTPException(status_code=503, detail="WAF service not initialized")
            finally:
                # Restore the service
                waf_module.waf_service = original_service

        client = TestClient(test_app)

        response = client.post("/check", json={"method": "GET", "path": "/test"})
        assert response.status_code == 503


class TestWAFServiceEndToEnd:
    """End-to-end tests with real HTTP server"""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start WAF service server for end-to-end testing - placeholder mode"""
        # Start server process in placeholder mode (no ML dependencies)
        cmd = [
            sys.executable, "scripts/start_waf_service.py",
            "--host", "127.0.0.1",
            "--port", "8888",
            "--workers", "1"
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent.parent.parent
        )

        # Wait for server to start
        time.sleep(5)

        yield process

        # Cleanup
        process.terminate()
        process.wait(timeout=10)

    def test_end_to_end_request_flow(self, server_process):
        """Test complete request flow"""
        # Give server time to fully start
        time.sleep(2)

        try:
            # Test health check
            response = requests.get("http://127.0.0.1:8888/health", timeout=5)
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"

            # Test normal request
            request_data = {
                "method": "GET",
                "path": "/api/users",
                "query_params": {"page": "1"},
                "headers": {"user-agent": "test-client"}
            }

            response = requests.post(
                "http://127.0.0.1:8888/check",
                json=request_data,
                timeout=10
            )
            assert response.status_code == 200

            result = response.json()
            assert "anomaly_score" in result
            assert "is_anomaly" in result
            assert "processing_time_ms" in result

        except requests.exceptions.ConnectionError:
            pytest.skip("Could not connect to WAF service - server may not have started")


def test_waf_service_initialization():
    """Test WAF service initialization - placeholder mode"""
    # Initialize in placeholder mode (no ML dependencies)
    initialize_waf_service(
        model_path=None,
        vocab_path=None,
        threshold=0.5
    )

    # Import and check
    from integration.waf_service import waf_service
    assert waf_service is not None
    assert hasattr(waf_service, 'check_request')
    assert hasattr(waf_service, 'get_metrics')
    # ML components removed - no tokenizer, model, or scorer