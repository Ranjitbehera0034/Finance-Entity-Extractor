"""
API Tests - Comprehensive Tests for FastAPI Endpoints.

This module contains unit tests for the REST API, including:
- Import tests
- Endpoint functionality tests
- Integration tests with TestClient

Run tests:
    $ pytest tests/test_api.py -v

Author: Ranjit Behera
License: MIT
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Import Tests
# =============================================================================

class TestAPIImports:
    """Test that all API components can be imported."""
    
    def test_import_server(self) -> None:
        """Test server module import."""
        from api.server import app, create_app
        assert app is not None
        assert callable(create_app)
    
    def test_import_extractor(self) -> None:
        """Test extractor import from data module."""
        from data.extractor import EntityExtractor
        extractor = EntityExtractor()
        assert extractor is not None
    
    def test_import_classifier(self) -> None:
        """Test classifier import from data module."""
        from data.classifier import EmailClassifier
        classifier = EmailClassifier()
        assert classifier is not None
    
    def test_import_models(self) -> None:
        """Test Pydantic models import."""
        from api.server import (
            EmailInput,
            EntityResponse,
            ClassificationResponse,
            HealthResponse,
        )
        assert EmailInput is not None
        assert EntityResponse is not None


# =============================================================================
# Logic Tests
# =============================================================================

class TestExtractionLogic:
    """Test entity extraction logic directly."""
    
    def test_extraction_basic(self) -> None:
        """Test basic entity extraction."""
        from data.extractor import EntityExtractor
        
        extractor = EntityExtractor()
        result = extractor.extract(
            "Rs.2500 debited from account 1234 on 05-01-26"
        )
        
        assert result.amount == "2500"
        assert result.type == "debit"
        assert result.account is not None
    
    def test_extraction_merchants(self) -> None:
        """Test merchant detection."""
        from data.extractor import EntityExtractor
        
        extractor = EntityExtractor()
        result = extractor.extract(
            "Rs.500 debited to swiggy@ybl via UPI"
        )
        
        assert result.merchant == "swiggy"
        assert result.payment_method == "upi"
    
    def test_extraction_full_email(self) -> None:
        """Test full email extraction."""
        from data.extractor import EntityExtractor
        
        extractor = EntityExtractor()
        result = extractor.extract(
            "HDFC Bank: Rs.2500.00 debited from A/c **3545 "
            "on 05-01-26 to VPA swiggy@ybl. Ref: 123456789012"
        )
        
        assert result.is_valid()
        assert result.confidence_score() >= 0.8


class TestClassificationLogic:
    """Test classification logic directly."""
    
    def test_finance_classification(self) -> None:
        """Test finance email classification."""
        from data.classifier import EmailClassifier
        
        classifier = EmailClassifier()
        result = classifier.classify(
            subject="Transaction Alert",
            sender="HDFC Bank",
            body="Rs.500 debited from your account"
        )
        
        assert result.category == "finance"
        assert result.is_transaction is True
    
    def test_shopping_classification(self) -> None:
        """Test shopping email classification."""
        from data.classifier import EmailClassifier
        
        classifier = EmailClassifier()
        result = classifier.classify(
            subject="Your order has shipped",
            sender="Amazon.in",
            body="Your order #12345 is on the way"
        )
        
        assert result.category == "shopping"
    
    def test_non_finance_classification(self) -> None:
        """Test non-finance classification."""
        from data.classifier import EmailClassifier
        
        classifier = EmailClassifier()
        result = classifier.classify(
            subject="Weekly Newsletter",
            sender="Substack",
            body="Top 10 articles this week"
        )
        
        assert result.category == "newsletter"
        assert result.is_transaction is False


# =============================================================================
# FastAPI Endpoint Tests
# =============================================================================

class TestFastAPIClient:
    """Test API endpoints using TestClient."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app)
    
    def test_root_endpoint(self, client) -> None:
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "endpoints" in data
        assert data["name"] == "LLM Mail Trainer API"
    
    def test_health_endpoint(self, client) -> None:
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data
    
    def test_stats_endpoint(self, client) -> None:
        """Test statistics endpoint."""
        response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "uptime_seconds" in data
    
    def test_extract_endpoint(self, client) -> None:
        """Test entity extraction endpoint."""
        response = client.post(
            "/extract",
            json={
                "subject": "Transaction Alert",
                "body": "Rs.2500.00 debited from account 3545 on 05-01-26",
                "sender": "HDFC Bank"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "entities" in data
        assert data["entities"]["amount"] == "2500.00"
    
    def test_extract_endpoint_validation(self, client) -> None:
        """Test extract endpoint validation."""
        response = client.post(
            "/extract",
            json={
                "body": ""  # Empty body should fail validation
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_classify_endpoint(self, client) -> None:
        """Test classification endpoint."""
        response = client.post(
            "/classify",
            json={
                "subject": "Transaction Alert",
                "body": "Your account has been debited",
                "sender": "HDFC Bank"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "finance"
        assert "confidence" in data
    
    def test_analyze_endpoint(self, client) -> None:
        """Test full analysis endpoint."""
        response = client.post(
            "/analyze",
            json={
                "subject": "Transaction Alert",
                "body": "Rs.500 debited from account 1234 on 01-01-26",
                "sender": "HDFC Bank"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "classification" in data
        assert "entities" in data  # Should have entities for finance
        assert data["classification"]["category"] == "finance"
    
    def test_batch_endpoint(self, client) -> None:
        """Test batch processing endpoint."""
        response = client.post(
            "/batch",
            json={
                "emails": [
                    {
                        "subject": "Transaction 1",
                        "body": "Rs.100 debited",
                        "sender": "Bank"
                    },
                    {
                        "subject": "Transaction 2",
                        "body": "Rs.200 credited",
                        "sender": "Bank"
                    }
                ]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 2
        assert "results" in data
        assert len(data["results"]) == 2


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app)
    
    def test_empty_body(self, client) -> None:
        """Test handling of empty body."""
        response = client.post(
            "/extract",
            json={
                "body": "   "  # Whitespace only
            }
        )
        
        assert response.status_code == 422
    
    def test_very_long_body(self, client) -> None:
        """Test handling of very long body."""
        long_body = "Rs.100 debited. " * 100
        response = client.post(
            "/extract",
            json={"body": long_body}
        )
        
        assert response.status_code == 200
    
    def test_unicode_content(self, client) -> None:
        """Test handling of unicode content."""
        response = client.post(
            "/extract",
            json={
                "body": "₹500 डेबिट from खाता 1234"
            }
        )
        
        assert response.status_code == 200
    
    def test_batch_empty_list(self, client) -> None:
        """Test batch with empty list."""
        response = client.post(
            "/batch",
            json={"emails": []}
        )
        
        assert response.status_code == 422  # Validation error


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Test API performance."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app)
    
    def test_extraction_speed(self, client) -> None:
        """Test extraction completes quickly."""
        import time
        
        start = time.time()
        response = client.post(
            "/extract",
            json={"body": "Rs.500 debited on 01-01-26"}
        )
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 1.0  # Should complete in under 1 second
    
    def test_batch_performance(self, client) -> None:
        """Test batch processing performance."""
        import time
        
        emails = [
            {"body": f"Rs.{i*100} debited", "subject": f"Txn {i}"}
            for i in range(10)
        ]
        
        start = time.time()
        response = client.post("/batch", json={"emails": emails})
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 5.0  # 10 emails in under 5 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
