"""
Tests for the FastAPI server.
Run with: pytest tests/test_api.py -v
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestAPIImports:
    """Test that API modules can be imported."""
    
    def test_import_server(self):
        """Test server module import."""
        from api.server import app, create_app
        assert app is not None
        assert create_app is not None
    
    def test_import_extractor(self):
        """Test extractor import in server."""
        from api.server import extractor
        assert extractor is not None
    
    def test_import_classifier(self):
        """Test classifier import in server."""
        from api.server import classifier
        assert classifier is not None


class TestAPIEndpoints:
    """Test API endpoint logic (without running server)."""
    
    @pytest.fixture
    def sample_email(self):
        return {
            "subject": "❗ UPI Transaction Alert",
            "body": "Rs.2500.00 has been debited from account 3545 to VPA swiggy@ybl on 28-12-25. Reference: 534567891234",
            "sender": "HDFC Bank"
        }
    
    def test_extraction_logic(self, sample_email):
        """Test entity extraction logic."""
        from api.server import extractor
        
        result = extractor.extract(sample_email["body"])
        
        assert result.amount == "2500.00"
        assert result.type == "debit"
        assert result.account == "3545"
        assert result.merchant == "swiggy"
        assert result.payment_method == "upi"
    
    def test_classification_logic(self, sample_email):
        """Test classification logic."""
        from api.server import classifier
        
        result = classifier.classify(
            subject=sample_email["subject"],
            sender=sample_email["sender"],
            body=sample_email["body"]
        )
        
        assert result.category == "finance"
        assert result.confidence == "high"
        assert result.is_transaction == True
    
    def test_classification_non_finance(self):
        """Test classification of non-finance email."""
        from api.server import classifier
        
        result = classifier.classify(
            subject="Your order has shipped",
            sender="Amazon.in",
            body="Your order #123 has been shipped and will arrive tomorrow"
        )
        
        assert result.category == "shopping"
        assert result.is_transaction == False


class TestFastAPIClient:
    """Test API using TestClient."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        try:
            from fastapi.testclient import TestClient
            from api.server import app
            return TestClient(app)
        except ImportError:
            pytest.skip("fastapi.testclient not available")
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "LLM Mail Trainer API"
    
    def test_health_endpoint(self, client):
        """Test health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_extract_endpoint(self, client):
        """Test extraction endpoint."""
        response = client.post(
            "/extract",
            json={
                "subject": "UPI Alert",
                "body": "Rs.500.00 debited from account 1234 on 01-01-25",
                "sender": "Bank"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["entities"]["amount"] == "500.00"
        assert data["entities"]["type"] == "debit"
    
    def test_classify_endpoint(self, client):
        """Test classification endpoint."""
        response = client.post(
            "/classify",
            json={
                "subject": "Transaction Alert",
                "body": "Rs.500 debited from your account",
                "sender": "HDFC Bank"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "finance"
    
    def test_analyze_endpoint(self, client):
        """Test full analysis endpoint."""
        response = client.post(
            "/analyze",
            json={
                "subject": "UPI Txn",
                "body": "Rs.1000.00 credited to account 5678 on 02-02-25",
                "sender": "ICICI Bank"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "classification" in data
        assert "entities" in data
        assert data["classification"]["category"] == "finance"
    
    def test_batch_endpoint(self, client):
        """Test batch processing."""
        response = client.post(
            "/batch",
            json={
                "emails": [
                    {
                        "subject": "Payment",
                        "body": "Rs.100 debited",
                        "sender": "Bank"
                    },
                    {
                        "subject": "Order shipped",
                        "body": "Your order is on the way",
                        "sender": "Amazon"
                    }
                ]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 2
        assert len(data["results"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
