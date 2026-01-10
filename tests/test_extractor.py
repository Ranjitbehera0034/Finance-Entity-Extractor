"""
Integration Tests for FinEE Extractor (Pipeline).

Tests the full 5-tier pipeline including:
- Additive merge (Regex + Rules + LLM)
- Graceful degradation (No backend)
- Mocked LLM responses
"""

import pytest
from unittest.mock import MagicMock, patch
from finee.extractor import FinEE, ExtractionConfig
from finee.schema import ExtractionResult, TransactionType, Category, ExtractionSource
from finee.backends import BaseBackend

# Mock Backend for testing Tier 3
class MockBackend(BaseBackend):
    def is_available(self):
        return True
    
    def load_model(self, path=None):
        self._loaded = True
        return True
    
    def generate(self, prompt, **kwargs):
        # Respond based on prompt content
        if "merchant" in prompt.lower():
            return "Uber"
        if "category" in prompt.lower():
            return "transport"
        if "date" in prompt.lower():
            return "01-01-2025"
        return ""

@pytest.fixture
def extractor_no_llm():
    """Extractor with LLM disabled (Regex + Rules only)."""
    config = ExtractionConfig(use_llm=False)
    return FinEE(config)

@pytest.fixture
def extractor_with_mock_llm():
    """Extractor with Mock LLM (Full Pipeline)."""
    config = ExtractionConfig(use_llm=True)
    extractor = FinEE(config)
    extractor._backend = MockBackend()
    extractor._backend_loaded = True
    return extractor

def test_tier1_regex_only(extractor_no_llm):
    """Test Tier 1 regex extraction works without LLM."""
    text = "Rs.500.00 debited from A/c 1234"
    result = extractor_no_llm.extract(text)
    
    assert result.amount == 500.0
    assert result.type == TransactionType.DEBIT
    assert result.account == "1234"
    assert result.confidence_score > 0.0  # Should have some confidence

def test_tier2_rules_enrichment(extractor_no_llm):
    """Test Tier 2 rules (VPA -> Merchant) works without LLM."""
    text = "Rs.250 paid to swiggy@ybl"
    result = extractor_no_llm.extract(text)
    
    assert result.amount == 250.0
    assert result.vpa == "swiggy@ybl"
    assert result.merchant == "Swiggy"      # Tier 2
    assert result.category == Category.FOOD # Tier 2

def test_tier3_additive_merge(extractor_with_mock_llm):
    """
    Test Additive Merge:
    - Tier 1 gets Amount
    - Tier 3 Mock LLM gets Merchant/Category
    """
    text = "Rs.500 paid for taxi ride"
    
    # Mock LLM will return "Uber" and "transport" when asked
    result = extractor_with_mock_llm.extract(text)
    
    # Check mix of sources
    assert result.amount == 500.0             # Tier 1 (Regex)
    assert result.merchant == "Uber"          # Tier 3 (Mock LLM)
    assert result.category == Category.TRANSPORT # Tier 3 (Mock LLM)
    
    # Check source metadata
    assert result.meta['amount'].source == ExtractionSource.REGEX
    assert result.meta['merchant'].source == ExtractionSource.LLM

def test_graceful_degradation():
    """Test that pipeline works even if LLM fails/is missing."""
    config = ExtractionConfig(use_llm=True)
    extractor = FinEE(config)
    
    # Force backend to be None (simulate no backends installed)
    extractor._backend = None
    extractor._backend_loaded = True
    
    text = "Rs.100 debited"
    result = extractor.extract(text)
    
    # Should still get regex results
    assert result.amount == 100.0
    assert result.type == TransactionType.DEBIT
    # Should not crash

def test_full_pipeline_consistency():
    """Test consistent output through full pipeline."""
    # This text contains everything extractable by Regex + Rules
    text = "Rs.2500 debited from HDFC A/c 1234 to swiggy@ybl on 01-01-2025"
    
    config = ExtractionConfig(use_llm=False) # Pure deterministic pipeline
    extractor = FinEE(config)
    
    result = extractor.extract(text)
    
    assert result.is_complete()
    assert result.amount == 2500.0
    assert result.merchant == "Swiggy"
    assert result.category == Category.FOOD
    assert result.date == "01-01-2025"
