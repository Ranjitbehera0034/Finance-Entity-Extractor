"""
Tests for FinEE Normalizer and Validator (Tier 4).
"""

import pytest
from finee.normalizer import (
    normalize_amount,
    normalize_date,
    normalize_account,
    normalize_vpa
)
from finee.validator import (
    validate_extraction_result,
    repair_llm_json,
    ExtractionResult,
    TransactionType,
    Category
)

def test_normalize_amount():
    cases = [
        ("Rs. 1,234.50", 1234.5),
        ("INR 500", 500.0),
        ("25000", 25000.0),
        ("₹ 100", 100.0),
        ("invalid", None),
    ]
    for input_val, expected in cases:
        assert normalize_amount(input_val) == expected

def test_normalize_date():
    cases = [
        ("28-12-2025", "28-12-2025"),
        ("28/12/25", "28-12-2025"),
        ("28 Dec 2025", "28-12-2025"),
        ("2025-12-28", "28-12-2025"),
        ("invalid_date", None),
    ]
    for input_val, expected in cases:
        assert normalize_date(input_val) == expected

def test_normalize_account():
    assert normalize_account("A/c 123456") == "123456"
    assert normalize_account("XXXX12345678", mask=True) == "****5678" # Needs enough digits
    assert normalize_account("12345678", mask=True) == "****5678"

def test_repair_llm_json():
    """Test fixing broken LLM JSON output."""
    broken_jsons = [
        # Missing quotes
        ('{amount: 500, type: "debit"}', {"amount": 500, "type": "debit"}),
        # Single quotes
        ("{'merchant': 'Swiggy'}", {"merchant": "Swiggy"}),
        # Trailing comma
        ('{"amount": 100,}', {"amount": 100}),
        # Wrapped in text
        ('Here is the JSON: {"amount": 500}', {"amount": 500}),
    ]
    
    for broken, expected in broken_jsons:
        repaired = repair_llm_json(broken)
        # Check subset equality
        for k, v in expected.items():
            assert repaired[k] == v

def test_validate_result_coercion():
    """Test coercion of dict to ExtractionResult."""
    data = {
        "amount": "Rs. 500",
        "type": "debited",
        "date": "28-12-2025",
        "category": "FOOD"
    }
    result = validate_extraction_result(data)
    
    assert result.amount == 500.0
    assert result.type == TransactionType.DEBIT
    assert result.category == Category.FOOD
