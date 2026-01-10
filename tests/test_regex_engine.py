"""
Tests for FinEE Regex Engine (Tier 1).
"""

import pytest
from finee.regex_engine import get_regex_engine, TransactionType

@pytest.fixture
def engine():
    return get_regex_engine()

def test_amount_extraction(engine):
    """Test amount extraction patterns."""
    cases = [
        ("Rs.500.00 debited", 500.0),
        ("INR 2500 debited", 2500.0),
        ("Rs 1,234.50 spent", 1234.5),
        ("Amt: 50.00", 50.0),
        ("Amount: 100", 100.0),
        ("debited 200.00 from", 200.0),
    ]
    for text, expected in cases:
        result = engine.extract(text)
        assert result.amount == expected, f"Failed on: {text}"

def test_date_extraction(engine):
    """Test date extraction patterns."""
    cases = [
        ("on 01-01-2025", "01-01-2025"),
        ("on 01/01/25", "01/01/25"),
        ("on 28 Dec 2025", "28 Dec 2025"),
        ("Date: 15-08-2024", "15-08-2024"),
    ]
    for text, expected in cases:
        result = engine.extract(text)
        assert result.date == expected, f"Failed on: {text}"

def test_type_extraction(engine):
    """Test transaction type extraction."""
    debit_cases = [
        "debited from", "starting debit", "spent on", "paid to", "sent to"
    ]
    credit_cases = [
        "credited to", "received from", "refund of", "cashback received", "reversed"
    ]
    
    for text in debit_cases:
        assert engine.extract(text).type == TransactionType.DEBIT, f"Failed debit: {text}"
        
    for text in credit_cases:
        assert engine.extract(text).type == TransactionType.CREDIT, f"Failed credit: {text}"

def test_account_extraction(engine):
    """Test account number extraction."""
    cases = [
        ("A/c 1234", "1234"),
        ("Account XXXXX1234", "1234"),
        ("ending with 5678", "5678"),
        ("A/c no. 4321", "4321"),
    ]
    for text, expected in cases:
        result = engine.extract(text)
        assert result.account == expected, f"Failed on: {text}"

def test_reference_extraction(engine):
    """Test reference/UTR extraction."""
    cases = [
        ("Ref: 123456789012", "123456789012"),
        ("UTR: 098765432109", "098765432109"),
        ("UPI Ref 112233445566", "112233445566"),
    ]
    for text, expected in cases:
        result = engine.extract(text)
        assert result.reference == expected, f"Failed on: {text}"

def test_vpa_extraction(engine):
    """Test UPI VPA extraction."""
    cases = [
        ("to swiggy@ybl", "swiggy@ybl"),
        ("VPA: merchant@okaxis", "merchant@okaxis"),
        ("from user@sbi", "user@sbi"),
    ]
    for text, expected in cases:
        result = engine.extract(text)
        assert result.vpa == expected, f"Failed on: {text}"

def test_complex_message(engine):
    """Test a full banking message."""
    text = "Rs.2500.00 debited from HDFC A/c 3545 to VPA swiggy@ybl on 28-12-25. Ref: 534567891234"
    result = engine.extract(text)
    
    assert result.amount == 2500.0
    assert result.type == TransactionType.DEBIT
    assert result.account == "3545"
    assert result.vpa == "swiggy@ybl"
    assert result.date == "28-12-25"
    assert result.reference == "534567891234"
    assert result.bank == "HDFC"
