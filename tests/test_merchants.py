"""
Tests for FinEE Merchants and Rules (Tier 2).
"""

from finee.merchants import (
    extract_merchant_from_vpa,
    get_category_from_merchant,
    get_merchant_and_category,
    get_category_from_text,
)
from finee.schema import Category

def test_vpa_to_merchant():
    """Test extracting merchant names from VPAs."""
    cases = [
        ("swiggy@ybl", "Swiggy"),
        ("zomato@paytm", "Zomato"),
        ("uber@okaxis", "Uber"),
        ("AMAZONPAY@ICICI", "Amazon"),
        ("zerodhabroking@kbl", "Zerodha"),
        ("unknown@ybl", None),
    ]
    for vpa, expected in cases:
        assert extract_merchant_from_vpa(vpa) == expected

def test_merchant_to_category():
    """Test mapping merchants to categories."""
    cases = [
        ("Swiggy", "food"),
        ("Uber", "transport"),
        ("Netflix", "entertainment"),
        ("Zerodha", "investment"),
        ("Apollo Pharmacy", "healthcare"),
        ("Unknown Shop", None),
    ]
    for merchant, expected in cases:
        assert get_category_from_merchant(merchant) == expected

def test_text_category_inference():
    """Test inferring category from text keywords."""
    cases = [
        ("Lunch at restaurant", "food"),
        ("Taxi ride to airport", "transport"),
        ("Monthly movie subscription", "entertainment"),
        ("Electricity bill payment", "utilities"),
        ("Unknown transaction", None),
    ]
    for text, expected in cases:
        assert get_category_from_text(text) == expected

def test_combined_lookup():
    """Test the combined lookup function."""
    merchant, category = get_merchant_and_category(vpa="swiggy@ybl")
    assert merchant == "Swiggy"
    assert category == "food"

    # Fallback to text for category
    merchant, category = get_merchant_and_category(vpa="unknown@ybl", text="payment for petrol")
    assert merchant is None
    assert category == "transport"
