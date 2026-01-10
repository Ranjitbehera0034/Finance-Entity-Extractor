"""
Tests for FinEE Confidence Scoring (Tier 4).
"""

from finee.confidence import (
    calculate_confidence_score,
    calculate_completeness,
    update_result_confidence,
    Confidence
)
from finee.schema import ExtractionResult, ExtractionSource, FieldMeta

def test_confidence_calculation():
    """Test confidence scoring logic."""
    # High confidence result (Regex)
    result = ExtractionResult(amount=500.0, type="debit")
    result.meta['amount'] = FieldMeta(source=ExtractionSource.REGEX, confidence=0.95)
    result.meta['type'] = FieldMeta(source=ExtractionSource.REGEX, confidence=0.95)
    
    score = calculate_confidence_score(result)
    assert score > 0.9
    
    # Low confidence result (LLM only)
    result_llm = ExtractionResult(amount=500.0)
    result_llm.meta['amount'] = FieldMeta(source=ExtractionSource.LLM, confidence=0.70)
    
    score_llm = calculate_confidence_score(result_llm)
    assert score_llm < 0.8

def test_completeness_check():
    """Test completeness calculation."""
    # Full result (needs reference too for 1.0)
    full = ExtractionResult(
        amount=100.0, type="debit", 
        merchant="Swiggy", category="food", 
        date="01-01-2025", reference="123456"
    )
    score = calculate_completeness(full)
    assert score == 1.0
    
    # Partial result (missing merchant/category)
    partial = ExtractionResult(amount=100.0, type="debit")
    score_part = calculate_completeness(partial)
    assert score_part < 1.0
    
    # Missing critical fields
    missing = ExtractionResult(merchant="Swiggy")
    score_miss = calculate_completeness(missing)
    assert score_miss < 0.5

def test_confidence_update():
    """Test updating result confidence enum."""
    # High confidence (needs high score)
    high = ExtractionResult(
        amount=500.0, type="debit", 
        date="01-01-2025", account="1234",
        merchant="Swiggy", category="food",
        reference="123456", vpa="swiggy@ybl"
    )
    high.meta['amount'] = FieldMeta(source=ExtractionSource.REGEX, confidence=0.95)
    high.meta['type'] = FieldMeta(source=ExtractionSource.REGEX, confidence=0.95)
    high.meta['date'] = FieldMeta(source=ExtractionSource.REGEX, confidence=0.95)
    high.meta['account'] = FieldMeta(source=ExtractionSource.REGEX, confidence=0.95)
    high.meta['merchant'] = FieldMeta(source=ExtractionSource.REGEX, confidence=0.95)
    high.meta['category'] = FieldMeta(source=ExtractionSource.REGEX, confidence=0.95)
    high.meta['reference'] = FieldMeta(source=ExtractionSource.REGEX, confidence=0.95)
    high.meta['vpa'] = FieldMeta(source=ExtractionSource.REGEX, confidence=0.95)
    
    updated_high = update_result_confidence(high)
    assert updated_high.confidence == Confidence.HIGH
    
    # Low confidence
    low = ExtractionResult(amount=500.0) # Missing type
    low.meta['amount'] = FieldMeta(source=ExtractionSource.LLM, confidence=0.6)
    
    updated_low = update_result_confidence(low)
    assert updated_low.confidence != Confidence.HIGH
