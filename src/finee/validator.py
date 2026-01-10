"""
FinEE Validator - JSON repair and validation.

Handles:
- Broken JSON repair (using json-repair)
- Schema validation
- Field type coercion
"""

import json
from typing import Dict, Any, Optional, List
import re

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False


from .schema import ExtractionResult, TransactionType, Category


def repair_llm_json(raw_output: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to repair and parse LLM JSON output.
    
    Handles common issues:
    - Missing quotes
    - Trailing commas
    - Single quotes instead of double
    - Incomplete JSON
    
    Args:
        raw_output: Raw LLM output string
        
    Returns:
        Parsed dictionary or None if repair fails
    """
    if not raw_output:
        return None
    
    # Try to extract JSON from the output
    json_str = extract_json_from_text(raw_output)
    
    if not json_str:
        return None
    
    # First, try direct parsing
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Use json-repair if available
    if HAS_JSON_REPAIR:
        try:
            repaired = repair_json(json_str)
            return json.loads(repaired)
        except (json.JSONDecodeError, Exception):
            pass
    
    # Manual repair attempts
    repaired = manual_json_repair(json_str)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON object from text that may contain other content.
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Extracted JSON string or None
    """
    if not text:
        return None
    
    # Look for JSON object pattern
    # Find first { and last }
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    
    return None


def manual_json_repair(json_str: str) -> str:
    """
    Manually repair common JSON issues.
    
    Args:
        json_str: Potentially broken JSON string
        
    Returns:
        Repaired JSON string
    """
    if not json_str:
        return json_str
    
    repaired = json_str
    
    # Replace single quotes with double quotes
    repaired = re.sub(r"'([^']*)':", r'"\1":', repaired)
    repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)
    
    # Remove trailing commas
    repaired = re.sub(r',\s*}', '}', repaired)
    repaired = re.sub(r',\s*]', ']', repaired)
    
    # Add missing quotes around unquoted keys
    repaired = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)
    
    # Handle Python-style None/True/False
    repaired = repaired.replace(': None', ': null')
    repaired = repaired.replace(':None', ':null')
    repaired = repaired.replace(': True', ': true')
    repaired = repaired.replace(':True', ':true')
    repaired = repaired.replace(': False', ': false')
    repaired = repaired.replace(':False', ':false')
    
    return repaired


def validate_extraction_result(data: Dict[str, Any]) -> ExtractionResult:
    """
    Validate and coerce a dictionary into an ExtractionResult.
    
    Args:
        data: Dictionary from parsed JSON
        
    Returns:
        Validated ExtractionResult
    """
    result = ExtractionResult()
    
    # Amount
    if 'amount' in data:
        amount = data['amount']
        if isinstance(amount, (int, float)):
            result.amount = float(amount)
        elif isinstance(amount, str):
            try:
                # Remove currency symbols
                cleaned = re.sub(r'[Rs\.₹,\s]', '', amount)
                result.amount = float(cleaned)
            except ValueError:
                pass
    
    # Type
    if 'type' in data:
        type_val = str(data['type']).lower()
        if 'debit' in type_val:
            result.type = TransactionType.DEBIT
        elif 'credit' in type_val:
            result.type = TransactionType.CREDIT
    
    # Date (keep as string)
    if 'date' in data:
        result.date = str(data['date'])
    
    # Simple string fields
    for field in ['account', 'reference', 'vpa', 'merchant', 'payment_method', 'bank']:
        if field in data and data[field]:
            setattr(result, field, str(data[field]))
    
    # Category
    if 'category' in data:
        cat_val = str(data['category']).lower()
        try:
            result.category = Category(cat_val)
        except ValueError:
            # Map common variations
            category_map = {
                'food': Category.FOOD,
                'dining': Category.FOOD,
                'restaurant': Category.FOOD,
                'grocery': Category.FOOD,
                'shop': Category.SHOPPING,
                'shopping': Category.SHOPPING,
                'retail': Category.SHOPPING,
                'travel': Category.TRANSPORT,
                'transport': Category.TRANSPORT,
                'cab': Category.TRANSPORT,
                'utility': Category.UTILITIES,
                'utilities': Category.UTILITIES,
                'bill': Category.UTILITIES,
                'entertainment': Category.ENTERTAINMENT,
                'movie': Category.ENTERTAINMENT,
                'transfer': Category.TRANSFER,
                'payment': Category.TRANSFER,
            }
            result.category = category_map.get(cat_val, Category.OTHER)
    
    return result


def is_valid_amount(amount: Optional[float]) -> bool:
    """Check if amount is valid."""
    if amount is None:
        return False
    return isinstance(amount, (int, float)) and amount > 0


def is_valid_date(date_str: Optional[str]) -> bool:
    """Check if date string is valid."""
    if not date_str:
        return False
    
    # Basic format check (DD-MM-YYYY)
    pattern = r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$'
    return bool(re.match(pattern, date_str))


def is_valid_reference(ref: Optional[str]) -> bool:
    """Check if reference number is valid."""
    if not ref:
        return False
    
    # Should be 10+ alphanumeric characters
    cleaned = re.sub(r'\W', '', ref)
    return len(cleaned) >= 10


def is_valid_vpa(vpa: Optional[str]) -> bool:
    """Check if VPA is valid."""
    if not vpa:
        return False
    
    # Basic VPA format: user@bank
    pattern = r'^[a-zA-Z0-9._-]+@[a-zA-Z0-9]+$'
    return bool(re.match(pattern, vpa.lower()))
