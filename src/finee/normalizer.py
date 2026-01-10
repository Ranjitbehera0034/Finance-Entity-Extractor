"""
FinEE Normalizer - Data normalization utilities.

Handles normalization of:
- Amounts (₹2,500.00 → 2500.0)
- Dates (various formats → DD-MM-YYYY)
- Account numbers (masking, formatting)
- Reference numbers (padding)
"""

import re
from datetime import datetime, date
from typing import Optional, Union
from dateutil import parser as date_parser


def normalize_amount(amount_str: Union[str, float, int, None]) -> Optional[float]:
    """
    Normalize amount string to float.
    
    Handles:
    - Currency symbols (Rs., ₹, INR)
    - Commas (2,500.00)
    - Spaces (Rs. 2 500)
    
    Args:
        amount_str: Amount in various formats
        
    Returns:
        Float amount or None if parsing fails
    """
    if amount_str is None:
        return None
    
    if isinstance(amount_str, (int, float)):
        return float(amount_str)
    
    if not isinstance(amount_str, str):
        return None
    
    # Remove currency symbols (specific prefixes)
    cleaned = amount_str.strip()
    cleaned = re.sub(r'^(?:Rs\.?|INR|₹)\s*', '', cleaned, flags=re.IGNORECASE)
    
    # Remove commas
    cleaned = cleaned.replace(',', '')
    
    # Handle Indian lakhs/crores notation (if present)
    cleaned = cleaned.replace(' ', '')
    
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_date(date_str: Optional[str], output_format: str = '%d-%m-%Y') -> Optional[str]:
    """
    Normalize date string to standard format.
    
    Handles:
    - DD-MM-YY, DD-MM-YYYY
    - DD/MM/YY, DD/MM/YYYY
    - DD Mon YYYY (28 Dec 2025)
    - YYYY-MM-DD (ISO format)
    
    Args:
        date_str: Date in various formats
        output_format: Output format (default: DD-MM-YYYY)
        
    Returns:
        Normalized date string or None if parsing fails
    """
    if not date_str:
        return None
    
    # Clean input
    date_str = date_str.strip()
    
    # Common Indian date formats to try
    formats = [
        '%d-%m-%Y',      # 28-12-2025
        '%d-%m-%y',      # 28-12-25
        '%d/%m/%Y',      # 28/12/2025
        '%d/%m/%y',      # 28/12/25
        '%d %b %Y',      # 28 Dec 2025
        '%d %b %y',      # 28 Dec 25
        '%d %B %Y',      # 28 December 2025
        '%d %B %y',      # 28 December 25
        '%Y-%m-%d',      # 2025-12-28 (ISO)
        '%d.%m.%Y',      # 28.12.2025
        '%d.%m.%y',      # 28.12.25
    ]
    
    # Try each format
    for fmt in formats:
        try:
            parsed = datetime.strptime(date_str, fmt)
            
            # Handle 2-digit years (assume 20xx for years < 50)
            if parsed.year < 100:
                if parsed.year < 50:
                    parsed = parsed.replace(year=parsed.year + 2000)
                else:
                    parsed = parsed.replace(year=parsed.year + 1900)
            
            return parsed.strftime(output_format)
        except ValueError:
            continue
    
    # Fallback to dateutil parser
    try:
        parsed = date_parser.parse(date_str, dayfirst=True)
        return parsed.strftime(output_format)
    except (ValueError, TypeError):
        return None


def normalize_account(account_str: Optional[str], mask: bool = False) -> Optional[str]:
    """
    Normalize account number.
    
    Args:
        account_str: Account number string
        mask: If True, mask all but last 4 digits
        
    Returns:
        Normalized account number
    """
    if not account_str:
        return None
    
    # Extract digits only
    digits = re.sub(r'\D', '', str(account_str))
    
    if not digits:
        return None
    
    if mask and len(digits) > 4:
        return '*' * (len(digits) - 4) + digits[-4:]
    
    return digits


def normalize_reference(ref_str: Optional[str]) -> Optional[str]:
    """
    Normalize transaction reference number.
    
    Args:
        ref_str: Reference number string
        
    Returns:
        Normalized reference number
    """
    if not ref_str:
        return None
    
    # Extract alphanumeric characters
    cleaned = re.sub(r'[^A-Za-z0-9]', '', str(ref_str))
    
    return cleaned if cleaned else None


def normalize_vpa(vpa_str: Optional[str]) -> Optional[str]:
    """
    Normalize UPI VPA.
    
    Args:
        vpa_str: VPA string
        
    Returns:
        Lowercase VPA
    """
    if not vpa_str:
        return None
    
    # Remove extra whitespace and lowercase
    cleaned = vpa_str.strip().lower()
    
    # Validate VPA format (should have @)
    if '@' not in cleaned:
        return None
    
    return cleaned


def normalize_merchant(merchant_str: Optional[str]) -> Optional[str]:
    """
    Normalize merchant name.
    
    Args:
        merchant_str: Merchant name string
        
    Returns:
        Cleaned merchant name
    """
    if not merchant_str:
        return None
    
    # Title case and clean
    cleaned = merchant_str.strip()
    
    # Remove common prefixes/suffixes
    prefixes = ['payment to', 'paid to', 'transfer to', 'upi-']
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    
    return cleaned if cleaned else None


def normalize_type(type_str: Optional[str]) -> Optional[str]:
    """
    Normalize transaction type.
    
    Args:
        type_str: Type string (debit/credit variants)
        
    Returns:
        'debit' or 'credit'
    """
    if not type_str:
        return None
    
    type_lower = str(type_str).lower().strip()
    
    debit_keywords = ['debit', 'debited', 'withdrawn', 'sent', 'paid', 'spent', 'purchase']
    credit_keywords = ['credit', 'credited', 'received', 'refund', 'cashback', 'reversed']
    
    for kw in debit_keywords:
        if kw in type_lower:
            return 'debit'
    
    for kw in credit_keywords:
        if kw in type_lower:
            return 'credit'
    
    return None
