"""
FinEE Regex Engine - Tier 1 pattern-based extraction.

High-performance regex patterns for extracting financial entities from
Indian banking messages. Covers HDFC, ICICI, SBI, Axis, Kotak and
payment apps (PhonePe, GPay, Paytm).
"""

import re
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

from .schema import ExtractionResult, TransactionType, ExtractionSource, FieldMeta


@dataclass
class RegexPattern:
    """A compiled regex pattern with metadata."""
    name: str
    pattern: re.Pattern
    field: str
    priority: int = 0  # Higher = preferred
    extractor: callable = None  # Optional post-processing


class RegexEngine:
    """
    Tier 1 extraction engine using regex patterns.
    
    Extracts: amount, date, reference, account, vpa, type
    Does NOT extract: merchant, category (handled by Tier 2/3)
    """
    
    def __init__(self):
        """Initialize regex patterns."""
        self._patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[RegexPattern]]:
        """Compile all regex patterns organized by field."""
        
        patterns = {
            'amount': [
                # Lakhs notation: 1.5 Lakh, 2 lacs, etc.
                RegexPattern(
                    'amount_lakhs',
                    re.compile(r'([\d.]+)\s*(?:lakh|lac|L)s?\b', re.IGNORECASE),
                    'amount',
                    priority=15,
                    extractor=lambda m: str(float(m.group(1)) * 100000)
                ),
                # Rs.2500.00 or Rs 2500 or INR 2,500.00 or ₹2,500
                RegexPattern(
                    'amount_rs',
                    re.compile(r'(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{1,2})?)', re.IGNORECASE),
                    'amount',
                    priority=10
                ),
                # 2500.00 debited/credited (amount before action, even without space)
                RegexPattern(
                    'amount_action_before',
                    re.compile(r'([\d,]+(?:\.\d{1,2})?)\s*(?:has been\s+)?(?:debited|credited|transferred)', re.IGNORECASE),
                    'amount',
                    priority=5
                ),
                # debited/credited 2500.00 (action before amount)
                RegexPattern(
                    'amount_action_after',
                    re.compile(r'(?:debited|credited|transferred|spent)\s+(?:Rs\.?|INR|₹)?\s*([\d,]+(?:\.\d{1,2})?)', re.IGNORECASE),
                    'amount',
                    priority=5
                ),
                # Amt: 2500 or Amount: 2500
                RegexPattern(
                    'amount_label',
                    re.compile(r'(?:Amt|Amount)[:\s]*([\d,]+(?:\.\d{1,2})?)', re.IGNORECASE),
                    'amount',
                    priority=8
                ),
            ],
            
            'type': [
                # Explicit debit/credit
                RegexPattern(
                    'type_explicit',
                    re.compile(r'\b(debited|debit|withdrawn|sent|paid|spent)\b', re.IGNORECASE),
                    'type',
                    priority=10,
                    extractor=lambda m: TransactionType.DEBIT
                ),
                RegexPattern(
                    'type_credit',
                    re.compile(r'\b(credited|credit|received|refund|cashback|reversed)\b', re.IGNORECASE),
                    'type',
                    priority=10,
                    extractor=lambda m: TransactionType.CREDIT
                ),
            ],
            
            'date': [
                # DD-MM-YY or DD-MM-YYYY
                RegexPattern(
                    'date_dmy',
                    re.compile(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b'),
                    'date',
                    priority=10
                ),
                # DD Mon YYYY (28 Dec 2025)
                RegexPattern(
                    'date_text',
                    re.compile(r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b', re.IGNORECASE),
                    'date',
                    priority=8
                ),
                # on DD/MM/YYYY at HH:MM
                RegexPattern(
                    'date_on',
                    re.compile(r'on\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', re.IGNORECASE),
                    'date',
                    priority=12
                ),
            ],
            
            'reference': [
                # UPI reference (12-16 digits)
                RegexPattern(
                    'ref_upi',
                    re.compile(r'(?:Ref(?:erence)?|UTR|UPI\s*Ref)[:\s#]*(\d{12,16})', re.IGNORECASE),
                    'reference',
                    priority=10
                ),
                # Transaction ID
                RegexPattern(
                    'ref_txn',
                    re.compile(r'(?:Txn|Transaction)\s*(?:ID|No|#)?[:\s]*([A-Z0-9]{10,20})', re.IGNORECASE),
                    'reference',
                    priority=8
                ),
                # Standalone 12-digit number (likely UPI ref)
                RegexPattern(
                    'ref_standalone',
                    re.compile(r'\b(\d{12})\b'),
                    'reference',
                    priority=3  # Low priority, might be phone number
                ),
            ],
            
            'account': [
                # A/c XX1234 or Account 1234 or XXXXX1234
                RegexPattern(
                    'account_ac',
                    re.compile(r'(?:A/c|Acct?|Account)(?:\s*(?:no\.?|number))?[:\s]*(?:[*X]{2,})?(\d{4,})', re.IGNORECASE),
                    'account',
                    priority=10
                ),
                # from XXXX1234
                RegexPattern(
                    'account_from',
                    re.compile(r'from\s+(?:[*X]{2,})?(\d{4,})', re.IGNORECASE),
                    'account',
                    priority=8
                ),
                # ending with 1234
                RegexPattern(
                    'account_ending',
                    re.compile(r'ending\s+(?:with\s+)?(\d{4})', re.IGNORECASE),
                    'account',
                    priority=6
                ),
            ],
            
            'vpa': [
                # UPI VPA (user@bank)
                RegexPattern(
                    'vpa_upi',
                    re.compile(r'(?:VPA|to|from)\s+([a-zA-Z0-9._-]+@[a-zA-Z0-9]+)', re.IGNORECASE),
                    'vpa',
                    priority=10
                ),
                # Standalone VPA pattern
                RegexPattern(
                    'vpa_standalone',
                    re.compile(r'\b([a-zA-Z0-9._-]+@(?:ybl|paytm|okaxis|oksbi|okhdfcbank|axl|ibl|upi|apl|fbl|icici|hdfcbank|sbi))\b', re.IGNORECASE),
                    'vpa',
                    priority=8
                ),
            ],
            
            'bank': [
                # Bank names
                RegexPattern(
                    'bank_name',
                    re.compile(r'\b(HDFC|ICICI|SBI|Axis|Kotak|PNB|BOB|IDFC|Yes Bank|IndusInd|RBL|Federal)\b', re.IGNORECASE),
                    'bank',
                    priority=10
                ),
            ],
            
            'payment_method': [
                # Payment methods
                RegexPattern(
                    'method_upi',
                    re.compile(r'\b(UPI|IMPS|NEFT|RTGS|NACH)\b', re.IGNORECASE),
                    'payment_method',
                    priority=10
                ),
                # Card
                RegexPattern(
                    'method_card',
                    re.compile(r'\b(Debit Card|Credit Card|Card)\b', re.IGNORECASE),
                    'payment_method',
                    priority=8
                ),
            ],
        }
        
        return patterns
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extract all possible fields from text using regex.
        
        Args:
            text: Input text (bank SMS, email, etc.)
            
        Returns:
            ExtractionResult with extracted fields
        """
        result = ExtractionResult(raw_input=text)
        
        for field_name, patterns in self._patterns.items():
            value = self._extract_field(text, patterns)
            if value is not None:
                # Handle amount parsing
                if field_name == 'amount':
                    try:
                        # Remove commas and parse as float
                        value = float(value.replace(',', ''))
                    except (ValueError, AttributeError):
                        continue
                
                setattr(result, field_name, value)
                result.meta[field_name] = FieldMeta(
                    source=ExtractionSource.REGEX,
                    confidence=0.95,
                    raw_value=str(value)
                )
        
        return result
    
    def _extract_field(self, text: str, patterns: List[RegexPattern]) -> Optional[Any]:
        """
        Extract a single field using multiple patterns.
        
        Returns the first match from the highest priority pattern.
        """
        # Sort by priority (highest first)
        sorted_patterns = sorted(patterns, key=lambda p: p.priority, reverse=True)
        
        for pattern in sorted_patterns:
            match = pattern.pattern.search(text)
            if match:
                if pattern.extractor:
                    return pattern.extractor(match)
                else:
                    return match.group(1)
        
        return None
    
    def extract_all_matches(self, text: str, field: str) -> List[Tuple[str, int]]:
        """
        Extract all matches for a specific field.
        
        Returns list of (value, priority) tuples.
        """
        if field not in self._patterns:
            return []
        
        matches = []
        for pattern in self._patterns[field]:
            for match in pattern.pattern.finditer(text):
                value = match.group(1) if match.lastindex else match.group(0)
                if pattern.extractor:
                    value = pattern.extractor(match)
                matches.append((value, pattern.priority))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)


# Module-level singleton
_engine: Optional[RegexEngine] = None


def get_regex_engine() -> RegexEngine:
    """Get or create the global regex engine instance."""
    global _engine
    if _engine is None:
        _engine = RegexEngine()
    return _engine


def extract_with_regex(text: str) -> ExtractionResult:
    """Convenience function for extraction."""
    return get_regex_engine().extract(text)
