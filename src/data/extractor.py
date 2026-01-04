"""
Entity Extractor Module
Extracts financial entities from email text using regex patterns.
"""

import re
import json
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class FinancialEntity:
    """Represents extracted financial entities."""
    amount: Optional[str] = None
    type: Optional[str] = None  # 'debit' or 'credit'
    account: Optional[str] = None
    date: Optional[str] = None
    reference: Optional[str] = None
    merchant: Optional[str] = None
    payment_method: Optional[str] = None
    category: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def is_valid(self) -> bool:
        """Check if entity has minimum required fields."""
        return self.amount is not None and self.type is not None


class EntityExtractor:
    """Extract financial entities from email text."""
    
    # Regex patterns for entity extraction
    PATTERNS = {
        # Amount: Rs.1890.28, Rs 1,890.28, ₹1890, INR 500
        'amount': r'(?:Rs\.?|₹|INR)\s*([\d,]+(?:\.\d{1,2})?)',
        
        # Account formats: account 3545, A/C 1234, a/c XX0556, account **3545
        'account': r'(?:account|A/C|a/c)\s*[:\s]?\s*\*{0,2}(\d{3,})',
        
        # Date: DD-MM-YY, DD-MM-YYYY, DD/MM/YY
        'date': r'(\d{2}[-/]\d{2}[-/]\d{2,4})',
        
        # UPI Reference: reference number is XXXXX
        'reference': r'reference\s*(?:number|no\.?|#)?\s*(?:is)?\s*[:\s]?(\d{8,})',
        
        # VPA/UPI ID: xxx@ybl, xxx@okicici, etc.
        'vpa': r'\b([a-zA-Z0-9._-]+@[a-z]{2,})\b',
    }
    
    # Known merchants and their variants
    MERCHANTS = {
        'swiggy': ['swiggy', 'swiggy@'],
        'zomato': ['zomato', 'zomato@'],
        'amazon': ['amazon', 'amazon@', 'amzn'],
        'flipkart': ['flipkart', 'fkart'],
        'phonepe': ['phonepe'],
        'paytm': ['paytm'],
        'gpay': ['gpay', 'googlepay', 'google pay'],
        'uber': ['uber'],
        'ola': ['ola'],
        'myntra': ['myntra'],
        'bigbasket': ['bigbasket'],
        'zepto': ['zepto'],
        'blinkit': ['blinkit', 'grofers'],
    }
    
    # Payment method keywords
    PAYMENT_METHODS = {
        'upi': ['upi', 'vpa', '@ybl', '@okicici', '@paytm', '@axisbank'],
        'neft': ['neft'],
        'imps': ['imps'],
        'rtgs': ['rtgs'],
        'card': ['card', 'visa', 'mastercard', 'rupay', 'debit card', 'credit card'],
        'netbanking': ['netbanking', 'net banking', 'internet banking'],
    }
    
    # Transaction categories
    CATEGORIES = {
        'food': ['swiggy', 'zomato', 'food', 'restaurant', 'cafe', 'dining'],
        'shopping': ['amazon', 'flipkart', 'myntra', 'shopping', 'purchase'],
        'groceries': ['bigbasket', 'zepto', 'blinkit', 'grocery', 'grofers'],
        'transport': ['uber', 'ola', 'cab', 'taxi', 'metro', 'fuel', 'petrol'],
        'bills': ['electricity', 'water', 'gas', 'broadband', 'mobile', 'recharge'],
        'entertainment': ['netflix', 'spotify', 'prime', 'hotstar', 'movie'],
        'transfer': ['transfer', 'sent to', 'received from'],
    }
    
    def __init__(self):
        """Initialize extractor with compiled patterns."""
        self._compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PATTERNS.items()
        }
    
    def extract(self, text: str) -> FinancialEntity:
        """
        Extract financial entities from text.
        
        Args:
            text: Email body text
            
        Returns:
            FinancialEntity with extracted values
        """
        entity = FinancialEntity()
        text_lower = text.lower()
        
        # Extract amount
        amount_match = self._compiled_patterns['amount'].search(text)
        if amount_match:
            entity.amount = amount_match.group(1).replace(',', '')
        
        # Extract transaction type
        if 'debited' in text_lower:
            entity.type = 'debit'
        elif 'credited' in text_lower:
            entity.type = 'credit'
        
        # Extract account
        account_match = self._compiled_patterns['account'].search(text)
        if account_match:
            entity.account = account_match.group(1)
        
        # Extract date
        date_match = self._compiled_patterns['date'].search(text)
        if date_match:
            entity.date = date_match.group(1)
        
        # Extract reference
        ref_match = self._compiled_patterns['reference'].search(text)
        if ref_match:
            entity.reference = ref_match.group(1)
        
        # Extract merchant
        entity.merchant = self._extract_merchant(text_lower)
        
        # Extract payment method
        entity.payment_method = self._extract_payment_method(text_lower)
        
        # Determine category
        entity.category = self._determine_category(text_lower, entity.merchant)
        
        return entity
    
    def extract_to_dict(self, text: str) -> Dict:
        """Extract entities and return as dictionary."""
        return self.extract(text).to_dict()
    
    def extract_to_json(self, text: str, indent: int = 2) -> str:
        """Extract entities and return as JSON string."""
        return self.extract(text).to_json(indent)
    
    def _extract_merchant(self, text: str) -> Optional[str]:
        """Extract merchant name from text."""
        for merchant, keywords in self.MERCHANTS.items():
            for keyword in keywords:
                if keyword in text:
                    return merchant
        return None
    
    def _extract_payment_method(self, text: str) -> Optional[str]:
        """Extract payment method from text."""
        for method, keywords in self.PAYMENT_METHODS.items():
            for keyword in keywords:
                if keyword in text:
                    return method
        return None
    
    def _determine_category(self, text: str, merchant: Optional[str] = None) -> Optional[str]:
        """Determine transaction category."""
        # First check if merchant gives us a hint
        if merchant:
            for category, keywords in self.CATEGORIES.items():
                if merchant in keywords:
                    return category
        
        # Otherwise search text
        for category, keywords in self.CATEGORIES.items():
            for keyword in keywords:
                if keyword in text:
                    return category
        
        return None
    
    def batch_extract(self, texts: List[str]) -> List[FinancialEntity]:
        """Extract entities from multiple texts."""
        return [self.extract(text) for text in texts]


# Standalone function for backward compatibility
def extract_entities(text: str) -> Dict:
    """
    Extract financial entities from email text.
    
    This is a convenience function that creates an EntityExtractor
    and returns results as a dictionary.
    
    Args:
        text: Email body text
        
    Returns:
        Dictionary with extracted entities
    """
    extractor = EntityExtractor()
    return extractor.extract_to_dict(text)


if __name__ == "__main__":
    # Test extraction
    test_emails = [
        """
        HDFC BANK Dear Customer, Rs.2500.00 has been debited from account 3545 
        to VPA swiggy@ybl for Swiggy order on 28-12-25. 
        Your UPI transaction reference number is 534567891234.
        """,
        """
        Dear Customer, Rs.45,000.00 has been credited to your account 7890 
        on 27-12-25. Salary from ACME CORP. Reference number is 123456789.
        """,
    ]
    
    extractor = EntityExtractor()
    
    for i, email in enumerate(test_emails, 1):
        print(f"\n=== Email {i} ===")
        result = extractor.extract(email)
        print(result.to_json())
        print(f"Valid: {result.is_valid()}")
