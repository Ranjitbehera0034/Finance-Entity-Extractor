"""
Financial Entity Extractor Module.

This module provides enterprise-grade extraction of financial entities from
transaction emails across multiple Indian banks and payment platforms.

Supported Banks:
    - HDFC Bank
    - ICICI Bank
    - State Bank of India (SBI)
    - Axis Bank
    - Kotak Mahindra Bank

Supported Payment Platforms:
    - PhonePe
    - Google Pay (GPay)
    - Paytm
    - BHIM UPI

Example:
    >>> from src.data.extractor import EntityExtractor
    >>> extractor = EntityExtractor()
    >>> result = extractor.extract("Rs.2500 debited from A/c 3545 on 05-01-26")
    >>> print(result.to_dict())
    {'amount': '2500', 'type': 'debit', 'account': '3545', 'date': '05-01-26'}

Author: Ranjit Behera
License: MIT
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import (
    Dict,
    List,
    Optional,
    Pattern,
    Tuple,
    Any,
    ClassVar,
)

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class FinancialEntity:
    """
    Represents extracted financial entities from a transaction message.
    
    This dataclass holds all extracted fields from a financial transaction
    notification, including amount, type, account details, and optional
    merchant/category information.
    
    Attributes:
        amount: Transaction amount as string (preserves decimal precision).
        type: Transaction type - 'debit' or 'credit'.
        account: Account number (usually last 4 digits, masked).
        date: Transaction date in original format from message.
        reference: UPI/IMPS/NEFT reference number.
        merchant: Identified merchant name (e.g., 'swiggy', 'amazon').
        payment_method: Payment method - 'upi', 'neft', 'imps', 'card'.
        category: Transaction category - 'food', 'shopping', 'transport', etc.
        bank: Source bank name if identified.
        balance: Available balance after transaction.
        raw_text: Original text used for extraction (for debugging).
    
    Example:
        >>> entity = FinancialEntity(
        ...     amount="2500.00",
        ...     type="debit",
        ...     account="3545",
        ...     date="05-01-26",
        ...     merchant="swiggy",
        ...     category="food"
        ... )
        >>> entity.is_valid()
        True
        >>> entity.to_dict()
        {'amount': '2500.00', 'type': 'debit', ...}
    """
    
    amount: Optional[str] = None
    type: Optional[str] = None
    account: Optional[str] = None
    date: Optional[str] = None
    reference: Optional[str] = None
    merchant: Optional[str] = None
    payment_method: Optional[str] = None
    category: Optional[str] = None
    bank: Optional[str] = None
    balance: Optional[str] = None
    raw_text: str = field(default="", repr=False)
    
    # Validation constants
    VALID_TYPES: ClassVar[set] = {"debit", "credit"}
    VALID_PAYMENT_METHODS: ClassVar[set] = {"upi", "neft", "imps", "rtgs", "card", "wallet"}
    VALID_CATEGORIES: ClassVar[set] = {
        "food", "shopping", "transport", "bills", "grocery",
        "entertainment", "health", "education", "transfer", "other"
    }
    
    def __post_init__(self) -> None:
        """Validate and normalize fields after initialization."""
        # Normalize type to lowercase
        if self.type:
            self.type = self.type.lower()
        
        # Normalize payment method
        if self.payment_method:
            self.payment_method = self.payment_method.lower()
        
        # Normalize category
        if self.category:
            self.category = self.category.lower()
    
    def is_valid(self) -> bool:
        """
        Check if the entity has minimum required fields.
        
        A valid entity must have at least an amount and transaction type.
        
        Returns:
            bool: True if entity has minimum required fields.
        
        Example:
            >>> entity = FinancialEntity(amount="100", type="debit")
            >>> entity.is_valid()
            True
            >>> entity = FinancialEntity(amount="100")
            >>> entity.is_valid()
            False
        """
        return bool(self.amount and self.type)
    
    def is_complete(self) -> bool:
        """
        Check if the entity has all core fields populated.
        
        A complete entity has amount, type, account, and date.
        
        Returns:
            bool: True if entity has all core fields.
        """
        return bool(
            self.amount and 
            self.type and 
            self.account and 
            self.date
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entity to dictionary, excluding None values.
        
        Returns:
            Dict[str, Any]: Dictionary with non-None fields only.
        
        Example:
            >>> entity = FinancialEntity(amount="500", type="debit")
            >>> entity.to_dict()
            {'amount': '500', 'type': 'debit'}
        """
        return {
            key: value 
            for key, value in asdict(self).items() 
            if value is not None and key != "raw_text"
        }
    
    def to_json_string(self) -> str:
        """
        Convert entity to JSON string for model training.
        
        Returns:
            str: JSON representation of the entity.
        """
        import json
        return json.dumps(self.to_dict(), indent=2)
    
    def confidence_score(self) -> float:
        """
        Calculate confidence score based on populated fields.
        
        Returns:
            float: Score between 0.0 and 1.0.
        """
        required_fields = ["amount", "type"]
        optional_fields = ["account", "date", "reference", "merchant", "category"]
        
        required_score = sum(
            1 for f in required_fields 
            if getattr(self, f) is not None
        ) / len(required_fields)
        
        optional_score = sum(
            1 for f in optional_fields 
            if getattr(self, f) is not None
        ) / len(optional_fields)
        
        return required_score * 0.6 + optional_score * 0.4


class EntityExtractor:
    """
    Production-grade financial entity extractor using regex patterns.
    
    This extractor uses a comprehensive set of regex patterns to extract
    structured financial data from transaction notifications. It supports
    multiple Indian banks and payment platforms with high accuracy.
    
    Features:
        - Amount extraction (Rs., ₹, INR formats)
        - Transaction type detection (debit/credit)
        - Account number extraction (masked formats)
        - Date parsing (multiple formats)
        - Reference number extraction
        - Merchant identification
        - Payment method detection
        - Category classification
    
    Attributes:
        AMOUNT_PATTERNS: Compiled regex patterns for amount extraction.
        DEBIT_KEYWORDS: Keywords indicating debit transactions.
        CREDIT_KEYWORDS: Keywords indicating credit transactions.
        MERCHANTS: Merchant name to keyword mapping.
        CATEGORIES: Category to merchant mapping.
    
    Example:
        >>> extractor = EntityExtractor()
        >>> result = extractor.extract(
        ...     "HDFC Bank: Rs.2500.00 debited from A/c **3545 on 05-01-26"
        ... )
        >>> print(result.amount)
        '2500.00'
        >>> print(result.merchant)
        None  # Would need VPA for merchant detection
    
    Note:
        For best results, pass the complete transaction message including
        sender information and subject line when available.
    """
    
    # Amount extraction patterns (ordered by specificity)
    AMOUNT_PATTERNS: ClassVar[List[Pattern]] = [
        re.compile(r'(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{2})?)', re.IGNORECASE),
        re.compile(r'(?:amount|amt)[:\s]*([\d,]+(?:\.\d{2})?)', re.IGNORECASE),
        re.compile(r'(?:debited|credited)[:\s]*(?:Rs\.?|INR|₹)?\s*([\d,]+(?:\.\d{2})?)', re.IGNORECASE),
    ]
    
    # Transaction type keywords
    DEBIT_KEYWORDS: ClassVar[set] = {
        'debited', 'debit', 'paid', 'sent', 'withdrawn', 
        'purchase', 'payment', 'transferred out', 'dr'
    }
    
    CREDIT_KEYWORDS: ClassVar[set] = {
        'credited', 'credit', 'received', 'deposited', 
        'refund', 'cashback', 'transferred in', 'cr'
    }
    
    # Account extraction patterns
    ACCOUNT_PATTERNS: ClassVar[List[Pattern]] = [
        re.compile(r'(?:a/c|acct?|account)\s*(?:no\.?)?\s*[:\s]*\**[xX]*(\d{4,})', re.IGNORECASE),
        re.compile(r'[*xX]+(\d{4})\b'),
        re.compile(r'(?:XX|xx)(\d{4})\b'),
    ]
    
    # Date extraction patterns
    DATE_PATTERNS: ClassVar[List[Pattern]] = [
        re.compile(r'(\d{2}[-/]\d{2}[-/]\d{2,4})'),
        re.compile(r'(\d{2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})', re.IGNORECASE),
        re.compile(r'(\d{4}[-/]\d{2}[-/]\d{2})'),
    ]
    
    # Reference number patterns
    REFERENCE_PATTERNS: ClassVar[List[Pattern]] = [
        re.compile(r'(?:ref(?:erence)?|txn|upi|imps)\s*(?:no\.?|id)?[:\s]*(\d{10,})', re.IGNORECASE),
        re.compile(r'(?:transaction)\s*(?:id)?[:\s]*(\d{10,})', re.IGNORECASE),
    ]
    
    # Merchant identification
    MERCHANTS: ClassVar[Dict[str, List[str]]] = {
        # Food delivery
        'swiggy': ['swiggy', 'swiggy@'],
        'zomato': ['zomato', 'zomato@'],
        'dominos': ['dominos', 'domino'],
        'mcdonalds': ['mcdonald', 'mcd@'],
        'kfc': ['kfc@', 'kfc '],
        'starbucks': ['starbucks', 'sbux'],
        
        # E-commerce
        'amazon': ['amazon', 'amzn', 'amazon@'],
        'flipkart': ['flipkart', 'fkrt'],
        'myntra': ['myntra'],
        'ajio': ['ajio'],
        'nykaa': ['nykaa'],
        
        # Transport
        'uber': ['uber'],
        'ola': ['ola@', 'olacabs'],
        'rapido': ['rapido'],
        'metro': ['metro', 'dmrc', 'bmrc'],
        
        # Bills & Utilities
        'airtel': ['airtel'],
        'jio': ['jio@', 'reliancejio'],
        'vodafone': ['vodafone', 'vi@'],
        'electricity': ['bescom', 'electricity', 'power'],
        'gas': ['indane', 'bharat gas', 'hp gas'],
        
        # Grocery
        'bigbasket': ['bigbasket', 'bb@'],
        'zepto': ['zepto'],
        'blinkit': ['blinkit', 'grofers'],
        'dmart': ['dmart', 'd-mart'],
    }
    
    # Category mapping
    CATEGORY_KEYWORDS: ClassVar[Dict[str, List[str]]] = {
        'food': ['swiggy', 'zomato', 'dominos', 'mcdonald', 'kfc', 'restaurant', 'cafe', 'food'],
        'shopping': ['amazon', 'flipkart', 'myntra', 'ajio', 'nykaa', 'shopping'],
        'transport': ['uber', 'ola', 'rapido', 'metro', 'cab', 'taxi', 'fuel', 'petrol'],
        'bills': ['airtel', 'jio', 'vodafone', 'electricity', 'water', 'gas', 'broadband'],
        'grocery': ['bigbasket', 'zepto', 'blinkit', 'dmart', 'grocery', 'supermarket'],
        'entertainment': ['netflix', 'prime', 'hotstar', 'spotify', 'movie', 'theatre'],
        'health': ['pharmacy', 'medical', 'hospital', 'doctor', 'health'],
        'education': ['school', 'college', 'course', 'udemy', 'education'],
    }
    
    # Payment method patterns
    PAYMENT_PATTERNS: ClassVar[Dict[str, List[str]]] = {
        'upi': ['upi', 'vpa', '@ybl', '@oksbi', '@okicici', '@paytm', '@axisbank', '@icici'],
        'neft': ['neft'],
        'imps': ['imps'],
        'rtgs': ['rtgs'],
        'card': ['card', 'visa', 'mastercard', 'rupay', 'credit card', 'debit card'],
        'wallet': ['wallet', 'paytm wallet', 'phonepe wallet'],
    }
    
    # Bank identification
    BANK_PATTERNS: ClassVar[Dict[str, List[str]]] = {
        'hdfc': ['hdfc', 'hdfcbank'],
        'icici': ['icici'],
        'sbi': ['sbi', 'state bank'],
        'axis': ['axis'],
        'kotak': ['kotak'],
        'pnb': ['pnb', 'punjab national'],
        'bob': ['bob', 'bank of baroda'],
        'canara': ['canara'],
        'union': ['union bank'],
        'idbi': ['idbi'],
    }
    
    def __init__(self, debug: bool = False) -> None:
        """
        Initialize the EntityExtractor.
        
        Args:
            debug: If True, enables debug logging.
        
        Example:
            >>> extractor = EntityExtractor(debug=True)
        """
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)
        
        logger.info("EntityExtractor initialized")
    
    def extract(self, text: str) -> FinancialEntity:
        """
        Extract financial entities from transaction text.
        
        This is the main entry point for entity extraction. It processes
        the input text and returns a FinancialEntity object with all
        detected fields populated.
        
        Args:
            text: The transaction message text to process.
        
        Returns:
            FinancialEntity: Extracted entity with populated fields.
        
        Example:
            >>> extractor = EntityExtractor()
            >>> result = extractor.extract(
            ...     "Rs.2500 debited from A/c 3545 to swiggy@ybl on 05-01-26"
            ... )
            >>> print(result.amount)
            '2500'
            >>> print(result.merchant)
            'swiggy'
        
        Note:
            The method never raises exceptions. On failure, it returns
            an entity with None fields and logs the error.
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided")
            return FinancialEntity(raw_text=str(text) if text else "")
        
        text_lower = text.lower()
        
        try:
            entity = FinancialEntity(
                amount=self._extract_amount(text),
                type=self._extract_type(text_lower),
                account=self._extract_account(text),
                date=self._extract_date(text),
                reference=self._extract_reference(text),
                merchant=self._extract_merchant(text_lower),
                payment_method=self._extract_payment_method(text_lower),
                category=self._extract_category(text_lower),
                bank=self._extract_bank(text_lower),
                balance=self._extract_balance(text),
                raw_text=text,
            )
            
            logger.debug(f"Extracted entity: {entity.to_dict()}")
            return entity
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            return FinancialEntity(raw_text=text)
    
    def extract_to_dict(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and return as dictionary.
        
        Convenience method that extracts and converts to dict in one call.
        
        Args:
            text: The transaction message text.
        
        Returns:
            Dict[str, Any]: Dictionary of extracted entities.
        """
        return self.extract(text).to_dict()
    
    def _extract_amount(self, text: str) -> Optional[str]:
        """Extract transaction amount from text."""
        for pattern in self.AMOUNT_PATTERNS:
            match = pattern.search(text)
            if match:
                amount = match.group(1).replace(',', '')
                logger.debug(f"Found amount: {amount}")
                return amount
        return None
    
    def _extract_type(self, text_lower: str) -> Optional[str]:
        """Determine if transaction is debit or credit."""
        # Check debit keywords first (more common)
        for keyword in self.DEBIT_KEYWORDS:
            if keyword in text_lower:
                return 'debit'
        
        # Then check credit keywords
        for keyword in self.CREDIT_KEYWORDS:
            if keyword in text_lower:
                return 'credit'
        
        return None
    
    def _extract_account(self, text: str) -> Optional[str]:
        """Extract account number from text."""
        for pattern in self.ACCOUNT_PATTERNS:
            match = pattern.search(text)
            if match:
                account = match.group(1)
                # Return last 4 digits only
                return account[-4:] if len(account) > 4 else account
        return None
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract transaction date from text."""
        for pattern in self.DATE_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None
    
    def _extract_reference(self, text: str) -> Optional[str]:
        """Extract reference/transaction number."""
        for pattern in self.REFERENCE_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None
    
    def _extract_merchant(self, text_lower: str) -> Optional[str]:
        """Identify merchant from text."""
        for merchant, keywords in self.MERCHANTS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return merchant
        return None
    
    def _extract_payment_method(self, text_lower: str) -> Optional[str]:
        """Detect payment method."""
        for method, keywords in self.PAYMENT_PATTERNS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return method
        return None
    
    def _extract_category(self, text_lower: str) -> Optional[str]:
        """Classify transaction category."""
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category
        return None
    
    def _extract_bank(self, text_lower: str) -> Optional[str]:
        """Identify source bank."""
        for bank, keywords in self.BANK_PATTERNS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return bank
        return None
    
    def _extract_balance(self, text: str) -> Optional[str]:
        """Extract available balance if mentioned."""
        patterns = [
            re.compile(r'(?:bal(?:ance)?|avl\.?\s*bal)[:\s]*(?:Rs\.?|INR|₹)?\s*([\d,]+(?:\.\d{2})?)', re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).replace(',', '')
        return None


# Module-level convenience function
def extract_entities(text: str) -> Dict[str, Any]:
    """
    Convenience function to extract entities without instantiating class.
    
    Args:
        text: Transaction message text.
    
    Returns:
        Dict[str, Any]: Extracted entities as dictionary.
    
    Example:
        >>> from src.data.extractor import extract_entities
        >>> result = extract_entities("Rs.500 debited from account 1234")
        >>> print(result['amount'])
        '500'
    """
    return EntityExtractor().extract_to_dict(text)


if __name__ == "__main__":
    # Self-test when run directly
    logging.basicConfig(level=logging.DEBUG)
    
    extractor = EntityExtractor(debug=True)
    
    test_cases = [
        "HDFC Bank: Rs.2500.00 debited from A/c **3545 on 05-01-26 to VPA swiggy@ybl. Ref: 123456789012",
        "Dear Customer, INR 45000 credited to A/c 7890 on 04-01-2026. Salary from ACME Corp.",
        "You paid Rs.599 to Amazon from HDFC Bank a/c XX4567. UPI Ref: 987654321012",
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test[:50]}...")
        result = extractor.extract(test)
        print(f"Result: {result.to_dict()}")
        print(f"Valid: {result.is_valid()}, Confidence: {result.confidence_score():.2%}")
