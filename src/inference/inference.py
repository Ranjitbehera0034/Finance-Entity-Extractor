"""
Finance Entity Extractor - Professional Inference Module.

Provides structured API with JSON schema enforcement for
extracting financial entities from Indian banking emails.

Author: Ranjit Behera
License: MIT
Version: 0.8.0

Example:
    >>> from inference import FinanceExtractor
    >>> extractor = FinanceExtractor()
    >>> result = extractor.extract("Rs.2500.00 debited from account 3545...")
    >>> print(result.amount)  # "2500.00"
"""

import json
import re
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
from enum import Enum


class TransactionType(str, Enum):
    """Transaction type enumeration."""
    CREDIT = "credit"
    DEBIT = "debit"
    UNKNOWN = "unknown"


class ExtractionFormat(str, Enum):
    """Supported input formats."""
    EMAIL = "email"
    BANK_STATEMENT = "bank_statement"
    PHONEPE = "phonepe"
    GPAY = "gpay"
    PAYTM = "paytm"


@dataclass
class FinanceEntity:
    """
    Structured financial entity extracted from text.
    
    All fields are validated and typed. Missing fields are None.
    """
    amount: Optional[str] = None
    type: Optional[str] = None
    date: Optional[str] = None
    account: Optional[str] = None
    reference: Optional[str] = None
    merchant: Optional[str] = None
    category: Optional[str] = None
    bank: Optional[str] = None
    raw_response: Optional[str] = field(default=None, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values and internal fields."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None and k != 'raw_response':
                result[k] = v
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def is_valid(self) -> bool:
        """Check if extraction has minimum required fields."""
        return self.amount is not None and self.type is not None
    
    def __str__(self) -> str:
        return self.to_json()


def build_prompt(text: str, format_type: ExtractionFormat = ExtractionFormat.EMAIL) -> str:
    """
    Build a standardized prompt for the model.
    
    This is the official prompt format that the model was trained on.
    Do not modify this format - it will degrade extraction quality.
    
    Args:
        text: The input text (email body, statement row, etc.)
        format_type: The type of input format
        
    Returns:
        Formatted prompt string
    """
    # Format-specific prefixes (as used in training)
    prefixes = {
        ExtractionFormat.EMAIL: "",
        ExtractionFormat.BANK_STATEMENT: "[BANK_STATEMENT] ",
        ExtractionFormat.PHONEPE: "[PHONEPE] ",
        ExtractionFormat.GPAY: "[GPAY] ",
        ExtractionFormat.PAYTM: "[PAYTM] ",
    }
    
    prefix = prefixes.get(format_type, "")
    
    # Standard prompt format (trained on this exact format)
    prompt = f"""{prefix}Extract financial entities from this email:

{text}

Extract: amount, type, date, account, reference, merchant, category
Output JSON:"""
    
    return prompt


def parse_json_response(response: str) -> Dict[str, Any]:
    """
    Parse JSON from model response with fallback patterns.
    
    Handles various response formats:
    - Clean JSON: {"amount": "500"}
    - Markdown JSON: ```json {"amount": "500"} ```
    - Conversational: "Here is the data: {..."
    
    Args:
        response: Raw model output string
        
    Returns:
        Parsed dictionary or empty dict if parsing fails
    """
    # Try direct JSON parse first
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in response
    patterns = [
        r'\{[^{}]+\}',  # Simple object
        r'```json\s*(\{[^`]+\})\s*```',  # Markdown code block
        r'```\s*(\{[^`]+\})\s*```',  # Generic code block
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if match.lastindex else match.group(0)
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                continue
    
    return {}


def validate_entity(data: Dict[str, Any]) -> FinanceEntity:
    """
    Validate and normalize extracted entity data.
    
    Args:
        data: Raw parsed dictionary
        
    Returns:
        Validated FinanceEntity object
    """
    # Normalize type field
    txn_type = data.get('type', '').lower()
    if txn_type not in ('credit', 'debit'):
        txn_type = None
    
    # Normalize amount (remove commas, validate numeric)
    amount = data.get('amount', '')
    if amount:
        amount = str(amount).replace(',', '').strip()
        # Validate it's numeric
        try:
            float(amount.replace('.', '').replace('-', ''))
        except ValueError:
            amount = None
    else:
        amount = None
    
    return FinanceEntity(
        amount=amount,
        type=txn_type,
        date=data.get('date'),
        account=str(data.get('account', '')) if data.get('account') else None,
        reference=str(data.get('reference', '')) if data.get('reference') else None,
        merchant=data.get('merchant'),
        category=data.get('category'),
        bank=data.get('bank'),
    )


class FinanceExtractor:
    """
    High-level API for financial entity extraction.
    
    Provides a clean, validated interface for extracting
    financial data from Indian banking emails and statements.
    
    Example:
        >>> extractor = FinanceExtractor()
        >>> result = extractor.extract(
        ...     "Rs.2500.00 debited from account 3545 to VPA swiggy@ybl"
        ... )
        >>> print(result.amount)  # "2500.00"
        >>> print(result.to_json())
    """
    
    def __init__(self, model_path: str = None, adapter_path: str = None):
        """
        Initialize the extractor.
        
        Args:
            model_path: Path to base model (default: from HuggingFace)
            adapter_path: Path to LoRA adapters (default: from HuggingFace)
        """
        self.model_path = model_path
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load model on first use."""
        if self._model is not None:
            return
        
        try:
            from mlx_lm import load
        except ImportError:
            raise ImportError(
                "mlx_lm is required for MLX inference. "
                "Install with: pip install mlx-lm>=0.19.0"
            )
        
        if self.model_path and self.adapter_path:
            self._model, self._tokenizer = load(
                self.model_path,
                adapter_path=self.adapter_path
            )
        else:
            # Load from HuggingFace
            self._model, self._tokenizer = load(
                "Ranjit0034/finance-entity-extractor"
            )
    
    def extract(
        self,
        text: str,
        format_type: ExtractionFormat = ExtractionFormat.EMAIL,
        max_tokens: int = 200,
    ) -> FinanceEntity:
        """
        Extract financial entities from text.
        
        Args:
            text: Input text (email body, statement row, etc.)
            format_type: Type of input format
            max_tokens: Maximum tokens to generate
            
        Returns:
            FinanceEntity with extracted data
        """
        self._load_model()
        
        from mlx_lm import generate
        
        prompt = build_prompt(text, format_type)
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        
        # Parse and validate
        data = parse_json_response(response)
        entity = validate_entity(data)
        entity.raw_response = response
        
        return entity
    
    def extract_batch(
        self,
        texts: List[str],
        format_type: ExtractionFormat = ExtractionFormat.EMAIL,
    ) -> List[FinanceEntity]:
        """
        Extract entities from multiple texts.
        
        Args:
            texts: List of input texts
            format_type: Type of input format
            
        Returns:
            List of FinanceEntity objects
        """
        return [self.extract(text, format_type) for text in texts]


# Convenience function for simple usage
def extract(text: str, format_type: str = "email") -> Dict[str, Any]:
    """
    Simple extraction function.
    
    Args:
        text: Input text to extract from
        format_type: One of "email", "bank_statement", "phonepe", "gpay", "paytm"
        
    Returns:
        Dictionary with extracted entities
        
    Example:
        >>> from inference import extract
        >>> result = extract("Rs.500 debited from A/c 1234")
        >>> print(result["amount"])  # "500"
    """
    format_map = {
        "email": ExtractionFormat.EMAIL,
        "bank_statement": ExtractionFormat.BANK_STATEMENT,
        "phonepe": ExtractionFormat.PHONEPE,
        "gpay": ExtractionFormat.GPAY,
        "paytm": ExtractionFormat.PAYTM,
    }
    
    extractor = FinanceExtractor()
    fmt = format_map.get(format_type.lower(), ExtractionFormat.EMAIL)
    entity = extractor.extract(text, fmt)
    
    return entity.to_dict()


if __name__ == "__main__":
    # Demo usage
    demo_email = """
    HDFC BANK Dear Customer,
    Rs.2500.00 has been debited from account 3545 to VPA swiggy@ybl 
    SWIGGY INDIA on 28-12-25.
    Your UPI transaction reference number is 534567891234.
    """
    
    print("=" * 60)
    print("Finance Entity Extractor v0.8.0 - Demo")
    print("=" * 60)
    print(f"\nInput:\n{demo_email.strip()}")
    print("\nBuilding prompt...")
    prompt = build_prompt(demo_email)
    print(f"Prompt:\n{prompt[:200]}...")
    
    # Simulate response (for testing without model)
    mock_response = '''{"amount": "2500.00", "type": "debit", "date": "28-12-25", "account": "3545", "reference": "534567891234", "merchant": "swiggy", "category": "food"}'''
    
    print("\nParsing response...")
    data = parse_json_response(mock_response)
    entity = validate_entity(data)
    
    print(f"\nExtracted Entity:")
    print(entity.to_json())
    print(f"\nValid: {entity.is_valid()}")
