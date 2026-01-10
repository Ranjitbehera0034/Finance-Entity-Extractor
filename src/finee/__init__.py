"""
FinEE - Finance Entity Extractor

A production-ready library for extracting structured financial entities
from Indian banking messages (SMS, email, statements).

Example:
    >>> from finee import extract
    >>> result = extract("Rs.500 debited from A/c 1234 on 01-01-25")
    >>> print(result.amount)
    500.0
    >>> print(result.to_json())
    {"amount": 500.0, "type": "debit", "date": "01-01-2025", ...}
"""

__version__ = "1.0.0"
__author__ = "Ranjit Behera"

from .schema import (
    ExtractionResult,
    ExtractionConfig,
    TransactionType,
    Category,
    Confidence,
    ExtractionSource,
)

from .extractor import (
    FinEE,
    extract,
    get_extractor,
)

from .cache import (
    LRUCache,
    get_cache,
    clear_cache,
    get_cache_stats,
)

from .regex_engine import (
    RegexEngine,
    extract_with_regex,
)

from .merchants import (
    extract_merchant_from_vpa,
    get_category_from_merchant,
    get_merchant_and_category,
)

from .normalizer import (
    normalize_amount,
    normalize_date,
    normalize_account,
    normalize_reference,
)

from .validator import (
    repair_llm_json,
    validate_extraction_result,
)

from .confidence import (
    calculate_confidence_score,
    update_result_confidence,
)

from .backends import (
    get_available_backends,
    get_backend,
)

__all__ = [
    # Version
    "__version__",
    
    # Main API
    "extract",
    "FinEE",
    "get_extractor",
    
    # Data classes
    "ExtractionResult",
    "ExtractionConfig",
    "TransactionType",
    "Category",
    "Confidence",
    "ExtractionSource",
    
    # Cache
    "LRUCache",
    "get_cache",
    "clear_cache",
    "get_cache_stats",
    
    # Regex
    "RegexEngine",
    "extract_with_regex",
    
    # Merchants
    "extract_merchant_from_vpa",
    "get_category_from_merchant",
    "get_merchant_and_category",
    
    # Normalizer
    "normalize_amount",
    "normalize_date",
    "normalize_account",
    "normalize_reference",
    
    # Validator
    "repair_llm_json",
    "validate_extraction_result",
    
    # Confidence
    "calculate_confidence_score",
    "update_result_confidence",
    
    # Backends
    "get_available_backends",
    "get_backend",
]
