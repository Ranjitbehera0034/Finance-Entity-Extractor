"""
FinEE Schema - Core data structures for financial entity extraction.

This module defines the data classes used throughout the extraction pipeline.
All fields are optional to support partial extraction and additive merging.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import date
import json


class TransactionType(str, Enum):
    """Transaction type enumeration."""
    DEBIT = "debit"
    CREDIT = "credit"
    UNKNOWN = "unknown"


class Category(str, Enum):
    """Transaction category enumeration."""
    FOOD = "food"
    SHOPPING = "shopping"
    TRANSPORT = "transport"
    UTILITIES = "utilities"
    ENTERTAINMENT = "entertainment"
    TRANSFER = "transfer"
    SALARY = "salary"
    INVESTMENT = "investment"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    OTHER = "other"


class Confidence(str, Enum):
    """Extraction confidence levels."""
    HIGH = "high"        # All fields from regex/rules
    MEDIUM = "medium"    # Mix of regex + LLM
    LOW = "low"          # Mostly LLM or incomplete
    FAILED = "failed"    # Extraction failed


class ExtractionSource(str, Enum):
    """Source of each extracted field."""
    REGEX = "regex"
    RULES = "rules"
    LLM = "llm"
    CACHE = "cache"


@dataclass
class FieldMeta:
    """Metadata for a single extracted field."""
    source: ExtractionSource
    confidence: float  # 0.0 to 1.0
    raw_value: Optional[str] = None  # Original value before normalization


@dataclass
class ExtractionResult:
    """
    Complete extraction result with all financial entities.
    
    All fields are optional to support partial extraction.
    The `meta` dict tracks the source and confidence of each field.
    """
    # Core fields
    amount: Optional[float] = None
    type: Optional[TransactionType] = None
    date: Optional[str] = None  # Normalized to DD-MM-YYYY
    
    # Transaction details
    account: Optional[str] = None
    reference: Optional[str] = None
    vpa: Optional[str] = None
    
    # Enrichment fields
    merchant: Optional[str] = None
    category: Optional[Category] = None
    payment_method: Optional[str] = None
    bank: Optional[str] = None
    
    # Metadata
    confidence: Confidence = Confidence.LOW
    confidence_score: float = 0.0
    processing_time_ms: float = 0.0
    from_cache: bool = False
    
    # Field-level metadata
    meta: Dict[str, FieldMeta] = field(default_factory=dict)
    
    # Raw data
    raw_input: Optional[str] = None
    raw_llm_output: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values and meta."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None and k not in ('meta', 'raw_input', 'raw_llm_output'):
                if isinstance(v, Enum):
                    result[k] = v.value
                elif k == 'meta':
                    continue
                else:
                    result[k] = v
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def get_missing_fields(self, required: List[str] = None, desired: List[str] = None) -> List[str]:
        """Get list of missing fields."""
        if required is None:
            required = ['amount', 'type']
        if desired is None:
            desired = ['merchant', 'category', 'date', 'reference']
        
        missing = []
        for field_name in required + desired:
            if getattr(self, field_name, None) is None:
                missing.append(field_name)
        return missing
    
    def is_complete(self) -> bool:
        """Check if all required fields are present."""
        return self.amount is not None and self.type is not None
    
    def merge(self, other: 'ExtractionResult', overwrite: bool = False) -> 'ExtractionResult':
        """
        Merge another result into this one (additive).
        
        By default, existing values are NOT overwritten.
        Set overwrite=True to prefer `other`'s values.
        """
        for field_name in ['amount', 'type', 'date', 'account', 'reference', 
                           'vpa', 'merchant', 'category', 'payment_method', 'bank']:
            current_value = getattr(self, field_name)
            other_value = getattr(other, field_name)
            
            if other_value is not None:
                if current_value is None or overwrite:
                    setattr(self, field_name, other_value)
                    if field_name in other.meta:
                        self.meta[field_name] = other.meta[field_name]
        
        return self


@dataclass
class ExtractionConfig:
    """Configuration for the extraction pipeline."""
    # Cache settings
    cache_enabled: bool = True
    cache_max_size: int = 1000
    
    # LLM settings
    use_llm: bool = False  # Set to True to enable LLM (requires model download)
    llm_timeout_seconds: float = 10.0
    llm_max_tokens: int = 200
    llm_temperature: float = 0.1
    
    # Model settings
    model_path: Optional[str] = None
    model_id: str = "Ranjit0034/finance-entity-extractor"
    
    # Pipeline settings
    required_fields: List[str] = field(default_factory=lambda: ['amount', 'type'])
    desired_fields: List[str] = field(default_factory=lambda: ['merchant', 'category', 'date', 'reference'])
    
    # Confidence thresholds
    high_confidence_threshold: float = 0.9
    medium_confidence_threshold: float = 0.7


# Type aliases for clarity
RawText = str
JSONOutput = str
