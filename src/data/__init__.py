"""
Data Module - Email Processing and Entity Extraction.

This module provides production-grade tools for processing financial emails,
extracting structured data, and classifying email content.

Components:
    - EmailParser: Parse emails from MBOX files
    - EntityExtractor: Extract financial entities from text
    - EmailClassifier: Classify emails into categories
    - PDFExtractor: Extract transactions from bank statement PDFs
    - BankEmailGenerator: Generate synthetic training data

Example:
    >>> from src.data import EntityExtractor, EmailClassifier
    >>> 
    >>> extractor = EntityExtractor()
    >>> result = extractor.extract("Rs.500 debited from account 1234")
    >>> print(result.to_dict())
    {'amount': '500', 'type': 'debit', 'account': '1234'}
    >>> 
    >>> classifier = EmailClassifier()
    >>> result = classifier.classify(subject="Transaction Alert", ...)
    >>> print(result.category)
    'finance'

Author: Ranjit Behera
License: MIT
"""

from __future__ import annotations

# Core exports
from src.data.extractor import (
    EntityExtractor,
    FinancialEntity,
    extract_entities,
)

from src.data.classifier import (
    EmailClassifier,
    EmailCategory,
    ClassificationResult,
    classify_email,
)

from src.data.parser import EmailParser

# Optional exports (may not be installed)
try:
    from src.data.pdf_extractor import PDFExtractor, Transaction
except ImportError:
    PDFExtractor = None  # type: ignore
    Transaction = None  # type: ignore

try:
    from src.data.bank_templates import BankEmailGenerator
except ImportError:
    BankEmailGenerator = None  # type: ignore

try:
    from src.data.labeling import LabelingPipeline, LabeledExample
except ImportError:
    LabelingPipeline = None  # type: ignore
    LabeledExample = None  # type: ignore

# Public API
__all__ = [
    # Core classes
    "EntityExtractor",
    "FinancialEntity",
    "EmailClassifier",
    "EmailCategory",
    "ClassificationResult",
    "EmailParser",
    
    # Convenience functions
    "extract_entities",
    "classify_email",
    
    # Optional classes
    "PDFExtractor",
    "Transaction",
    "BankEmailGenerator",
    "LabelingPipeline",
    "LabeledExample",
]


def get_version() -> str:
    """Get the package version."""
    from src import __version__
    return __version__
