"""
LLM Mail Trainer - Finance Entity Extraction Package.

A production-grade system for extracting financial entities from emails
using fine-tuned LLMs on Apple Silicon (MLX).

Features:
    - Multi-bank email parsing (HDFC, ICICI, SBI, Axis, Kotak)
    - UPI/NEFT/IMPS transaction detection
    - Merchant and category classification
    - REST API for inference
    - LoRA fine-tuning support

Example:
    >>> from src.data import EntityExtractor
    >>> extractor = EntityExtractor()
    >>> result = extractor.extract("Rs.500 debited from account 1234")
    >>> print(result.amount)
    '500'

Author: Ranjit Behera
License: MIT
Version: 0.3.0
"""

__version__ = "0.3.0"
__author__ = "Ranjit Behera"
__email__ = "ranjit@example.com"
__license__ = "MIT"

# Package-level imports for convenience
from src.data.extractor import EntityExtractor, FinancialEntity
from src.data.classifier import EmailClassifier, ClassificationResult
from src.data.parser import EmailParser

__all__ = [
    "EntityExtractor",
    "FinancialEntity", 
    "EmailClassifier",
    "ClassificationResult",
    "EmailParser",
    "__version__",
]
