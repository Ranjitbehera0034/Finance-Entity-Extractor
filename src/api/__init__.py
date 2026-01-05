"""
API Module - FastAPI REST API for Email Analysis.

This module provides production-grade REST API endpoints for financial
email entity extraction and classification.

Features:
    - Entity extraction from transaction emails
    - Email classification by category
    - Batch processing support
    - Health check and metrics endpoints
    - OpenAPI documentation

Example:
    Start the server:
        >>> from src.api import app
        >>> # Run with: uvicorn src.api.server:app --reload
    
    Or from command line:
        $ python -m src.api.server
        $ uvicorn src.api.server:app --reload --port 8000

Author: Ranjit Behera
License: MIT
"""

from __future__ import annotations

from src.api.server import (
    app,
    create_app,
    # Models
    EmailInput,
    BatchEmailInput,
    EntityResponse,
    ClassificationResponse,
    FullAnalysisResponse,
    HealthResponse,
    StatsResponse,
    ErrorResponse,
)

__all__ = [
    # Application
    "app",
    "create_app",
    
    # Request models
    "EmailInput",
    "BatchEmailInput",
    
    # Response models
    "EntityResponse",
    "ClassificationResponse",
    "FullAnalysisResponse",
    "HealthResponse",
    "StatsResponse",
    "ErrorResponse",
]
