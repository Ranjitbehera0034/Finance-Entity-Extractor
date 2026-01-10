"""
FastAPI Server for LLM Mail Trainer.

Production-grade REST API for financial email entity extraction and
classification. Designed for high performance and reliability.

Features:
    - Entity extraction endpoint (/extract)
    - Email classification endpoint (/classify)
    - Full analysis endpoint (/analyze)
    - Batch processing endpoint (/batch)
    - Health check and metrics endpoints
    - OpenAPI documentation
    - CORS support
    - Request validation
    - Error handling

Endpoints:
    GET  /           - API information
    GET  /health     - Health check
    GET  /stats      - Usage statistics
    POST /extract    - Extract entities from email
    POST /classify   - Classify email category
    POST /analyze    - Full analysis (classify + extract)
    POST /batch      - Process multiple emails

Example:
    Start the server:
        $ uvicorn src.api.server:app --reload --port 8000
    
    Make a request:
        $ curl -X POST http://localhost:8000/extract \\
            -H "Content-Type: application/json" \\
            -d '{"body": "Rs.500 debited from account 1234"}'

Author: Ranjit Behera
License: MIT
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.extractor import EntityExtractor, FinancialEntity
from data.classifier import EmailClassifier, ClassificationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models (Request/Response Schemas)
# =============================================================================

class EmailInput(BaseModel):
    """
    Input model for email analysis requests.
    
    Attributes:
        subject: Email subject line (optional).
        body: Email body text (required).
        sender: Sender name or email address (optional).
    
    Example:
        {
            "subject": "Transaction Alert",
            "body": "Rs.500 debited from account 1234",
            "sender": "HDFC Bank"
        }
    """
    
    subject: str = Field(
        default="",
        description="Email subject line",
        max_length=500,
    )
    body: str = Field(
        ...,
        description="Email body text (required)",
        min_length=1,
        max_length=10000,
    )
    sender: str = Field(
        default="",
        description="Sender name or email address",
        max_length=200,
    )
    
    @field_validator("body")
    @classmethod
    def body_not_empty(cls, v: str) -> str:
        """Validate body is not just whitespace."""
        if not v.strip():
            raise ValueError("Body cannot be empty or whitespace only")
        return v.strip()
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "subject": "❗ You have done a UPI txn. Check details!",
                    "body": "Dear Customer, Rs.2500.00 has been debited from account 3545 to VPA swiggy@ybl on 28-12-25. Reference: 534567891234.",
                    "sender": "HDFC Bank InstaAlerts"
                }
            ]
        }
    }


class BatchEmailInput(BaseModel):
    """
    Input model for batch processing.
    
    Attributes:
        emails: List of emails to process (max 100).
    """
    
    emails: List[EmailInput] = Field(
        ...,
        description="List of emails to process",
        min_length=1,
        max_length=100,
    )


class EntityResponse(BaseModel):
    """
    Response model for entity extraction.
    
    Attributes:
        success: Whether extraction found valid entities.
        entities: Dictionary of extracted entities.
        extraction_time_ms: Processing time in milliseconds.
        confidence: Confidence score (0.0 to 1.0).
    """
    
    success: bool = Field(description="Extraction found valid entities")
    entities: Dict[str, Any] = Field(description="Extracted entities")
    extraction_time_ms: float = Field(description="Processing time in milliseconds")
    confidence: float = Field(default=0.0, description="Confidence score")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "entities": {
                        "amount": "2500.00",
                        "type": "debit",
                        "account": "3545",
                        "date": "28-12-25",
                        "reference": "534567891234",
                        "merchant": "swiggy",
                        "category": "food"
                    },
                    "extraction_time_ms": 1.5,
                    "confidence": 0.85
                }
            ]
        }
    }


class ClassificationResponse(BaseModel):
    """
    Response model for email classification.
    
    Attributes:
        category: Predicted email category.
        confidence: Confidence level (high/medium/low).
        reason: Explanation for classification.
        is_transaction: Whether email is a financial transaction.
    """
    
    category: str = Field(description="Predicted category")
    confidence: str = Field(description="Confidence level")
    reason: str = Field(description="Classification reasoning")
    is_transaction: bool = Field(description="Is financial transaction")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "category": "finance",
                    "confidence": "high",
                    "reason": "Matched: sender:hdfc, debited, account",
                    "is_transaction": True
                }
            ]
        }
    }


class FullAnalysisResponse(BaseModel):
    """
    Response model for full email analysis.
    
    Combines classification and entity extraction results.
    """
    
    classification: ClassificationResponse
    entities: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extracted entities (only for finance emails)"
    )
    processing_time_ms: float = Field(description="Total processing time")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    timestamp: str = Field(description="Current timestamp")
    uptime_seconds: float = Field(description="Server uptime")


class StatsResponse(BaseModel):
    """API statistics response."""
    
    total_requests: int = Field(description="Total requests processed")
    entities_extracted: int = Field(description="Successful extractions")
    emails_classified: int = Field(description="Emails classified")
    uptime_seconds: float = Field(description="Server uptime")
    requests_per_minute: float = Field(description="Request rate")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None)


# =============================================================================
# Application State and Configuration
# =============================================================================

class AppState:
    """
    Application state container.
    
    Holds global state including statistics, service instances,
    and configuration.
    """
    
    def __init__(self) -> None:
        self.start_time = datetime.now()
        self.total_requests = 0
        self.entities_extracted = 0
        self.emails_classified = 0
        
        # Initialize services
        self.extractor = EntityExtractor()
        self.classifier = EmailClassifier(use_llm=False)
        
        logger.info("Application state initialized")
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate server uptime in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def requests_per_minute(self) -> float:
        """Calculate request rate."""
        uptime_minutes = self.uptime_seconds / 60
        if uptime_minutes < 1:
            return self.total_requests * 60
        return self.total_requests / uptime_minutes


# Global state
state = AppState()


# =============================================================================
# Application Factory
# =============================================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application instance.
    
    Example:
        >>> app = create_app()
        >>> # Run with: uvicorn src.api.server:app
    """
    
    app = FastAPI(
        title="🧠 LLM Mail Trainer API",
        description="""
## Financial Email Entity Extraction API

Production-grade API for extracting structured financial data from emails.

### Features
- **Entity Extraction**: Amount, type, account, date, reference, merchant, category
- **Email Classification**: Finance, shopping, work, newsletter, promotional, etc.
- **Batch Processing**: Process multiple emails efficiently
- **High Performance**: Optimized for speed with < 5ms response time

### Supported Banks
HDFC, ICICI, SBI, Axis, Kotak, PNB, BoB, and more.

### Supported Payment Platforms
PhonePe, GPay, Paytm, BHIM UPI

### Quick Example
```python
import requests

response = requests.post(
    "http://localhost:8000/extract",
    json={
        "body": "Rs.500 debited from account 1234 on 01-01-26",
        "subject": "Transaction Alert"
    }
)
print(response.json())
```

### Links
- [Model on HuggingFace](https://huggingface.co/Ranjit0034/finance-entity-extractor)
- [GitHub Repository](https://github.com/ranjit/llm-mail-trainer)
        """,
        version="0.3.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "Ranjit Behera",
            "email": "ranjit@example.com",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, 
    exc: HTTPException
) -> JSONResponse:
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request, 
    exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
        ).model_dump(),
    )


# =============================================================================
# API Endpoints
# =============================================================================

@app.get(
    "/",
    tags=["General"],
    summary="API Information",
    response_description="API metadata and available endpoints",
)
async def root() -> Dict[str, Any]:
    """
    Get API information and available endpoints.
    
    Returns a summary of the API including version, documentation links,
    and available endpoints.
    """
    return {
        "name": "LLM Mail Trainer API",
        "version": "0.3.0",
        "description": "Financial email entity extraction and classification",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
        "endpoints": {
            "extract": "POST /extract - Extract entities from email",
            "classify": "POST /classify - Classify email category",
            "analyze": "POST /analyze - Full analysis (classify + extract)",
            "batch": "POST /batch - Process multiple emails",
            "health": "GET /health - Health check",
            "stats": "GET /stats - API statistics",
        },
        "model": "Ranjit0034/finance-entity-extractor",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["General"],
    summary="Health Check",
    response_description="Service health status",
)
async def health_check() -> HealthResponse:
    """
    Check API health status.
    
    Returns the current health status of the API including version
    and uptime information.
    """
    return HealthResponse(
        status="healthy",
        version="0.3.0",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=round(state.uptime_seconds, 2),
    )


@app.get(
    "/stats",
    response_model=StatsResponse,
    tags=["General"],
    summary="Usage Statistics",
    response_description="API usage statistics",
)
async def get_stats() -> StatsResponse:
    """
    Get API usage statistics.
    
    Returns metrics including total requests, successful extractions,
    and performance data.
    """
    return StatsResponse(
        total_requests=state.total_requests,
        entities_extracted=state.entities_extracted,
        emails_classified=state.emails_classified,
        uptime_seconds=round(state.uptime_seconds, 2),
        requests_per_minute=round(state.requests_per_minute, 2),
    )


@app.post(
    "/extract",
    response_model=EntityResponse,
    tags=["Entity Extraction"],
    summary="Extract Financial Entities",
    response_description="Extracted entities from email",
)
async def extract_entities(email: EmailInput) -> EntityResponse:
    """
    Extract financial entities from an email.
    
    Analyzes the email text and extracts structured data including:
    - **amount**: Transaction amount
    - **type**: Debit or credit
    - **account**: Account number (masked)
    - **date**: Transaction date
    - **reference**: UPI/IMPS reference number
    - **merchant**: Identified merchant name
    - **category**: Transaction category (food, shopping, etc.)
    
    Args:
        email: Email content with subject, body, and sender.
    
    Returns:
        EntityResponse: Extracted entities with success status.
    
    Raises:
        HTTPException: If extraction fails critically.
    """
    state.total_requests += 1
    start = datetime.now()
    
    try:
        # Combine subject and body for extraction
        full_text = f"Subject: {email.subject}\n\n{email.body}"
        result = state.extractor.extract(full_text)
        
        elapsed = (datetime.now() - start).total_seconds() * 1000
        
        if result.is_valid():
            state.entities_extracted += 1
        
        return EntityResponse(
            success=result.is_valid(),
            entities=result.to_dict(),
            extraction_time_ms=round(elapsed, 2),
            confidence=round(result.confidence_score(), 2),
        )
        
    except Exception as e:
        logger.error(f"Extraction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(e)}"
        )


@app.post(
    "/classify",
    response_model=ClassificationResponse,
    tags=["Classification"],
    summary="Classify Email",
    response_description="Email classification result",
)
async def classify_email(email: EmailInput) -> ClassificationResponse:
    """
    Classify an email into a category.
    
    Categories:
    - **finance**: Bank transactions, payments, investments
    - **shopping**: Orders, deliveries, e-commerce
    - **work**: Job-related, recruitment, meetings
    - **newsletter**: Digests, articles, subscriptions
    - **promotional**: Marketing, offers, discounts
    - **social**: Social networks, personal messages
    - **other**: Uncategorized emails
    
    Args:
        email: Email content to classify.
    
    Returns:
        ClassificationResponse: Category with confidence and reasoning.
    """
    state.total_requests += 1
    state.emails_classified += 1
    
    try:
        result = state.classifier.classify(
            subject=email.subject,
            sender=email.sender,
            body=email.body,
        )
        
        return ClassificationResponse(
            category=result.category,
            confidence=result.confidence,
            reason=result.reason,
            is_transaction=result.is_transaction,
        )
        
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )


@app.post(
    "/analyze",
    response_model=FullAnalysisResponse,
    tags=["Analysis"],
    summary="Full Email Analysis",
    response_description="Complete analysis with classification and entities",
)
async def full_analysis(email: EmailInput) -> FullAnalysisResponse:
    """
    Perform full analysis: classify the email and extract entities.
    
    This endpoint combines classification and entity extraction in one call.
    Entities are only extracted if the email is classified as finance-related.
    
    Args:
        email: Email content to analyze.
    
    Returns:
        FullAnalysisResponse: Classification and extracted entities.
    """
    state.total_requests += 1
    start = datetime.now()
    
    try:
        # Classify first
        classification = state.classifier.classify(
            subject=email.subject,
            sender=email.sender,
            body=email.body,
        )
        state.emails_classified += 1
        
        # Extract entities if finance-related
        entities = None
        if classification.category == "finance" or classification.is_transaction:
            full_text = f"Subject: {email.subject}\n\n{email.body}"
            result = state.extractor.extract(full_text)
            entities = result.to_dict()
            if result.is_valid():
                state.entities_extracted += 1
        
        elapsed = (datetime.now() - start).total_seconds() * 1000
        
        return FullAnalysisResponse(
            classification=ClassificationResponse(
                category=classification.category,
                confidence=classification.confidence,
                reason=classification.reason,
                is_transaction=classification.is_transaction,
            ),
            entities=entities,
            processing_time_ms=round(elapsed, 2),
        )
        
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post(
    "/batch",
    tags=["Batch Processing"],
    summary="Batch Process Emails",
    response_description="Results for all emails in batch",
)
async def batch_process(batch: BatchEmailInput) -> Dict[str, Any]:
    """
    Process multiple emails at once.
    
    Each email is classified and entities are extracted for finance emails.
    Results are returned in the same order as input.
    
    Args:
        batch: List of emails to process (max 100).
    
    Returns:
        Dict with processing results for each email.
    
    Note:
        Failed individual emails don't fail the entire batch.
        Check the 'error' field in each result.
    """
    state.total_requests += 1
    start = datetime.now()
    results = []
    
    for email in batch.emails:
        try:
            # Classify
            classification = state.classifier.classify(
                subject=email.subject,
                sender=email.sender,
                body=email.body,
            )
            state.emails_classified += 1
            
            # Extract if finance
            entities = None
            if classification.category == "finance" or classification.is_transaction:
                full_text = f"Subject: {email.subject}\n\n{email.body}"
                result = state.extractor.extract(full_text)
                entities = result.to_dict()
                if result.is_valid():
                    state.entities_extracted += 1
            
            results.append({
                "subject": email.subject[:50] if email.subject else "(no subject)",
                "classification": {
                    "category": classification.category,
                    "confidence": classification.confidence,
                    "is_transaction": classification.is_transaction,
                },
                "entities": entities,
            })
            
        except Exception as e:
            logger.warning(f"Batch item error: {e}")
            results.append({
                "subject": email.subject[:50] if email.subject else "(no subject)",
                "error": str(e),
            })
    
    elapsed = (datetime.now() - start).total_seconds() * 1000
    
    return {
        "total_processed": len(results),
        "successful": sum(1 for r in results if "error" not in r),
        "failed": sum(1 for r in results if "error" in r),
        "processing_time_ms": round(elapsed, 2),
        "results": results,
    }


# =============================================================================
# CLI Runner
# =============================================================================

def main() -> None:
    """
    Run the API server from command line.
    
    Usage:
        python -m src.api.server
        
    Environment Variables:
        HOST: Server host (default: 0.0.0.0)
        PORT: Server port (default: 8000)
        LOG_LEVEL: Logging level (default: info)
    """
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              🧠 LLM Mail Trainer API Server                  ║
╠══════════════════════════════════════════════════════════════╣
║  Swagger Docs:    http://{host}:{port}/docs                       ║
║  ReDoc:           http://{host}:{port}/redoc                      ║
║  Health Check:    http://{host}:{port}/health                     ║
║  OpenAPI JSON:    http://{host}:{port}/openapi.json               ║
╠══════════════════════════════════════════════════════════════╣
║  Model: Ranjit0034/finance-entity-extractor                  ║
║  Version: 0.3.0                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=True,
    )


if __name__ == "__main__":
    main()
