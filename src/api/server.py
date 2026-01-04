"""
FastAPI Server for LLM Mail Trainer
REST API for email entity extraction and classification.
"""

import os
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.extractor import EntityExtractor, FinancialEntity
from data.classifier import EmailClassifier, ClassificationResult


# ============================================
# Pydantic Models (Request/Response)
# ============================================

class EmailInput(BaseModel):
    """Input model for email analysis."""
    subject: str = Field(default="", description="Email subject line")
    body: str = Field(..., description="Email body text")
    sender: str = Field(default="", description="Sender name/email")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "subject": "❗ You have done a UPI txn. Check details!",
                "body": "Dear Customer, Rs.2500.00 has been debited from account 3545 to VPA swiggy@ybl for Swiggy order on 28-12-25. Your UPI transaction reference number is 534567891234.",
                "sender": "HDFC Bank InstaAlerts"
            }
        }
    }


class BatchEmailInput(BaseModel):
    """Input model for batch processing."""
    emails: List[EmailInput] = Field(..., description="List of emails to process")


class EntityResponse(BaseModel):
    """Response model for entity extraction."""
    success: bool
    entities: dict
    extraction_time_ms: float
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "entities": {
                    "amount": "2500.00",
                    "type": "debit",
                    "account": "3545",
                    "date": "28-12-25",
                    "reference": "534567891234",
                    "merchant": "swiggy",
                    "payment_method": "upi",
                    "category": "food"
                },
                "extraction_time_ms": 1.5
            }
        }
    }


class ClassificationResponse(BaseModel):
    """Response model for email classification."""
    category: str
    confidence: str
    reason: str
    is_transaction: bool
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "category": "finance",
                "confidence": "high",
                "reason": "Matched 5 patterns for finance",
                "is_transaction": True
            }
        }
    }


class FullAnalysisResponse(BaseModel):
    """Response model for full email analysis."""
    classification: ClassificationResponse
    entities: Optional[dict] = None
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str


class StatsResponse(BaseModel):
    """API statistics response."""
    total_requests: int
    entities_extracted: int
    emails_classified: int
    uptime_seconds: float


# ============================================
# Application Setup
# ============================================

# Global instances
extractor = EntityExtractor()
classifier = EmailClassifier(use_llm=False)

# Stats tracking
stats = {
    "total_requests": 0,
    "entities_extracted": 0,
    "emails_classified": 0,
    "start_time": datetime.now()
}


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="🧠 LLM Mail Trainer API",
        description="""
## Financial Email Entity Extraction API

Extract structured financial data from emails using ML-powered analysis.

### Features:
- **Entity Extraction**: Amount, type, account, date, reference, merchant, category
- **Email Classification**: Finance, shopping, work, newsletter, promotional, etc.
- **Batch Processing**: Process multiple emails at once

### Example Usage:
```python
import requests

response = requests.post(
    "http://localhost:8000/extract",
    json={
        "body": "Rs.500 debited from account 1234 on 01-01-25",
        "subject": "Transaction Alert"
    }
)
print(response.json())
```
        """,
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc"
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


# ============================================
# API Endpoints
# ============================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LLM Mail Trainer API",
        "version": "0.2.0",
        "docs": "/docs",
        "endpoints": {
            "extract": "POST /extract - Extract entities from email",
            "classify": "POST /classify - Classify email category",
            "analyze": "POST /analyze - Full analysis (classify + extract)",
            "batch": "POST /batch - Process multiple emails",
            "health": "GET /health - Health check",
            "stats": "GET /stats - API statistics"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.2.0",
        timestamp=datetime.now().isoformat()
    )


@app.get("/stats", response_model=StatsResponse, tags=["General"])
async def get_stats():
    """Get API usage statistics."""
    uptime = (datetime.now() - stats["start_time"]).total_seconds()
    return StatsResponse(
        total_requests=stats["total_requests"],
        entities_extracted=stats["entities_extracted"],
        emails_classified=stats["emails_classified"],
        uptime_seconds=uptime
    )


@app.post("/extract", response_model=EntityResponse, tags=["Entity Extraction"])
async def extract_entities(email: EmailInput):
    """
    Extract financial entities from an email.
    
    Returns structured data including:
    - amount, type (debit/credit), account, date, reference
    - merchant name (swiggy, amazon, etc.)
    - payment method (upi, neft, card)
    - category (food, shopping, transport)
    """
    stats["total_requests"] += 1
    
    start = datetime.now()
    
    try:
        # Combine subject and body for extraction
        full_text = f"Subject: {email.subject}\n\n{email.body}"
        result = extractor.extract(full_text)
        
        elapsed = (datetime.now() - start).total_seconds() * 1000
        stats["entities_extracted"] += 1
        
        return EntityResponse(
            success=result.is_valid(),
            entities=result.to_dict(),
            extraction_time_ms=round(elapsed, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify", response_model=ClassificationResponse, tags=["Classification"])
async def classify_email(email: EmailInput):
    """
    Classify an email into a category.
    
    Categories:
    - finance: Bank transactions, payments, investments
    - shopping: Orders, deliveries, e-commerce
    - work: Job-related, recruitment, meetings
    - newsletter: Digests, articles, subscriptions
    - promotional: Marketing, offers, discounts
    - social: Social networks, personal messages
    - other: Everything else
    """
    stats["total_requests"] += 1
    stats["emails_classified"] += 1
    
    try:
        result = classifier.classify(
            subject=email.subject,
            sender=email.sender,
            body=email.body
        )
        
        return ClassificationResponse(
            category=result.category,
            confidence=result.confidence,
            reason=result.reason,
            is_transaction=result.is_transaction
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=FullAnalysisResponse, tags=["Analysis"])
async def full_analysis(email: EmailInput):
    """
    Perform full analysis: classify the email and extract entities if financial.
    
    This combines classification and entity extraction in one call.
    Entities are only extracted if the email is classified as finance.
    """
    stats["total_requests"] += 1
    
    start = datetime.now()
    
    try:
        # Classify first
        classification = classifier.classify(
            subject=email.subject,
            sender=email.sender,
            body=email.body
        )
        stats["emails_classified"] += 1
        
        # Extract entities if finance-related
        entities = None
        if classification.category == "finance" or classification.is_transaction:
            full_text = f"Subject: {email.subject}\n\n{email.body}"
            result = extractor.extract(full_text)
            entities = result.to_dict()
            stats["entities_extracted"] += 1
        
        elapsed = (datetime.now() - start).total_seconds() * 1000
        
        return FullAnalysisResponse(
            classification=ClassificationResponse(
                category=classification.category,
                confidence=classification.confidence,
                reason=classification.reason,
                is_transaction=classification.is_transaction
            ),
            entities=entities,
            processing_time_ms=round(elapsed, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", tags=["Batch Processing"])
async def batch_process(batch: BatchEmailInput):
    """
    Process multiple emails at once.
    
    Returns analysis results for each email in the batch.
    """
    stats["total_requests"] += 1
    
    start = datetime.now()
    results = []
    
    for email in batch.emails:
        try:
            # Classify
            classification = classifier.classify(
                subject=email.subject,
                sender=email.sender,
                body=email.body
            )
            stats["emails_classified"] += 1
            
            # Extract if finance
            entities = None
            if classification.category == "finance" or classification.is_transaction:
                full_text = f"Subject: {email.subject}\n\n{email.body}"
                result = extractor.extract(full_text)
                entities = result.to_dict()
                stats["entities_extracted"] += 1
            
            results.append({
                "subject": email.subject[:50],
                "classification": {
                    "category": classification.category,
                    "confidence": classification.confidence,
                    "is_transaction": classification.is_transaction
                },
                "entities": entities
            })
        except Exception as e:
            results.append({
                "subject": email.subject[:50],
                "error": str(e)
            })
    
    elapsed = (datetime.now() - start).total_seconds() * 1000
    
    return {
        "total_processed": len(results),
        "processing_time_ms": round(elapsed, 2),
        "results": results
    }


# ============================================
# CLI Runner
# ============================================

def main():
    """Run the API server."""
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║          🧠 LLM Mail Trainer API Server                  ║
╠══════════════════════════════════════════════════════════╣
║  Docs:    http://{host}:{port}/docs                          ║
║  ReDoc:   http://{host}:{port}/redoc                         ║
║  Health:  http://{host}:{port}/health                        ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=True
    )


if __name__ == "__main__":
    main()
