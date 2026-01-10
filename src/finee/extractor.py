"""
FinEE Extractor - Main orchestrator for the extraction pipeline.

Implements the 5-tier additive extraction pipeline:
- Tier 0: Hash Cache
- Tier 1: Regex Engine
- Tier 2: Rule-Based Mapping
- Tier 3: LLM (targeted extraction)
- Tier 4: Validation + Normalization
"""

import time
import logging
from typing import Optional, List, Dict, Any

from .schema import (
    ExtractionResult, ExtractionConfig, TransactionType, 
    Category, Confidence, ExtractionSource, FieldMeta
)
from .cache import LRUCache, get_cache
from .regex_engine import RegexEngine, get_regex_engine
from .merchants import get_merchant_and_category
from .normalizer import normalize_amount, normalize_date, normalize_vpa
from .validator import repair_llm_json, validate_extraction_result
from .confidence import update_result_confidence, should_use_llm
from .prompt import get_targeted_prompt, get_full_extraction_prompt, parse_targeted_response
from .backends import get_backend, get_available_backends, BaseBackend

logger = logging.getLogger(__name__)


class FinEE:
    """
    Finance Entity Extractor - Main extraction class.
    
    Orchestrates the 5-tier additive extraction pipeline with graceful degradation.
    Always returns a result, never crashes.
    """
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        """
        Initialize the extractor.
        
        Args:
            config: Extraction configuration (uses defaults if None)
        """
        self.config = config or ExtractionConfig()
        
        # Initialize components
        self._cache: Optional[LRUCache] = None
        self._regex_engine: Optional[RegexEngine] = None
        self._backend: Optional[BaseBackend] = None
        self._backend_loaded = False
        
        # Initialize cache if enabled
        if self.config.cache_enabled:
            self._cache = get_cache(self.config.cache_max_size)
        
        # Initialize regex engine
        self._regex_engine = get_regex_engine()
    
    def _lazy_load_backend(self) -> bool:
        """
        Lazy load LLM backend.
        
        Returns:
            True if backend is available
        """
        if self._backend_loaded:
            return self._backend is not None
        
        self._backend_loaded = True
        
        if not self.config.use_llm:
            return False
        
        try:
            self._backend = get_backend(model_id=self.config.model_id)
            if self._backend:
                logger.info(f"Backend loaded: {self._backend.name}")
                return True
        except Exception as e:
            logger.warning(f"Failed to load LLM backend: {e}")
        
        return False
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extract financial entities from text.
        
        This is the main entry point. It runs the full 5-tier pipeline
        with graceful degradation.
        
        Args:
            text: Transaction text (bank SMS, email, etc.)
            
        Returns:
            ExtractionResult with extracted entities
        """
        start_time = time.time()
        
        # Tier 0: Cache Check
        if self._cache:
            cached = self._cache.get(text)
            if cached:
                cached.processing_time_ms = (time.time() - start_time) * 1000
                return cached
        
        # Tier 1: Regex Extraction
        result = self._tier1_regex(text)
        
        # Tier 2: Rule-Based Mapping
        result = self._tier2_rules(result)
        
        # Tier 3: LLM (if needed and available)
        missing_fields = result.get_missing_fields(
            self.config.required_fields,
            self.config.desired_fields
        )
        
        if missing_fields and self.config.use_llm:
            result = self._tier3_llm(text, result, missing_fields)
        
        # Tier 4: Validation + Normalization
        result = self._tier4_validate(result)
        
        # Calculate processing time
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        # Store in cache
        if self._cache and result.is_complete():
            self._cache.set(text, result)
        
        return result
    
    def _tier1_regex(self, text: str) -> ExtractionResult:
        """
        Tier 1: Extract entities using regex patterns.
        
        Args:
            text: Input text
            
        Returns:
            ExtractionResult with regex-extracted fields
        """
        try:
            result = self._regex_engine.extract(text)
            result.raw_input = text
            return result
        except Exception as e:
            logger.warning(f"Tier 1 (regex) failed: {e}")
            return ExtractionResult(raw_input=text)
    
    def _tier2_rules(self, result: ExtractionResult) -> ExtractionResult:
        """
        Tier 2: Enrich with rule-based mappings.
        
        Args:
            result: Current extraction result
            
        Returns:
            Enriched ExtractionResult
        """
        try:
            # Get merchant and category from VPA
            merchant, category = get_merchant_and_category(
                vpa=result.vpa,
                text=result.raw_input
            )
            
            if merchant and not result.merchant:
                result.merchant = merchant
                result.meta['merchant'] = FieldMeta(
                    source=ExtractionSource.RULES,
                    confidence=0.85
                )
            
            if category and not result.category:
                result.category = Category(category) if category in [c.value for c in Category] else Category.OTHER
                result.meta['category'] = FieldMeta(
                    source=ExtractionSource.RULES,
                    confidence=0.80
                )
            
            return result
            
        except Exception as e:
            logger.warning(f"Tier 2 (rules) failed: {e}")
            return result
    
    def _tier3_llm(self, text: str, result: ExtractionResult, 
                   missing_fields: List[str]) -> ExtractionResult:
        """
        Tier 3: Fill missing fields using LLM.
        
        Uses targeted prompts for specific fields rather than full extraction.
        
        Args:
            text: Original input text
            result: Current extraction result
            missing_fields: Fields to extract with LLM
            
        Returns:
            Updated ExtractionResult
        """
        if not self._lazy_load_backend():
            logger.debug("No LLM backend available, skipping Tier 3")
            return result
        
        try:
            # Load model if not already loaded
            if not self._backend.is_loaded:
                self._backend.load_model(self.config.model_path)
            
            # Use targeted prompts for specific fields
            for field in missing_fields:
                if field in ['merchant', 'category', 'date', 'reference']:
                    value = self._extract_single_field(text, field)
                    if value:
                        self._set_field(result, field, value, ExtractionSource.LLM)
            
            # If still missing critical fields, try full extraction
            still_missing = result.get_missing_fields(self.config.required_fields, [])
            if still_missing:
                llm_result = self._full_llm_extraction(text)
                if llm_result:
                    result.merge(llm_result, overwrite=False)
            
            return result
            
        except Exception as e:
            logger.warning(f"Tier 3 (LLM) failed: {e}")
            return result
    
    def _extract_single_field(self, text: str, field: str) -> Optional[str]:
        """Extract a single field using targeted prompt."""
        try:
            prompt = get_targeted_prompt(field, text)
            response = self._backend.generate(
                prompt,
                max_tokens=50,
                temperature=self.config.llm_temperature
            )
            return parse_targeted_response(field, response)
        except Exception as e:
            logger.debug(f"Single field extraction failed for {field}: {e}")
            return None
    
    def _full_llm_extraction(self, text: str) -> Optional[ExtractionResult]:
        """Run full LLM extraction as fallback."""
        try:
            prompt = get_full_extraction_prompt(text)
            response = self._backend.generate(
                prompt,
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature
            )
            
            parsed = repair_llm_json(response)
            if parsed:
                result = validate_extraction_result(parsed)
                result.raw_llm_output = response
                
                # Mark all fields as LLM-sourced
                for field in ['amount', 'type', 'date', 'account', 'reference', 
                              'vpa', 'merchant', 'category']:
                    if getattr(result, field, None) is not None:
                        result.meta[field] = FieldMeta(
                            source=ExtractionSource.LLM,
                            confidence=0.70
                        )
                
                return result
            
        except Exception as e:
            logger.debug(f"Full LLM extraction failed: {e}")
        
        return None
    
    def _set_field(self, result: ExtractionResult, field: str, 
                   value: Any, source: ExtractionSource) -> None:
        """Set a field on the result with metadata."""
        if field == 'category':
            try:
                value = Category(value.lower())
            except (ValueError, AttributeError):
                value = Category.OTHER
        elif field == 'date':
            value = normalize_date(value)
        
        setattr(result, field, value)
        result.meta[field] = FieldMeta(
            source=source,
            confidence=0.70 if source == ExtractionSource.LLM else 0.85,
            raw_value=str(value)
        )
    
    def _tier4_validate(self, result: ExtractionResult) -> ExtractionResult:
        """
        Tier 4: Validate and normalize all fields.
        
        Args:
            result: Extraction result to validate
            
        Returns:
            Validated and normalized result
        """
        try:
            # Normalize amount
            if result.amount is not None:
                result.amount = normalize_amount(result.amount)
            
            # Normalize date
            if result.date:
                result.date = normalize_date(result.date)
            
            # Normalize VPA
            if result.vpa:
                result.vpa = normalize_vpa(result.vpa)
            
            # Update confidence
            result = update_result_confidence(
                result,
                self.config.high_confidence_threshold,
                self.config.medium_confidence_threshold
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Tier 4 (validation) failed: {e}")
            result.confidence = Confidence.LOW
            return result
    
    def extract_batch(self, texts: List[str]) -> List[ExtractionResult]:
        """
        Extract entities from multiple texts.
        
        Args:
            texts: List of transaction texts
            
        Returns:
            List of ExtractionResults
        """
        return [self.extract(text) for text in texts]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        stats = {
            'cache_enabled': self.config.cache_enabled,
            'llm_enabled': self.config.use_llm,
            'available_backends': get_available_backends(),
            'active_backend': self._backend.name if self._backend else None,
        }
        
        if self._cache:
            cache_stats = self._cache.get_stats()
            stats['cache'] = cache_stats.to_dict()
        
        return stats


# Module-level singleton
_extractor: Optional[FinEE] = None


def get_extractor(config: Optional[ExtractionConfig] = None) -> FinEE:
    """Get or create the global extractor instance."""
    global _extractor
    if _extractor is None or config is not None:
        _extractor = FinEE(config)
    return _extractor


def extract(text: str) -> ExtractionResult:
    """
    Extract financial entities from text.
    
    Convenience function that uses the global extractor.
    
    Args:
        text: Transaction text
        
    Returns:
        ExtractionResult
    """
    return get_extractor().extract(text)
