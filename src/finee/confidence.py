"""
FinEE Confidence - Scoring logic for extraction results.

Calculates confidence scores based on:
- Source of extraction (regex > rules > LLM)
- Completeness of fields
- Consistency between sources
"""

from typing import Dict, List, Optional
from .schema import ExtractionResult, Confidence, ExtractionSource, FieldMeta


# Field weights for confidence calculation
FIELD_WEIGHTS = {
    'amount': 0.25,       # Critical field
    'type': 0.15,         # Critical field
    'date': 0.15,
    'account': 0.10,
    'reference': 0.10,
    'merchant': 0.10,
    'category': 0.10,
    'vpa': 0.05,
}

# Source reliability scores
SOURCE_SCORES = {
    ExtractionSource.REGEX: 0.95,
    ExtractionSource.RULES: 0.85,
    ExtractionSource.LLM: 0.70,
    ExtractionSource.CACHE: 1.0,  # Cached results are already validated
}


def calculate_confidence_score(result: ExtractionResult) -> float:
    """
    Calculate overall confidence score (0.0 to 1.0).
    
    Args:
        result: Extraction result with metadata
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not result:
        return 0.0
    
    total_weight = 0.0
    weighted_score = 0.0
    
    for field_name, weight in FIELD_WEIGHTS.items():
        value = getattr(result, field_name, None)
        
        if value is not None:
            # Get source-based score
            if field_name in result.meta:
                source = result.meta[field_name].source
                field_score = SOURCE_SCORES.get(source, 0.5)
                
                # Apply field-specific confidence if available
                if result.meta[field_name].confidence:
                    field_score *= result.meta[field_name].confidence
            else:
                # Default score for fields without metadata
                field_score = 0.5
            
            weighted_score += weight * field_score
            total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return weighted_score / total_weight


def calculate_completeness(result: ExtractionResult,
                          required: List[str] = None,
                          desired: List[str] = None) -> float:
    """
    Calculate field completeness score.
    
    Args:
        result: Extraction result
        required: List of required field names
        desired: List of desired field names
        
    Returns:
        Completeness score (0.0 to 1.0)
    """
    if required is None:
        required = ['amount', 'type']
    if desired is None:
        desired = ['merchant', 'category', 'date', 'reference']
    
    required_score = 0.0
    for field in required:
        if getattr(result, field, None) is not None:
            required_score += 1.0
    required_score /= len(required) if required else 1.0
    
    desired_score = 0.0
    for field in desired:
        if getattr(result, field, None) is not None:
            desired_score += 1.0
    desired_score /= len(desired) if desired else 1.0
    
    # Required fields are weighted more heavily
    return 0.7 * required_score + 0.3 * desired_score


def determine_confidence_level(score: float,
                              high_threshold: float = 0.9,
                              medium_threshold: float = 0.7) -> Confidence:
    """
    Determine confidence level from score.
    
    Args:
        score: Confidence score (0.0 to 1.0)
        high_threshold: Threshold for HIGH confidence
        medium_threshold: Threshold for MEDIUM confidence
        
    Returns:
        Confidence enum value
    """
    if score >= high_threshold:
        return Confidence.HIGH
    elif score >= medium_threshold:
        return Confidence.MEDIUM
    elif score > 0:
        return Confidence.LOW
    else:
        return Confidence.FAILED


def update_result_confidence(result: ExtractionResult,
                            high_threshold: float = 0.9,
                            medium_threshold: float = 0.7) -> ExtractionResult:
    """
    Update the confidence fields on an ExtractionResult.
    
    Args:
        result: Extraction result to update
        high_threshold: Threshold for HIGH confidence
        medium_threshold: Threshold for MEDIUM confidence
        
    Returns:
        Updated ExtractionResult
    """
    # Calculate score
    score = calculate_confidence_score(result)
    
    # Factor in completeness
    completeness = calculate_completeness(result)
    combined_score = 0.7 * score + 0.3 * completeness
    
    # Update result
    result.confidence_score = combined_score
    result.confidence = determine_confidence_level(
        combined_score, 
        high_threshold, 
        medium_threshold
    )
    
    return result


def should_use_llm(result: ExtractionResult,
                  required: List[str] = None,
                  desired: List[str] = None) -> bool:
    """
    Determine if LLM should be used for additional extraction.
    
    Args:
        result: Current extraction result
        required: Required fields
        desired: Desired fields
        
    Returns:
        True if LLM extraction is recommended
    """
    missing = result.get_missing_fields(required, desired)
    
    # Always use LLM if required fields are missing
    if required:
        for field in required:
            if field in missing:
                return True
    
    # Use LLM if more than half of desired fields are missing
    if desired:
        missing_desired = [f for f in missing if f in desired]
        if len(missing_desired) > len(desired) / 2:
            return True
    
    return False


def get_extraction_summary(result: ExtractionResult) -> Dict[str, str]:
    """
    Get a summary of extraction sources for each field.
    
    Args:
        result: Extraction result
        
    Returns:
        Dict mapping field names to source descriptions
    """
    summary = {}
    
    for field_name in FIELD_WEIGHTS.keys():
        value = getattr(result, field_name, None)
        
        if value is not None:
            if field_name in result.meta:
                source = result.meta[field_name].source.value
                conf = result.meta[field_name].confidence
                summary[field_name] = f"{source} ({conf:.0%})"
            else:
                summary[field_name] = "unknown"
        else:
            summary[field_name] = "missing"
    
    return summary
