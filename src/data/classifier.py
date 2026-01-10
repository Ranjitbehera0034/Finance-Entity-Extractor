"""
Email Classification Module.

This module provides production-grade email classification using both
rule-based and LLM-based approaches. It categorizes emails into predefined
categories with confidence scoring and transaction detection.

Categories:
    - finance: Bank transactions, investments, payments
    - shopping: E-commerce, orders, deliveries
    - work: Job-related, meetings, projects
    - newsletter: Digests, articles, subscriptions
    - promotional: Marketing, offers, discounts
    - social: Social networks, personal messages
    - other: Uncategorized emails

Example:
    >>> from src.data.classifier import EmailClassifier
    >>> classifier = EmailClassifier()
    >>> result = classifier.classify(
    ...     subject="Transaction Alert",
    ...     sender="HDFC Bank",
    ...     body="Rs.500 debited from your account"
    ... )
    >>> print(result.category)
    'finance'
    >>> print(result.is_transaction)
    True

Author: Ranjit Behera
License: MIT
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

# Configure module logger
logger = logging.getLogger(__name__)


class EmailCategory(Enum):
    """
    Enumeration of email categories.
    
    Each category represents a distinct type of email with specific
    characteristics and handling requirements.
    
    Attributes:
        FINANCE: Bank and financial transaction emails.
        SHOPPING: E-commerce and order-related emails.
        WORK: Professional and job-related emails.
        NEWSLETTER: News, articles, and subscription content.
        PROMOTIONAL: Marketing and advertising emails.
        SOCIAL: Social network and personal communication.
        OTHER: Emails that don't fit other categories.
    """
    
    FINANCE = "finance"
    SHOPPING = "shopping"
    WORK = "work"
    NEWSLETTER = "newsletter"
    PROMOTIONAL = "promotional"
    SOCIAL = "social"
    OTHER = "other"
    
    @classmethod
    def from_string(cls, value: str) -> EmailCategory:
        """
        Convert string to EmailCategory enum.
        
        Args:
            value: Category name as string.
        
        Returns:
            EmailCategory: Corresponding enum value.
        
        Raises:
            ValueError: If value doesn't match any category.
        """
        try:
            return cls(value.lower())
        except ValueError:
            logger.warning(f"Unknown category '{value}', defaulting to OTHER")
            return cls.OTHER


@dataclass
class ClassificationResult:
    """
    Result of email classification.
    
    Contains the predicted category, confidence level, reasoning,
    and whether the email is a financial transaction.
    
    Attributes:
        category: Predicted email category.
        confidence: Confidence level ('high', 'medium', 'low').
        reason: Human-readable explanation for classification.
        is_transaction: True if email is a financial transaction.
        scores: Optional dict of category scores for debugging.
    
    Example:
        >>> result = ClassificationResult(
        ...     category="finance",
        ...     confidence="high",
        ...     reason="Contains debit keywords and amount",
        ...     is_transaction=True
        ... )
        >>> result.to_dict()
        {'category': 'finance', 'confidence': 'high', ...}
    """
    
    category: str
    confidence: str
    reason: str
    is_transaction: bool = False
    scores: Optional[Dict[str, float]] = None
    
    # Validation
    VALID_CONFIDENCE_LEVELS: ClassVar[set] = {"high", "medium", "low"}
    
    def __post_init__(self) -> None:
        """Validate classification result."""
        if self.confidence not in self.VALID_CONFIDENCE_LEVELS:
            logger.warning(f"Invalid confidence '{self.confidence}', setting to 'low'")
            self.confidence = "low"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.
        
        Returns:
            Dict[str, Any]: Classification result as dictionary.
        """
        result = asdict(self)
        if self.scores is None:
            del result['scores']
        return result
    
    def to_json(self) -> str:
        """
        Convert result to JSON string.
        
        Returns:
            str: JSON representation.
        """
        return json.dumps(self.to_dict(), indent=2)


class EmailClassifier:
    """
    Production-grade email classifier with rule-based and LLM support.
    
    This classifier uses a sophisticated pattern matching system to
    categorize emails and detect financial transactions. It can optionally
    use an LLM for more nuanced classification.
    
    Features:
        - Multi-pattern rule-based classification
        - Sender-based categorization
        - Transaction detection
        - Confidence scoring
        - Optional LLM integration
    
    Attributes:
        use_llm: Whether to use LLM for classification.
        model: LLM model instance (if use_llm=True).
        tokenizer: LLM tokenizer (if use_llm=True).
    
    Example:
        >>> classifier = EmailClassifier()
        >>> result = classifier.classify(
        ...     subject="Your order has shipped",
        ...     sender="Amazon.in",
        ...     body="Your order #123 is on the way"
        ... )
        >>> print(result.category)
        'shopping'
    
    Note:
        For production use, the rule-based classifier is recommended
        due to its speed and consistency. LLM mode requires additional
        dependencies and model loading time.
    """
    
    # Transaction detection keywords
    TRANSACTION_KEYWORDS: ClassVar[set] = {
        'debited', 'credited', 'transaction', 'transfer', 'payment',
        'withdrawn', 'deposited', 'paid', 'received', 'upi', 'neft', 'imps',
        'rtgs', 'mandate', 'autopay', 'emi', 'refund', 'cashback'
    }
    
    # Category patterns with senders and keywords
    CATEGORY_PATTERNS: ClassVar[Dict[EmailCategory, Dict[str, List[str]]]] = {
        EmailCategory.FINANCE: {
            'senders': [
                'hdfc', 'icici', 'sbi', 'axis', 'kotak', 'pnb', 'bob',
                'canara', 'union bank', 'idbi', 'yes bank', 'indusind',
                'bank', 'banking', 'credit card', 'mutual fund', 'zerodha',
                'groww', 'upstox', 'cred', 'slice', 'paytm', 'phonepe', 'gpay'
            ],
            'keywords': [
                'transaction', 'statement', 'balance', 'debited', 'credited',
                'payment', 'transfer', 'upi', 'neft', 'imps', 'account',
                'investment', 'dividend', 'interest', 'emi', 'loan', 'credit',
                'debit', 'mandate', 'autopay', 'sip', 'mutual fund'
            ],
        },
        EmailCategory.SHOPPING: {
            'senders': [
                'amazon', 'flipkart', 'myntra', 'ajio', 'nykaa', 'meesho',
                'bigbasket', 'zepto', 'blinkit', 'swiggy', 'zomato', 'dunzo',
                'decathlon', 'ikea', 'pepperfry', 'urban ladder'
            ],
            'keywords': [
                'order', 'shipped', 'delivered', 'delivery', 'tracking',
                'purchase', 'cart', 'checkout', 'invoice', 'receipt',
                'refund', 'return', 'exchange', 'dispatched', 'out for delivery'
            ],
        },
        EmailCategory.WORK: {
            'senders': [
                'linkedin', 'indeed', 'naukri', 'glassdoor', 'angel.co',
                'slack', 'zoom', 'teams', 'meet', 'jira', 'confluence',
                'github', 'gitlab', 'bitbucket', 'notion', 'asana', 'trello'
            ],
            'keywords': [
                'interview', 'meeting', 'agenda', 'project', 'deadline',
                'review', 'standup', 'sprint', 'task', 'report', 'submission',
                'application', 'resume', 'cv', 'job', 'position', 'hiring',
                'salary', 'offer letter', 'joining', 'onboarding'
            ],
        },
        EmailCategory.NEWSLETTER: {
            'senders': [
                'substack', 'medium', 'morning brew', 'digest', 'newsletter',
                'daily', 'weekly', 'update', 'news', 'times', 'hindu', 'express'
            ],
            'keywords': [
                'newsletter', 'digest', 'weekly', 'daily', 'update', 'news',
                'article', 'read', 'story', 'trending', 'top stories',
                'this week', 'this month', 'roundup', 'edition'
            ],
        },
        EmailCategory.PROMOTIONAL: {
            'senders': [
                'offer', 'deal', 'discount', 'sale', 'promo', 'marketing',
                'shopify', 'mailchimp', 'campaign'
            ],
            'keywords': [
                'offer', 'discount', 'sale', 'deal', 'coupon', 'promo',
                'limited time', 'exclusive', 'special', 'hurry', 'ends soon',
                'flash sale', 'clearance', 'save', '% off', 'free shipping',
                'buy now', 'shop now', 'don\'t miss'
            ],
        },
        EmailCategory.SOCIAL: {
            'senders': [
                'facebook', 'instagram', 'twitter', 'whatsapp', 'telegram',
                'snapchat', 'tiktok', 'youtube', 'reddit', 'discord', 'quora'
            ],
            'keywords': [
                'friend request', 'like', 'comment', 'share', 'mentioned',
                'tagged', 'message', 'follow', 'notification', 'birthday',
                'invitation', 'event', 'rsvp', 'group'
            ],
        },
    }
    
    # LLM prompt template
    CLASSIFICATION_PROMPT: ClassVar[str] = """You are an email classifier. Analyze this email and categorize it.

EMAIL:
Subject: {subject}
From: {sender}
Body: {body}

TASK:
Classify this email into exactly ONE category.

CATEGORIES:
- finance: Banks, payments, transactions, investments, credit cards, loans
- shopping: Orders, deliveries, purchases, e-commerce, food delivery
- work: Job-related, recruitment, office, meetings, projects
- newsletter: Digests, subscriptions, blogs, articles
- promotional: Marketing, offers, discounts, advertisements
- social: Social networks, personal messages, invitations
- other: Anything that doesn't fit above

OUTPUT FORMAT (JSON only, no other text):
{{"category": "<category>", "confidence": "<high/medium/low>", "reason": "<brief reason>"}}
"""
    
    def __init__(
        self, 
        use_llm: bool = False,
        model_path: Optional[str] = None,
        debug: bool = False
    ) -> None:
        """
        Initialize the EmailClassifier.
        
        Args:
            use_llm: If True, use LLM for classification (slower but more accurate).
            model_path: Path to LLM model (required if use_llm=True).
            debug: If True, enable debug logging.
        
        Example:
            >>> classifier = EmailClassifier()  # Rule-based
            >>> classifier = EmailClassifier(use_llm=True, model_path="path/to/model")
        
        Raises:
            ValueError: If use_llm=True but model_path not provided.
        """
        self.use_llm = use_llm
        self.debug = debug
        self.model = None
        self.tokenizer = None
        
        if debug:
            logger.setLevel(logging.DEBUG)
        
        if use_llm:
            if not model_path:
                raise ValueError("model_path required when use_llm=True")
            self._load_model(model_path)
        
        logger.info(f"EmailClassifier initialized (use_llm={use_llm})")
    
    def _load_model(self, model_path: str) -> None:
        """Load LLM model for classification."""
        try:
            from mlx_lm import load
            self.model, self.tokenizer = load(model_path)
            logger.info(f"Loaded LLM from {model_path}")
        except ImportError:
            logger.error("mlx_lm not installed. Install with: pip install mlx-lm")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def classify(
        self, 
        subject: str = "",
        sender: str = "",
        body: str = ""
    ) -> ClassificationResult:
        """
        Classify an email into a category.
        
        This method accepts the email components and returns a classification
        result with category, confidence, and reasoning.
        
        Args:
            subject: Email subject line.
            sender: Sender name or email address.
            body: Email body text.
        
        Returns:
            ClassificationResult: Classification with category and confidence.
        
        Example:
            >>> classifier = EmailClassifier()
            >>> result = classifier.classify(
            ...     subject="Transaction Alert",
            ...     sender="HDFC Bank",
            ...     body="Rs.500 debited from your account"
            ... )
            >>> print(result.category)
            'finance'
            >>> print(result.is_transaction)
            True
        
        Note:
            At least one of subject, sender, or body should be non-empty.
            Empty input returns 'other' category with low confidence.
        """
        # Validate input
        if not any([subject, sender, body]):
            logger.warning("Empty input provided")
            return ClassificationResult(
                category=EmailCategory.OTHER.value,
                confidence="low",
                reason="No content to classify",
                is_transaction=False
            )
        
        try:
            if self.use_llm and self.model is not None:
                return self._classify_llm(subject, sender, body)
            else:
                return self._classify_rules(subject, sender, body)
        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)
            return ClassificationResult(
                category=EmailCategory.OTHER.value,
                confidence="low",
                reason=f"Classification error: {str(e)}",
                is_transaction=False
            )
    
    def _classify_rules(
        self, 
        subject: str, 
        sender: str, 
        body: str
    ) -> ClassificationResult:
        """Classify using rule-based approach."""
        combined = f"{subject} {sender} {body}".lower()
        
        # Check for transaction first
        is_transaction = any(kw in combined for kw in self.TRANSACTION_KEYWORDS)
        
        # Score each category
        scores: Dict[EmailCategory, int] = {}
        reasons: Dict[EmailCategory, List[str]] = {}
        
        for category, patterns in self.CATEGORY_PATTERNS.items():
            score = 0
            matched = []
            
            # Check sender patterns (strong signal)
            sender_lower = sender.lower()
            for pattern in patterns.get('senders', []):
                if pattern in sender_lower:
                    score += 3
                    matched.append(f"sender:{pattern}")
            
            # Check keyword patterns
            for keyword in patterns.get('keywords', []):
                if keyword in combined:
                    score += 1
                    if len(matched) < 3:  # Limit reasons
                        matched.append(keyword)
            
            if score > 0:
                scores[category] = score
                reasons[category] = matched
        
        # Handle no matches
        if not scores:
            return ClassificationResult(
                category=EmailCategory.OTHER.value,
                confidence="low",
                reason="No matching patterns found",
                is_transaction=is_transaction
            )
        
        # Get highest scoring category
        best_category = max(scores, key=lambda k: scores[k])
        best_score = scores[best_category]
        best_reasons = reasons.get(best_category, [])
        
        # Determine confidence based on score
        if best_score >= 5:
            confidence = "high"
        elif best_score >= 3:
            confidence = "medium"
        else:
            confidence = "low"
        
        reason = f"Matched: {', '.join(best_reasons[:3])}"
        
        logger.debug(f"Classification: {best_category.value} ({confidence}), scores: {scores}")
        
        return ClassificationResult(
            category=best_category.value,
            confidence=confidence,
            reason=reason,
            is_transaction=is_transaction,
            scores={k.value: v for k, v in scores.items()} if self.debug else None
        )
    
    def _classify_llm(
        self, 
        subject: str, 
        sender: str, 
        body: str
    ) -> ClassificationResult:
        """Classify using LLM."""
        from mlx_lm import generate
        
        # Truncate body if too long
        max_body_length = 1000
        body_truncated = body[:max_body_length] if len(body) > max_body_length else body
        
        # Build prompt
        prompt = self.CLASSIFICATION_PROMPT.format(
            subject=subject,
            sender=sender,
            body=body_truncated
        )
        
        # Generate response
        response = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=100,
            verbose=False
        )
        
        # Parse JSON response
        return self._parse_llm_response(response)
    
    def _parse_llm_response(self, response: str) -> ClassificationResult:
        """Parse LLM JSON response into ClassificationResult."""
        # Find JSON in response
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        
        if not json_match:
            logger.warning(f"No JSON found in LLM response: {response[:100]}")
            return ClassificationResult(
                category=EmailCategory.OTHER.value,
                confidence="low",
                reason="Failed to parse LLM response",
                is_transaction=False
            )
        
        try:
            data = json.loads(json_match.group())
            category = data.get("category", "other").lower()
            
            # Validate category
            if category not in [c.value for c in EmailCategory]:
                category = "other"
            
            return ClassificationResult(
                category=category,
                confidence=data.get("confidence", "medium"),
                reason=data.get("reason", "LLM classification"),
                is_transaction=category == "finance"
            )
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return ClassificationResult(
                category=EmailCategory.OTHER.value,
                confidence="low",
                reason="Invalid JSON in response",
                is_transaction=False
            )
    
    def is_financial_email(self, subject: str, sender: str, body: str) -> bool:
        """
        Quick check if email is financial.
        
        Faster than full classification when you only need to know
        if the email is finance-related.
        
        Args:
            subject: Email subject.
            sender: Email sender.
            body: Email body.
        
        Returns:
            bool: True if email is finance-related.
        """
        result = self.classify(subject, sender, body)
        return result.category == EmailCategory.FINANCE.value or result.is_transaction


# Convenience function
def classify_email(
    subject: str = "",
    sender: str = "",
    body: str = ""
) -> ClassificationResult:
    """
    Convenience function to classify an email without instantiating class.
    
    Args:
        subject: Email subject.
        sender: Email sender.
        body: Email body.
    
    Returns:
        ClassificationResult: Classification result.
    
    Example:
        >>> from src.data.classifier import classify_email
        >>> result = classify_email(
        ...     subject="Order Shipped",
        ...     sender="Amazon",
        ...     body="Your order is on the way"
        ... )
        >>> print(result.category)
        'shopping'
    """
    return EmailClassifier().classify(subject, sender, body)


if __name__ == "__main__":
    # Self-test
    logging.basicConfig(level=logging.DEBUG)
    
    classifier = EmailClassifier(debug=True)
    
    test_cases = [
        ("Transaction Alert", "HDFC Bank", "Rs.500 debited from your account"),
        ("Your order has shipped", "Amazon.in", "Order #123 is on the way"),
        ("Interview Invitation", "LinkedIn", "You have an interview scheduled"),
        ("Weekly Digest", "Substack", "Top 10 articles this week"),
    ]
    
    for subject, sender, body in test_cases:
        result = classifier.classify(subject, sender, body)
        print(f"\n{subject} | {sender}")
        print(f"  → {result.category} ({result.confidence})")
        print(f"  → {result.reason}")
        print(f"  → Transaction: {result.is_transaction}")
