"""
Email Classifier Module
Classifies emails into categories using LLM or rule-based methods.
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class EmailCategory(Enum):
    """Email categories."""
    FINANCE = "finance"
    SHOPPING = "shopping"
    SOCIAL = "social"
    WORK = "work"
    NEWSLETTER = "newsletter"
    PROMOTIONAL = "promotional"
    SPAM = "spam"
    OTHER = "other"


@dataclass
class ClassificationResult:
    """Result of email classification."""
    category: str
    confidence: str  # 'high', 'medium', 'low'
    reason: str
    is_transaction: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class EmailClassifier:
    """
    Classify emails into categories.
    
    Supports both rule-based and LLM-based classification.
    """
    
    # Keywords for rule-based classification
    CATEGORY_KEYWORDS = {
        EmailCategory.FINANCE: {
            'senders': ['icici', 'hdfc', 'sbi', 'axis', 'kotak', 'groww', 
                       'zerodha', 'paytm', 'phonepe', 'gpay', 'bank'],
            'keywords': ['debited', 'credited', 'transaction', 'upi', 'neft',
                        'imps', 'balance', 'statement', 'emi', 'loan', 'credit card',
                        'debit card', 'payment', 'transfer', 'withdraw', 'deposit'],
        },
        EmailCategory.SHOPPING: {
            'senders': ['amazon', 'flipkart', 'myntra', 'ajio', 'nykaa', 
                       'meesho', 'swiggy', 'zomato', 'bigbasket'],
            'keywords': ['order', 'delivery', 'shipped', 'delivered', 'cart',
                        'purchase', 'invoice', 'tracking', 'refund', 'return'],
        },
        EmailCategory.SOCIAL: {
            'senders': ['facebook', 'instagram', 'twitter', 'linkedin', 
                       'whatsapp', 'telegram'],
            'keywords': ['friend request', 'connection', 'message', 'tagged',
                        'mentioned', 'commented', 'liked', 'shared', 'follow'],
        },
        EmailCategory.WORK: {
            'senders': ['hr', 'recruitment', 'naukri', 'indeed', 'glassdoor',
                       'upwork', 'freelancer'],
            'keywords': ['job', 'interview', 'offer letter', 'salary', 'meeting',
                        'project', 'deadline', 'submission', 'application',
                        'resume', 'position', 'candidate'],
        },
        EmailCategory.NEWSLETTER: {
            'senders': ['digest', 'newsletter', 'medium', 'substack', 'quora'],
            'keywords': ['digest', 'newsletter', 'weekly', 'daily', 'subscribe',
                        'unsubscribe', 'edition', 'article', 'blog', 'read more'],
        },
        EmailCategory.PROMOTIONAL: {
            'senders': ['sale', 'offer', 'deal', 'promo'],
            'keywords': ['discount', 'offer', 'sale', 'coupon', 'cashback',
                        'limited time', 'exclusive', 'free', 'save', 'off%',
                        '% off', 'special offer'],
        },
    }
    
    # Transaction indicators
    TRANSACTION_KEYWORDS = [
        'debited', 'credited', 'payment', 'transfer', 'upi', 'neft', 'imps',
        'transaction', 'txn', 'rs.', '₹', 'inr'
    ]
    
    # LLM classification prompt template
    CLASSIFICATION_PROMPT = """You are an email classifier. Analyze this email and categorize it.

EMAIL:
Subject: {subject}
From: {sender}
Body: {body}

TASK:
Classify this email into exactly ONE category.

CATEGORIES:
- finance: Banks, payments, transactions, investments, credit cards, loans, UPI, wallets
- shopping: Orders, deliveries, purchases, e-commerce
- social: Social networks, personal messages, invitations
- work: Job-related, recruitment, office, meetings, projects
- newsletter: Digests, subscriptions, blogs, articles
- promotional: Marketing, offers, discounts, advertisements
- other: Anything that doesn't fit above

OUTPUT FORMAT (JSON only, no other text):
{{"category": "<category>", "confidence": "<high/medium/low>", "reason": "<brief 5-10 word reason>"}}
"""
    
    def __init__(self, use_llm: bool = False, model=None, tokenizer=None):
        """
        Initialize classifier.
        
        Args:
            use_llm: Whether to use LLM for classification
            model: Pre-loaded LLM model (required if use_llm=True)
            tokenizer: Pre-loaded tokenizer (required if use_llm=True)
        """
        self.use_llm = use_llm
        self.model = model
        self.tokenizer = tokenizer
    
    def classify(
        self, 
        subject: str, 
        sender: str, 
        body: str
    ) -> ClassificationResult:
        """
        Classify an email.
        
        Args:
            subject: Email subject
            sender: Sender name/email
            body: Email body text
            
        Returns:
            ClassificationResult with category and confidence
        """
        if self.use_llm and self.model is not None:
            return self._classify_llm(subject, sender, body)
        else:
            return self._classify_rules(subject, sender, body)
    
    def _classify_rules(
        self, 
        subject: str, 
        sender: str, 
        body: str
    ) -> ClassificationResult:
        """Classify using rule-based approach."""
        combined = f"{subject} {sender} {body}".lower()
        
        # Check for transaction first (subset of finance)
        is_transaction = any(kw in combined for kw in self.TRANSACTION_KEYWORDS)
        
        # Score each category
        scores: Dict[EmailCategory, int] = {}
        
        for category, patterns in self.CATEGORY_KEYWORDS.items():
            score = 0
            
            # Check sender patterns
            sender_lower = sender.lower()
            for pattern in patterns.get('senders', []):
                if pattern in sender_lower:
                    score += 3  # Sender match is strong signal
            
            # Check keyword patterns
            for keyword in patterns.get('keywords', []):
                if keyword in combined:
                    score += 1
            
            if score > 0:
                scores[category] = score
        
        if not scores:
            return ClassificationResult(
                category=EmailCategory.OTHER.value,
                confidence='low',
                reason='No matching patterns found',
                is_transaction=False
            )
        
        # Get highest scoring category
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]
        
        # Determine confidence based on score
        if best_score >= 5:
            confidence = 'high'
        elif best_score >= 3:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return ClassificationResult(
            category=best_category.value,
            confidence=confidence,
            reason=f"Matched {best_score} patterns for {best_category.value}",
            is_transaction=is_transaction
        )
    
    def _classify_llm(
        self, 
        subject: str, 
        sender: str, 
        body: str
    ) -> ClassificationResult:
        """Classify using LLM."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer required for LLM classification")
        
        try:
            from mlx_lm import generate
        except ImportError:
            raise ImportError("mlx_lm required for LLM classification")
        
        # Build prompt
        prompt = self.CLASSIFICATION_PROMPT.format(
            subject=subject[:200],
            sender=sender[:100],
            body=body[:2000]
        )
        
        # Generate response
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=100,
            verbose=False
        )
        
        # Parse response
        return self._parse_llm_response(response)
    
    def _parse_llm_response(self, response: str) -> ClassificationResult:
        """Parse LLM JSON response."""
        # Extract JSON from response
        match = re.search(r'\{[^{}]*\}', response)
        
        if match:
            try:
                data = json.loads(match.group())
                return ClassificationResult(
                    category=data.get('category', 'other'),
                    confidence=data.get('confidence', 'low'),
                    reason=data.get('reason', 'LLM classification'),
                    is_transaction=self._is_transaction_category(data.get('category', ''))
                )
            except json.JSONDecodeError:
                pass
        
        return ClassificationResult(
            category='other',
            confidence='low',
            reason='Failed to parse LLM response',
            is_transaction=False
        )
    
    def _is_transaction_category(self, category: str) -> bool:
        """Check if category indicates a transaction email."""
        return category.lower() == 'finance'
    
    def batch_classify(
        self, 
        emails: List[Dict]
    ) -> List[ClassificationResult]:
        """
        Classify multiple emails.
        
        Args:
            emails: List of dicts with 'subject', 'sender', 'body' keys
            
        Returns:
            List of ClassificationResults
        """
        results = []
        for email in emails:
            result = self.classify(
                subject=email.get('subject', ''),
                sender=email.get('sender', ''),
                body=email.get('body', '')
            )
            results.append(result)
        return results
    
    def is_finance_email(self, subject: str, sender: str, body: str) -> bool:
        """Quick check if email is finance-related."""
        result = self.classify(subject, sender, body)
        return result.category == EmailCategory.FINANCE.value
    
    def is_transaction_email(self, subject: str, sender: str, body: str) -> bool:
        """Quick check if email contains transaction data."""
        result = self.classify(subject, sender, body)
        return result.is_transaction


if __name__ == "__main__":
    # Test classification
    classifier = EmailClassifier(use_llm=False)
    
    test_emails = [
        {
            'subject': '❗ You have done a UPI txn. Check details!',
            'sender': 'HDFC Bank InstaAlerts',
            'body': 'Rs.2500.00 has been debited from account 3545 to VPA swiggy@ybl'
        },
        {
            'subject': 'Your Amazon order has shipped',
            'sender': 'Amazon.in',
            'body': 'Your order #123-456 has been shipped and will arrive by tomorrow'
        },
        {
            'subject': 'Weekly Python Digest',
            'sender': 'Python Weekly',
            'body': 'This week in Python: New features, tutorials, and more'
        },
    ]
    
    for email in test_emails:
        print(f"\nSubject: {email['subject']}")
        result = classifier.classify(**email)
        print(f"Category: {result.category}")
        print(f"Confidence: {result.confidence}")
        print(f"Is Transaction: {result.is_transaction}")
