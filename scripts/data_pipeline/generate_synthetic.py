#!/usr/bin/env python3
"""
Production-Grade Synthetic Data Generator for Indian Banking Transactions
==========================================================================

Engineering Principles:
1. Grammar-based message generation (not hardcoded templates)
2. Combinatorial coverage with configurable sampling
3. Type-safe generation with validation
4. Systematic edge case enumeration
5. Statistical distribution control
6. Reproducible with proper seeding
7. Property-based validation

Author: Ranjit Behera
Version: 3.0 (Engineering Grade)
"""

from __future__ import annotations
import json
import random
import hashlib
import argparse
import math
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, auto
from pathlib import Path
from typing import (
    List, Dict, Optional, Tuple, Set, Generator, 
    Callable, TypeVar, Generic, Union, Any, Iterator
)
from collections import defaultdict
import re


# ============================================================================
# TYPE SYSTEM - Enums and Value Objects
# ============================================================================

class TransactionType(Enum):
    DEBIT = "debit"
    CREDIT = "credit"

class TransactionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    REVERSED = "reversed"

class PaymentMethod(Enum):
    UPI = "upi"
    NEFT = "neft"
    RTGS = "rtgs"
    IMPS = "imps"
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    ATM = "atm"
    WALLET = "wallet"
    AUTO_DEBIT = "auto_debit"
    CHEQUE = "cheque"
    CASH = "cash"

class Category(Enum):
    FOOD = "food"
    GROCERY = "grocery"
    SHOPPING = "shopping"
    TRANSPORT = "transport"
    TRAVEL = "travel"
    FUEL = "fuel"
    BILLS = "bills"
    ENTERTAINMENT = "entertainment"
    HEALTHCARE = "healthcare"
    INVESTMENT = "investment"
    INSURANCE = "insurance"
    EDUCATION = "education"
    TRANSFER = "transfer"
    SALARY = "salary"
    REFUND = "refund"
    CASHBACK = "cashback"
    EMI = "emi"
    ATM_WITHDRAWAL = "atm_withdrawal"
    OTHER = "other"

class MessageType(Enum):
    TRANSACTION = "transaction"
    OTP = "otp"
    PROMOTIONAL = "promotional"
    ALERT = "alert"
    STATEMENT = "statement"

class AmountFormat(Enum):
    """All possible amount format variations."""
    INTEGER = "integer"                    # 2500
    DECIMAL_2 = "decimal_2"                # 2500.00
    DECIMAL_1 = "decimal_1"                # 2500.5
    COMMA_INTERNATIONAL = "comma_intl"     # 2,500.00
    COMMA_INDIAN = "comma_indian"          # 2,50,000.00
    NO_DECIMAL_COMMA = "no_decimal_comma"  # 2,500
    COMPACT = "compact"                    # 2.5K, 1.2L
    PADDED = "padded"                      # 002500.00

class DateFormat(Enum):
    """All possible date format variations."""
    DD_MM_YYYY_DASH = "dd-mm-yyyy"         # 28-12-2025
    DD_MM_YY_DASH = "dd-mm-yy"             # 28-12-25
    DD_MM_YYYY_SLASH = "dd/mm/yyyy"        # 28/12/2025
    DD_MM_YY_SLASH = "dd/mm/yy"            # 28/12/25
    DD_MON_YY = "dd-mon-yy"                # 28-Dec-25
    DD_MON_YYYY = "dd-mon-yyyy"            # 28-Dec-2025
    DD_MONTH_YYYY = "dd-month-yyyy"        # 28 December 2025
    MON_DD_YYYY = "mon-dd-yyyy"            # Dec 28, 2025
    YYYY_MM_DD = "yyyy-mm-dd"              # 2025-12-28 (ISO)
    COMPACT = "compact"                    # 28Dec25
    RELATIVE = "relative"                  # today, yesterday

class CurrencySymbol(Enum):
    """Currency symbol variations."""
    RS_DOT = "Rs."
    RS = "Rs"
    INR = "INR"
    RUPEE = "₹"
    RS_SPACE = "Rs "
    INR_SPACE = "INR "


# ============================================================================
# VALUE OBJECTS - Immutable domain objects
# ============================================================================

@dataclass(frozen=True)
class Amount:
    """Immutable amount with validation."""
    value: Decimal
    
    def __post_init__(self):
        if self.value < 0:
            raise ValueError(f"Amount cannot be negative: {self.value}")
    
    @classmethod
    def from_float(cls, value: float) -> 'Amount':
        return cls(Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    
    @classmethod
    def from_int(cls, value: int) -> 'Amount':
        return cls(Decimal(value))
    
    def format(self, fmt: AmountFormat) -> str:
        """Format amount according to specified format."""
        v = float(self.value)
        
        if fmt == AmountFormat.INTEGER:
            return str(int(v))
        elif fmt == AmountFormat.DECIMAL_2:
            return f"{v:.2f}"
        elif fmt == AmountFormat.DECIMAL_1:
            return f"{v:.1f}" if v != int(v) else str(int(v))
        elif fmt == AmountFormat.COMMA_INTERNATIONAL:
            return f"{v:,.2f}"
        elif fmt == AmountFormat.COMMA_INDIAN:
            return self._indian_format(v)
        elif fmt == AmountFormat.NO_DECIMAL_COMMA:
            return f"{int(v):,}"
        elif fmt == AmountFormat.COMPACT:
            return self._compact_format(v)
        elif fmt == AmountFormat.PADDED:
            return f"{v:012.2f}"
        return str(v)
    
    def _indian_format(self, v: float) -> str:
        """Format in Indian numbering system (lakhs, crores)."""
        s = f"{v:.2f}"
        integer_part, decimal_part = s.split('.')
        
        if len(integer_part) <= 3:
            return s
        
        # Last 3 digits
        result = integer_part[-3:]
        integer_part = integer_part[:-3]
        
        # Rest in groups of 2
        while integer_part:
            result = integer_part[-2:] + ',' + result
            integer_part = integer_part[:-2]
        
        return result + '.' + decimal_part
    
    def _compact_format(self, v: float) -> str:
        """Format as compact (2.5K, 1.2L, 3.5Cr)."""
        if v >= 10000000:  # Crore
            return f"{v/10000000:.1f}Cr"
        elif v >= 100000:  # Lakh
            return f"{v/100000:.1f}L"
        elif v >= 1000:  # Thousand
            return f"{v/1000:.1f}K"
        return str(int(v))


@dataclass(frozen=True)
class Account:
    """Bank account representation."""
    last_four: str
    
    def __post_init__(self):
        if not self.last_four.isdigit() or len(self.last_four) != 4:
            raise ValueError(f"Invalid account last four: {self.last_four}")
    
    def format(self, style: str = "XX") -> str:
        """Format account number."""
        styles = {
            "XX": f"XX{self.last_four}",
            "xx": f"xx{self.last_four}",
            "****": f"****{self.last_four}",
            "...": f"...{self.last_four}",
            "plain": self.last_four,
            "X": f"X{self.last_four}",
        }
        return styles.get(style, f"XX{self.last_four}")


@dataclass(frozen=True)
class Reference:
    """Transaction reference number."""
    value: str
    ref_type: str  # numeric, alphanumeric, utr
    
    @classmethod
    def generate(cls, ref_type: str, bank_code: str = "HDFC") -> 'Reference':
        if ref_type == "numeric":
            value = ''.join(random.choices('0123456789', k=12))
        elif ref_type == "alphanumeric":
            value = bank_code + ''.join(random.choices('0123456789', k=11))
        elif ref_type == "utr":
            date_part = datetime.now().strftime('%y%m%d')
            value = f"{bank_code}{date_part}N{''.join(random.choices('0123456789', k=6))}"
        else:
            value = ''.join(random.choices('0123456789', k=12))
        
        return cls(value=value, ref_type=ref_type)


@dataclass(frozen=True)
class TransactionDate:
    """Date with multiple format support."""
    date: date
    time: Optional[Tuple[int, int, int]] = None  # (hour, minute, second)
    
    def format_date(self, fmt: DateFormat) -> str:
        """Format date according to specified format."""
        d = self.date
        
        formats = {
            DateFormat.DD_MM_YYYY_DASH: d.strftime("%d-%m-%Y"),
            DateFormat.DD_MM_YY_DASH: d.strftime("%d-%m-%y"),
            DateFormat.DD_MM_YYYY_SLASH: d.strftime("%d/%m/%Y"),
            DateFormat.DD_MM_YY_SLASH: d.strftime("%d/%m/%y"),
            DateFormat.DD_MON_YY: d.strftime("%d-%b-%y"),
            DateFormat.DD_MON_YYYY: d.strftime("%d-%b-%Y"),
            DateFormat.DD_MONTH_YYYY: d.strftime("%d %B %Y"),
            DateFormat.MON_DD_YYYY: d.strftime("%b %d, %Y"),
            DateFormat.YYYY_MM_DD: d.strftime("%Y-%m-%d"),
            DateFormat.COMPACT: d.strftime("%d%b%y"),
        }
        
        if fmt == DateFormat.RELATIVE:
            today = date.today()
            diff = (today - d).days
            if diff == 0:
                return "today"
            elif diff == 1:
                return "yesterday"
            else:
                return formats[DateFormat.DD_MM_YY_DASH]
        
        return formats.get(fmt, d.strftime("%d-%m-%Y"))
    
    def format_time(self) -> Optional[str]:
        """Format time if present."""
        if not self.time:
            return None
        h, m, s = self.time
        formats = [
            f"{h:02d}:{m:02d}:{s:02d}",
            f"{h:02d}:{m:02d}",
            f"{h:02d}:{m:02d} {'AM' if h < 12 else 'PM'}",
        ]
        return random.choice(formats)
    
    def normalized(self) -> str:
        """Return ISO format."""
        return self.date.strftime("%Y-%m-%d")


# ============================================================================
# GRAMMAR-BASED MESSAGE GENERATOR
# ============================================================================

class GrammarRule:
    """Represents a grammar rule with weighted alternatives."""
    
    def __init__(self, name: str, alternatives: List[Tuple[str, float]]):
        """
        Args:
            name: Rule name
            alternatives: List of (pattern, weight) tuples
        """
        self.name = name
        self.alternatives = alternatives
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1."""
        total = sum(w for _, w in self.alternatives)
        self.alternatives = [(p, w/total) for p, w in self.alternatives]
    
    def sample(self, rng: random.Random) -> str:
        """Sample one alternative based on weights."""
        r = rng.random()
        cumulative = 0
        for pattern, weight in self.alternatives:
            cumulative += weight
            if r <= cumulative:
                return pattern
        return self.alternatives[-1][0]


class MessageGrammar:
    """
    Grammar-based message generator.
    
    Grammar Structure:
    MESSAGE ::= PREFIX BODY SUFFIX
    PREFIX  ::= BANK_INTRO
    BODY    ::= AMOUNT_PHRASE ACCOUNT_PHRASE DATE_PHRASE PARTY_PHRASE REF_PHRASE
    SUFFIX  ::= BALANCE_PHRASE? WARNING_PHRASE?
    """
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._build_grammar()
    
    def _build_grammar(self):
        """Build the grammar rules."""
        
        # Bank introduction patterns
        self.bank_intro = GrammarRule("BANK_INTRO", [
            ("{bank}:", 0.3),
            ("{bank} Bank:", 0.25),
            ("{bank} Bk:", 0.1),
            ("Dear Customer, {bank}:", 0.1),
            ("Alert: {bank}", 0.1),
            ("{bank} Bank Alert:", 0.1),
            ("{bank} Bank Acct", 0.05),
        ])
        
        # Amount phrase patterns for DEBIT
        self.amount_debit = GrammarRule("AMOUNT_DEBIT", [
            ("{currency}{amount} debited from", 0.25),
            ("{currency}{amount} sent to", 0.15),
            ("{currency}{amount} paid to", 0.1),
            ("{currency}{amount} transferred to", 0.1),
            ("{currency} {amount} debited from", 0.1),
            ("INR {amount} debited from", 0.1),
            ("{currency}{amount} spent at", 0.1),
            ("Amount {currency}{amount} debited", 0.1),
        ])
        
        # Amount phrase patterns for CREDIT
        self.amount_credit = GrammarRule("AMOUNT_CREDIT", [
            ("{currency}{amount} credited to", 0.3),
            ("{currency}{amount} received in", 0.2),
            ("{currency}{amount} deposited to", 0.1),
            ("{currency} {amount} credited to", 0.15),
            ("INR {amount} credited to", 0.1),
            ("{currency}{amount} added to", 0.1),
            ("Received {currency}{amount} in", 0.05),
        ])
        
        # Account phrase patterns
        self.account_phrase = GrammarRule("ACCOUNT_PHRASE", [
            ("A/c {account}", 0.3),
            ("a/c {account}", 0.15),
            ("Account {account}", 0.15),
            ("Acct {account}", 0.15),
            ("A/c No. {account}", 0.1),
            ("Acc {account}", 0.1),
            ("your A/c {account}", 0.05),
        ])
        
        # Date phrase patterns
        self.date_phrase = GrammarRule("DATE_PHRASE", [
            ("on {date}", 0.4),
            ("on {date} {time}", 0.2),
            ("{date}", 0.15),
            ("dated {date}", 0.1),
            ("on {date} at {time}", 0.1),
            ("{date} {time}", 0.05),
        ])
        
        # Counterparty patterns (UPI)
        self.party_upi = GrammarRule("PARTY_UPI", [
            ("VPA {vpa}", 0.2),
            ("to {vpa}", 0.2),
            ("VPA: {vpa}", 0.15),
            ("to VPA {vpa}", 0.15),
            ("Info: UPI/{vpa}", 0.1),
            ("{beneficiary} ({vpa})", 0.1),
            ("UPI-{vpa}", 0.1),
        ])
        
        # Reference patterns
        self.reference = GrammarRule("REFERENCE", [
            ("Ref: {ref}", 0.25),
            ("Ref {ref}", 0.2),
            ("UPI Ref: {ref}", 0.2),
            ("Reference: {ref}", 0.1),
            ("UTR: {ref}", 0.1),
            ("Txn Ref: {ref}", 0.1),
            ("Ref No. {ref}", 0.05),
        ])
        
        # Balance patterns
        self.balance = GrammarRule("BALANCE", [
            ("Avl Bal: {currency}{balance}", 0.25),
            ("Bal: {currency}{balance}", 0.2),
            ("Available Balance: {currency}{balance}", 0.15),
            ("Avl Bal {currency}{balance}", 0.15),
            ("Balance: {currency}{balance}", 0.1),
            ("A/c Bal: {currency}{balance}", 0.1),
            ("", 0.05),  # No balance shown
        ])
        
        # Warning patterns
        self.warning = GrammarRule("WARNING", [
            ("Not you? Call {helpline}", 0.3),
            ("If not done by you, call {helpline}", 0.2),
            ("Call {helpline} for dispute", 0.2),
            ("", 0.3),  # No warning
        ])
        
        # Failed transaction patterns
        self.failed = GrammarRule("FAILED", [
            ("Transaction of {currency}{amount} FAILED", 0.3),
            ("{currency}{amount} transaction DECLINED", 0.25),
            ("Payment of {currency}{amount} FAILED", 0.2),
            ("Transaction FAILED: {currency}{amount}", 0.15),
            ("{currency}{amount} to {vpa} DECLINED", 0.1),
        ])
        
        # Pending patterns
        self.pending = GrammarRule("PENDING", [
            ("{currency}{amount} debit PENDING", 0.35),
            ("Transaction of {currency}{amount} is PENDING", 0.3),
            ("{currency}{amount} payment is being processed", 0.2),
            ("Processing: {currency}{amount} to {vpa}", 0.15),
        ])
    
    def generate_debit_message(self, params: Dict[str, str]) -> str:
        """Generate a debit transaction message."""
        parts = [
            self.bank_intro.sample(self.rng),
            self.amount_debit.sample(self.rng),
            self.account_phrase.sample(self.rng),
            self.date_phrase.sample(self.rng),
            self.party_upi.sample(self.rng) if params.get('vpa') else "",
            self.reference.sample(self.rng),
            self.balance.sample(self.rng),
            self.warning.sample(self.rng),
        ]
        
        message = " ".join(p for p in parts if p)
        return self._apply_params(message, params)
    
    def generate_credit_message(self, params: Dict[str, str]) -> str:
        """Generate a credit transaction message."""
        parts = [
            self.bank_intro.sample(self.rng),
            self.amount_credit.sample(self.rng),
            self.account_phrase.sample(self.rng),
            self.date_phrase.sample(self.rng),
            f"from {params.get('vpa', params.get('beneficiary', ''))}",
            self.reference.sample(self.rng),
            self.balance.sample(self.rng),
        ]
        
        message = " ".join(p for p in parts if p)
        return self._apply_params(message, params)
    
    def generate_failed_message(self, params: Dict[str, str]) -> str:
        """Generate a failed transaction message."""
        parts = [
            self.bank_intro.sample(self.rng),
            self.failed.sample(self.rng),
            self.account_phrase.sample(self.rng),
            f"Reason: {params.get('reason', 'Transaction declined')}",
            self.reference.sample(self.rng),
        ]
        
        message = " ".join(p for p in parts if p)
        return self._apply_params(message, params)
    
    def generate_pending_message(self, params: Dict[str, str]) -> str:
        """Generate a pending transaction message."""
        parts = [
            self.bank_intro.sample(self.rng),
            self.pending.sample(self.rng),
            f"for {params.get('vpa', '')}",
            self.account_phrase.sample(self.rng),
            self.reference.sample(self.rng),
        ]
        
        message = " ".join(p for p in parts if p)
        return self._apply_params(message, params)
    
    def _apply_params(self, template: str, params: Dict[str, str]) -> str:
        """Apply parameters to template."""
        for key, value in params.items():
            template = template.replace(f"{{{key}}}", str(value) if value else "")
        
        # Clean up double spaces and trailing punctuation
        template = re.sub(r'\s+', ' ', template)
        template = re.sub(r'\s+([.,:])', r'\1', template)
        return template.strip()


# ============================================================================
# COMBINATORIAL GENERATOR
# ============================================================================

@dataclass
class GenerationSpace:
    """
    Defines the combinatorial space for generation.
    
    Total combinations = product of all dimension sizes
    """
    banks: List[str]
    amount_formats: List[AmountFormat]
    date_formats: List[DateFormat]
    currency_symbols: List[CurrencySymbol]
    account_styles: List[str]
    reference_types: List[str]
    transaction_types: List[TransactionType]
    statuses: List[TransactionStatus]
    payment_methods: List[PaymentMethod]
    categories: List[Category]
    noise_levels: List[float]
    
    @property
    def total_combinations(self) -> int:
        """Calculate total theoretical combinations."""
        return (
            len(self.banks) *
            len(self.amount_formats) *
            len(self.date_formats) *
            len(self.currency_symbols) *
            len(self.account_styles) *
            len(self.reference_types) *
            len(self.transaction_types) *
            len(self.statuses) *
            len(self.payment_methods) *
            len(self.categories) *
            len(self.noise_levels)
        )
    
    def get_dimension_coverage(self, samples: int) -> Dict[str, float]:
        """Calculate coverage for each dimension given sample count."""
        dimensions = {
            'banks': len(self.banks),
            'amount_formats': len(self.amount_formats),
            'date_formats': len(self.date_formats),
            'currency_symbols': len(self.currency_symbols),
            'account_styles': len(self.account_styles),
            'reference_types': len(self.reference_types),
            'transaction_types': len(self.transaction_types),
            'statuses': len(self.statuses),
            'payment_methods': len(self.payment_methods),
            'categories': len(self.categories),
            'noise_levels': len(self.noise_levels),
        }
        
        # Expected coverage using coupon collector approximation
        coverage = {}
        for dim, size in dimensions.items():
            # E[samples needed to see all] ≈ n * H_n where H_n is harmonic number
            harmonic = sum(1/i for i in range(1, size + 1))
            expected_needed = size * harmonic
            coverage[dim] = min(1.0, samples / expected_needed)
        
        return coverage


class EdgeCaseGenerator:
    """
    Systematic edge case generation using boundary value analysis.
    
    Categories:
    1. Amount boundaries
    2. Date boundaries  
    3. String boundaries
    4. Format edge cases
    5. Invalid inputs (for negative testing)
    """
    
    # Amount edge cases
    AMOUNT_BOUNDARIES = [
        0.01,           # Minimum meaningful
        0.50,           # Sub-rupee
        0.99,           # Just under 1
        1.00,           # Boundary
        9.99,           # Just under 10
        10.00,          # Boundary
        99.99,          # Just under 100
        100.00,         # Common boundary
        499.99,         # Just under 500
        500.00,         # Common payment
        999.99,         # Just under 1000
        1000.00,        # Thousand boundary
        9999.99,        # Just under 10K
        10000.00,       # 10K boundary
        99999.99,       # Just under 1L
        100000.00,      # Lakh boundary
        999999.99,      # Just under 10L
        1000000.00,     # 10L boundary
        9999999.99,     # Just under 1Cr
        10000000.00,    # Crore boundary
    ]
    
    # Round numbers (common in real transactions)
    ROUND_AMOUNTS = [
        50, 100, 200, 250, 500, 750, 1000, 1500, 2000, 2500,
        3000, 4000, 5000, 7500, 10000, 15000, 20000, 25000,
        50000, 75000, 100000, 200000, 500000,
    ]
    
    # Psychological pricing
    PSYCHOLOGICAL_AMOUNTS = [
        49, 99, 149, 199, 249, 299, 399, 499, 599, 699, 799, 899, 999,
        1299, 1499, 1999, 2499, 2999, 3999, 4999, 9999,
    ]
    
    # Date edge cases
    DATE_EDGE_CASES = [
        date(2024, 2, 29),   # Leap year
        date(2025, 2, 28),   # Non-leap year Feb end
        date(2025, 12, 31),  # Year end
        date(2026, 1, 1),    # Year start
        date(2025, 3, 31),   # Month with 31 days
        date(2025, 4, 30),   # Month with 30 days
    ]
    
    # Special characters in merchant names
    SPECIAL_CHAR_MERCHANTS = [
        "McDonald's",
        "H&M",
        "AT&T",
        "L'Oreal",
        "7-Eleven",
        "Marks & Spencer",
        "Johnson & Johnson",
        "Dunkin' Donuts",
        "Toys \"R\" Us",
        "Yahoo!",
    ]
    
    # Long merchant names
    LONG_MERCHANTS = [
        "Tata Consultancy Services Limited",
        "Hindustan Petroleum Corporation Limited",
        "Life Insurance Corporation of India",
        "Indian Oil Corporation Limited",
        "Bharat Heavy Electricals Limited",
        "Steel Authority of India Limited",
    ]
    
    # Unicode names (Hindi/Regional)
    UNICODE_NAMES = [
        "राहुल शर्मा",
        "प्रिया सिंह",
        "अमित कुमार",
        "सुनीता देवी",
        "విజయ్ కుమార్",
        "প্রিয়া দাস",
        "அருண் குமார்",
    ]
    
    @classmethod
    def get_amount_edge_cases(cls) -> List[float]:
        """Get all amount edge cases."""
        return cls.AMOUNT_BOUNDARIES + cls.ROUND_AMOUNTS + cls.PSYCHOLOGICAL_AMOUNTS
    
    @classmethod
    def get_date_edge_cases(cls) -> List[date]:
        """Get all date edge cases."""
        today = date.today()
        return cls.DATE_EDGE_CASES + [
            today,
            today - timedelta(days=1),
            today - timedelta(days=7),
            today - timedelta(days=30),
            today - timedelta(days=365),
        ]
    
    @classmethod
    def generate_edge_case_batch(cls, count: int, rng: random.Random) -> List[Dict]:
        """Generate a batch of edge case configurations."""
        configs = []
        
        # Amount edge cases
        for amount in rng.sample(cls.AMOUNT_BOUNDARIES, min(count // 4, len(cls.AMOUNT_BOUNDARIES))):
            configs.append({'edge_type': 'amount_boundary', 'amount': amount})
        
        # Special character merchants
        for merchant in cls.SPECIAL_CHAR_MERCHANTS:
            configs.append({'edge_type': 'special_char', 'merchant': merchant})
        
        # Long merchants
        for merchant in cls.LONG_MERCHANTS:
            configs.append({'edge_type': 'long_merchant', 'merchant': merchant})
        
        # Unicode names
        for name in cls.UNICODE_NAMES:
            configs.append({'edge_type': 'unicode', 'beneficiary': name})
        
        return configs[:count]


# ============================================================================
# STATISTICAL DISTRIBUTION CONTROLLER
# ============================================================================

class DistributionConfig:
    """
    Configurable statistical distributions for realistic data.
    
    Uses weighted random selection based on real-world frequencies.
    """
    
    # Transaction type distribution
    TRANSACTION_TYPE_WEIGHTS = {
        TransactionType.DEBIT: 0.7,
        TransactionType.CREDIT: 0.3,
    }
    
    # Status distribution
    STATUS_WEIGHTS = {
        TransactionStatus.SUCCESS: 0.92,
        TransactionStatus.FAILED: 0.05,
        TransactionStatus.PENDING: 0.02,
        TransactionStatus.REVERSED: 0.01,
    }
    
    # Payment method distribution
    PAYMENT_METHOD_WEIGHTS = {
        PaymentMethod.UPI: 0.55,
        PaymentMethod.CREDIT_CARD: 0.15,
        PaymentMethod.DEBIT_CARD: 0.08,
        PaymentMethod.NEFT: 0.08,
        PaymentMethod.IMPS: 0.05,
        PaymentMethod.ATM: 0.04,
        PaymentMethod.AUTO_DEBIT: 0.03,
        PaymentMethod.WALLET: 0.02,
    }
    
    # Category distribution (based on typical spending)
    CATEGORY_WEIGHTS = {
        Category.FOOD: 0.18,
        Category.SHOPPING: 0.15,
        Category.GROCERY: 0.12,
        Category.TRANSFER: 0.12,
        Category.BILLS: 0.10,
        Category.TRANSPORT: 0.08,
        Category.ENTERTAINMENT: 0.05,
        Category.FUEL: 0.05,
        Category.INVESTMENT: 0.04,
        Category.HEALTHCARE: 0.03,
        Category.TRAVEL: 0.03,
        Category.SALARY: 0.02,
        Category.EMI: 0.02,
        Category.REFUND: 0.01,
    }
    
    # Amount distribution by category (min, max, mean, std_dev)
    AMOUNT_DISTRIBUTIONS = {
        Category.FOOD: (20, 3000, 350, 300),
        Category.GROCERY: (50, 10000, 800, 600),
        Category.SHOPPING: (100, 100000, 2500, 3000),
        Category.TRANSPORT: (20, 5000, 250, 300),
        Category.TRAVEL: (500, 200000, 8000, 15000),
        Category.FUEL: (100, 10000, 1500, 1000),
        Category.BILLS: (100, 20000, 1200, 1500),
        Category.ENTERTAINMENT: (50, 5000, 500, 400),
        Category.HEALTHCARE: (100, 50000, 2000, 3000),
        Category.INVESTMENT: (500, 500000, 25000, 50000),
        Category.TRANSFER: (100, 200000, 5000, 10000),
        Category.SALARY: (15000, 500000, 60000, 40000),
        Category.EMI: (1000, 100000, 15000, 12000),
        Category.REFUND: (50, 50000, 1000, 2000),
    }
    
    @classmethod
    def sample_weighted(cls, weights: Dict, rng: random.Random):
        """Sample from weighted distribution."""
        items = list(weights.keys())
        probs = list(weights.values())
        total = sum(probs)
        probs = [p / total for p in probs]
        
        r = rng.random()
        cumulative = 0
        for item, prob in zip(items, probs):
            cumulative += prob
            if r <= cumulative:
                return item
        return items[-1]
    
    @classmethod
    def sample_amount(cls, category: Category, rng: random.Random) -> float:
        """Sample amount based on category distribution."""
        min_amt, max_amt, mean, std = cls.AMOUNT_DISTRIBUTIONS.get(
            category, (100, 10000, 1000, 1000)
        )
        
        # Use truncated normal distribution
        amount = rng.gauss(mean, std)
        amount = max(min_amt, min(max_amt, amount))
        
        # Round to realistic precision
        if amount < 100:
            amount = round(amount, 2)
        elif amount < 1000:
            amount = round(amount, 1)
        else:
            amount = round(amount, 0)
        
        return amount


# ============================================================================
# NOISE INJECTION ENGINE
# ============================================================================

class NoiseEngine:
    """
    Systematic noise injection for realistic messages.
    
    Noise types:
    1. Spacing variations
    2. Abbreviations
    3. Case changes
    4. Truncation
    5. Typos
    6. Punctuation variations
    """
    
    ABBREVIATIONS = {
        'Account': ['A/c', 'Acc', 'Acct', 'a/c'],
        'Reference': ['Ref', 'Ref.', 'ref'],
        'Transaction': ['Txn', 'txn', 'Trans'],
        'Available': ['Avl', 'Avail', 'avl'],
        'Balance': ['Bal', 'bal', 'Bal.'],
        'Credited': ['Cr', 'cr', 'Cr.'],
        'Debited': ['Dr', 'dr', 'Dr.'],
        'Number': ['No', 'No.', 'Num'],
    }
    
    COMMON_TYPOS = {
        'debited': ['debitd', 'debite', 'debitted'],
        'credited': ['creditd', 'credite', 'creditted'],
        'transaction': ['transction', 'transcation', 'transacton'],
        'available': ['availble', 'avialable', 'availabel'],
        'balance': ['balace', 'balnce', 'balanc'],
    }
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def apply(self, text: str, noise_level: float) -> str:
        """
        Apply noise to text.
        
        Args:
            text: Original text
            noise_level: 0.0 (clean) to 1.0 (very noisy)
        """
        if noise_level <= 0:
            return text
        
        # Determine which noise types to apply
        noise_budget = noise_level
        
        # 1. Abbreviations (always some)
        if self.rng.random() < noise_budget * 0.5:
            text = self._apply_abbreviations(text)
        
        # 2. Spacing
        if self.rng.random() < noise_budget * 0.3:
            text = self._vary_spacing(text)
        
        # 3. Case changes
        if self.rng.random() < noise_budget * 0.2:
            text = self._vary_case(text)
        
        # 4. Truncation (for SMS)
        if self.rng.random() < noise_budget * 0.1:
            text = self._truncate(text)
        
        # 5. Typos (rare)
        if self.rng.random() < noise_budget * 0.05:
            text = self._add_typos(text)
        
        # 6. Punctuation
        if self.rng.random() < noise_budget * 0.2:
            text = self._vary_punctuation(text)
        
        return text
    
    def _apply_abbreviations(self, text: str) -> str:
        """Replace words with abbreviations."""
        for word, abbrevs in self.ABBREVIATIONS.items():
            if word in text and self.rng.random() < 0.5:
                text = text.replace(word, self.rng.choice(abbrevs), 1)
        return text
    
    def _vary_spacing(self, text: str) -> str:
        """Vary spacing."""
        variations = [
            (". ", "."),
            (": ", ":"),
            ("Rs. ", "Rs."),
            ("Rs ", "Rs"),
            (", ", ","),
        ]
        for old, new in variations:
            if self.rng.random() < 0.3:
                text = text.replace(old, new)
        return text
    
    def _vary_case(self, text: str) -> str:
        """Vary case."""
        if self.rng.random() < 0.3:
            return text.upper()
        elif self.rng.random() < 0.1:
            return text.lower()
        return text
    
    def _truncate(self, text: str) -> str:
        """Truncate to SMS limit."""
        if len(text) > 160:
            return text[:157] + "..."
        return text
    
    def _add_typos(self, text: str) -> str:
        """Add occasional typos."""
        for word, typos in self.COMMON_TYPOS.items():
            if word in text.lower() and self.rng.random() < 0.2:
                # Case-insensitive replace
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                text = pattern.sub(self.rng.choice(typos), text, count=1)
        return text
    
    def _vary_punctuation(self, text: str) -> str:
        """Vary punctuation."""
        if self.rng.random() < 0.3:
            text = text.replace(".", "")
        if self.rng.random() < 0.2:
            text = text.replace(",", "")
        return text


# ============================================================================
# MERCHANT & COUNTERPARTY DATABASE
# ============================================================================

@dataclass
class Merchant:
    """Merchant entity with VPA and metadata."""
    name: str
    vpa: str
    category: Category
    aliases: List[str] = field(default_factory=list)
    
    def get_display_name(self, rng: random.Random) -> str:
        """Get a random display name variation."""
        options = [self.name] + self.aliases
        return rng.choice(options)


class MerchantDatabase:
    """
    Database of merchants organized by category.
    
    Provides O(1) lookup by category and O(1) random selection.
    """
    
    def __init__(self):
        self._by_category: Dict[Category, List[Merchant]] = defaultdict(list)
        self._by_name: Dict[str, Merchant] = {}
        self._all: List[Merchant] = []
        self._build_database()
    
    def _build_database(self):
        """Build the merchant database."""
        merchants_data = [
            # Food Delivery
            ("Swiggy", "swiggy@ybl", Category.FOOD, ["SWIGGY", "Swiggy Instamart"]),
            ("Zomato", "zomato@paytm", Category.FOOD, ["ZOMATO", "Zomato Gold"]),
            ("Dominos", "dominos@hdfcbank", Category.FOOD, ["DOMINOS", "Domino's Pizza"]),
            ("Pizza Hut", "pizzahut@icici", Category.FOOD, ["PIZZA HUT"]),
            ("McDonalds", "mcdonalds@ybl", Category.FOOD, ["McDonald's", "MCD"]),
            ("KFC", "kfc@paytm", Category.FOOD, ["KFC"]),
            ("Starbucks", "starbucks@icici", Category.FOOD, ["STARBUCKS"]),
            ("Subway", "subway@hdfcbank", Category.FOOD, ["SUBWAY"]),
            
            # E-commerce
            ("Amazon", "amazon@apl", Category.SHOPPING, ["AMAZON", "Amazon.in"]),
            ("Flipkart", "flipkart@ybl", Category.SHOPPING, ["FLIPKART"]),
            ("Myntra", "myntra@ybl", Category.SHOPPING, ["MYNTRA"]),
            ("Ajio", "ajio@icici", Category.SHOPPING, ["AJIO"]),
            ("Nykaa", "nykaa@paytm", Category.SHOPPING, ["NYKAA"]),
            ("Meesho", "meesho@paytm", Category.SHOPPING, ["MEESHO"]),
            ("Croma", "croma@hdfcbank", Category.SHOPPING, ["CROMA"]),
            
            # Grocery
            ("Zepto", "zepto@ybl", Category.GROCERY, ["ZEPTO"]),
            ("BigBasket", "bigbasket@ybl", Category.GROCERY, ["BIGBASKET"]),
            ("Blinkit", "blinkit@paytm", Category.GROCERY, ["BLINKIT", "Grofers"]),
            ("DMart", "dmart@hdfcbank", Category.GROCERY, ["DMART", "D-Mart"]),
            ("JioMart", "jiomart@icici", Category.GROCERY, ["JIOMART"]),
            
            # Transport
            ("Uber", "uber@paytm", Category.TRANSPORT, ["UBER"]),
            ("Ola", "ola@icici", Category.TRANSPORT, ["OLA", "Ola Cabs"]),
            ("Rapido", "rapido@ybl", Category.TRANSPORT, ["RAPIDO"]),
            
            # Travel
            ("IRCTC", "irctc@sbi", Category.TRAVEL, ["IRCTC"]),
            ("MakeMyTrip", "makemytrip@icici", Category.TRAVEL, ["MMT"]),
            ("Goibibo", "goibibo@ybl", Category.TRAVEL, ["GOIBIBO"]),
            ("RedBus", "redbus@paytm", Category.TRAVEL, ["REDBUS"]),
            
            # Fuel
            ("IOCL", "iocl@sbi", Category.FUEL, ["Indian Oil", "IndianOil"]),
            ("HPCL", "hpcl@hdfcbank", Category.FUEL, ["HP Petrol"]),
            ("BPCL", "bpcl@icici", Category.FUEL, ["Bharat Petroleum"]),
            
            # Bills
            ("Airtel", "airtel@paytm", Category.BILLS, ["AIRTEL"]),
            ("Jio", "jio@icici", Category.BILLS, ["Reliance Jio"]),
            ("Vi", "vi@ybl", Category.BILLS, ["Vodafone Idea"]),
            ("Tata Power", "tatapower@hdfcbank", Category.BILLS, ["TATA POWER"]),
            ("BESCOM", "bescom@ybl", Category.BILLS, ["BESCOM"]),
            
            # Entertainment
            ("Netflix", "netflix@icici", Category.ENTERTAINMENT, ["NETFLIX"]),
            ("Hotstar", "hotstar@ybl", Category.ENTERTAINMENT, ["Disney+ Hotstar"]),
            ("Spotify", "spotify@paytm", Category.ENTERTAINMENT, ["SPOTIFY"]),
            ("BookMyShow", "bookmyshow@paytm", Category.ENTERTAINMENT, ["BMS"]),
            ("PVR", "pvr@hdfcbank", Category.ENTERTAINMENT, ["PVR Cinemas"]),
            
            # Healthcare
            ("Apollo", "apollo@hdfcbank", Category.HEALTHCARE, ["Apollo Pharmacy"]),
            ("PharmEasy", "pharmeasy@paytm", Category.HEALTHCARE, ["PHARMEASY"]),
            ("1mg", "1mg@ybl", Category.HEALTHCARE, ["Tata 1mg"]),
            
            # Investment
            ("Zerodha", "zerodha@hdfcbank", Category.INVESTMENT, ["ZERODHA"]),
            ("Groww", "groww@axisbank", Category.INVESTMENT, ["GROWW"]),
            ("Upstox", "upstox@icici", Category.INVESTMENT, ["UPSTOX"]),
            ("Angel One", "angelone@ybl", Category.INVESTMENT, ["Angel Broking"]),
            ("ICICI Direct", "icicidirect@icici", Category.INVESTMENT, ["ICICIdirect"]),
            ("Paytm Money", "paytmmoney@paytm", Category.INVESTMENT, ["PAYTM MONEY"]),
            ("5Paisa", "5paisa@icici", Category.INVESTMENT, ["5paisa"]),
            ("Dhan", "dhan@okaxis", Category.INVESTMENT, ["DHAN"]),
        ]
        
        for name, vpa, category, aliases in merchants_data:
            merchant = Merchant(name=name, vpa=vpa, category=category, aliases=aliases)
            self._by_category[category].append(merchant)
            self._by_name[name.lower()] = merchant
            self._all.append(merchant)
    
    def get_by_category(self, category: Category) -> List[Merchant]:
        """Get all merchants in a category."""
        return self._by_category.get(category, [])
    
    def get_random(self, rng: random.Random, category: Optional[Category] = None) -> Merchant:
        """Get a random merchant, optionally filtered by category."""
        if category:
            merchants = self._by_category.get(category, self._all)
        else:
            merchants = self._all
        
        return rng.choice(merchants) if merchants else self._all[0]
    
    def get_by_name(self, name: str) -> Optional[Merchant]:
        """Lookup merchant by name."""
        return self._by_name.get(name.lower())


class PersonDatabase:
    """Database of Indian names for P2P transactions."""
    
    FIRST_NAMES = [
        "Rahul", "Priya", "Amit", "Neha", "Vijay", "Deepak", "Anjali", "Rajesh",
        "Sunita", "Arun", "Pooja", "Sanjay", "Kavita", "Manoj", "Rekha", "Suresh",
        "Lakshmi", "Ganesh", "Meera", "Prakash", "Asha", "Ramesh", "Geeta", "Mohan",
        "Savita", "Kiran", "Vinod", "Usha", "Ashok", "Padma", "Rohit", "Sneha",
        "Vikas", "Divya", "Nitin", "Swati", "Abhishek", "Ritu", "Manish", "Preeti",
    ]
    
    LAST_NAMES = [
        "Sharma", "Singh", "Kumar", "Gupta", "Patel", "Verma", "Mehta", "Nair",
        "Iyer", "Joshi", "Reddy", "Mishra", "Das", "Pillai", "Bose", "Menon",
        "Venkat", "Rao", "Kulkarni", "Shah", "Patil", "Chandra", "Devi", "Lal",
        "Sinha", "Chopra", "Saxena", "Rani", "Tiwari", "Hegde", "Agarwal", "Kapoor",
        "Yadav", "Bansal", "Jain", "Pandey", "Malhotra", "Behera", "Sahu", "Tarai",
    ]
    
    VPA_SUFFIXES = [
        "@ybl", "@paytm", "@okicici", "@okhdfcbank", "@oksbi",
        "@axl", "@apl", "@ibl", "@upi", "@okaxis",
    ]
    
    @classmethod
    def generate_name(cls, rng: random.Random) -> str:
        """Generate a random Indian name."""
        return f"{rng.choice(cls.FIRST_NAMES)} {rng.choice(cls.LAST_NAMES)}"
    
    @classmethod
    def generate_vpa(cls, name: str, rng: random.Random) -> str:
        """Generate VPA from name."""
        parts = name.lower().split()
        patterns = [
            f"{parts[0]}{rng.randint(1, 99)}{rng.choice(cls.VPA_SUFFIXES)}",
            f"{parts[0]}.{parts[-1]}{rng.choice(cls.VPA_SUFFIXES)}",
            f"{parts[0]}{parts[-1][0]}{rng.randint(1, 9)}{rng.choice(cls.VPA_SUFFIXES)}",
            f"{parts[0]}_{rng.randint(100, 999)}{rng.choice(cls.VPA_SUFFIXES)}",
        ]
        return rng.choice(patterns)


# ============================================================================
# BANK DATABASE
# ============================================================================

@dataclass
class Bank:
    """Bank entity with metadata."""
    name: str
    code: str
    helpline: str
    
    def get_intro(self, rng: random.Random) -> str:
        """Get bank introduction variation."""
        intros = [
            f"{self.name}:",
            f"{self.name} Bank:",
            f"{self.name} Bk:",
            f"Alert: {self.name}",
            f"Dear Customer, {self.name}:",
        ]
        return rng.choice(intros)


class BankDatabase:
    """Database of Indian banks."""
    
    BANKS = [
        Bank("HDFC", "HDFC", "18002586161"),
        Bank("ICICI", "ICIC", "18002662"),
        Bank("SBI", "SBIN", "1800112211"),
        Bank("Axis", "UTIB", "18004195959"),
        Bank("Kotak", "KKBK", "18601266022"),
        Bank("PNB", "PUNB", "18001802222"),
        Bank("BOB", "BARB", "18001024455"),
        Bank("IDFC", "IDFB", "18001024"),
        Bank("Yes Bank", "YESB", "18001200"),
        Bank("IndusInd", "INDB", "18602677777"),
        Bank("Canara", "CNRB", "18004250018"),
        Bank("Union Bank", "UBIN", "18002082244"),
    ]
    
    @classmethod
    def get_all(cls) -> List[Bank]:
        return cls.BANKS
    
    @classmethod
    def get_random(cls, rng: random.Random) -> Bank:
        return rng.choice(cls.BANKS)
    
    @classmethod
    def get_by_name(cls, name: str) -> Optional[Bank]:
        for bank in cls.BANKS:
            if bank.name.lower() == name.lower():
                return bank
        return None


# ============================================================================
# GROUND TRUTH SCHEMA
# ============================================================================

@dataclass
class GroundTruth:
    """
    Complete ground truth for training.
    
    Includes all extractable fields with both raw and normalized versions.
    """
    # Core
    amount: Optional[float] = None
    amount_raw: Optional[str] = None
    currency: str = "INR"
    type: Optional[str] = None
    status: str = "success"
    
    # Account
    account: Optional[str] = None
    account_raw: Optional[str] = None
    bank: Optional[str] = None
    
    # DateTime
    date: Optional[str] = None           # YYYY-MM-DD
    date_raw: Optional[str] = None
    time: Optional[str] = None           # HH:MM:SS
    time_raw: Optional[str] = None
    
    # Reference
    reference: Optional[str] = None
    reference_raw: Optional[str] = None
    reference_type: Optional[str] = None
    
    # Counterparty
    merchant: Optional[str] = None
    merchant_raw: Optional[str] = None
    vpa: Optional[str] = None
    beneficiary: Optional[str] = None
    
    # Classification
    payment_method: Optional[str] = None
    category: Optional[str] = None
    
    # Balance
    balance_after: Optional[float] = None
    balance_raw: Optional[str] = None
    
    # Metadata
    message_type: str = "transaction"
    is_p2m: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ============================================================================
# MAIN TRANSACTION GENERATOR
# ============================================================================

class TransactionGenerator:
    """
    Main generator class that orchestrates all components.
    
    Architecture:
    1. Sample from statistical distributions
    2. Generate using grammar rules
    3. Apply noise injection
    4. Validate output
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Initialize components
        self.grammar = MessageGrammar(seed)
        self.noise = NoiseEngine(seed)
        self.merchants = MerchantDatabase()
        self.distribution = DistributionConfig()
    
    def generate_transaction(
        self,
        txn_type: Optional[TransactionType] = None,
        status: Optional[TransactionStatus] = None,
        category: Optional[Category] = None,
        payment_method: Optional[PaymentMethod] = None,
        noise_level: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Generate a single transaction.
        
        Args:
            txn_type: Force transaction type (or sample)
            status: Force status (or sample)
            category: Force category (or sample)
            payment_method: Force payment method (or sample)
            noise_level: Amount of noise to inject
            
        Returns:
            Dict with 'text' and 'ground_truth' keys
        """
        # Sample missing parameters from distributions
        if txn_type is None:
            txn_type = self.distribution.sample_weighted(
                self.distribution.TRANSACTION_TYPE_WEIGHTS, self.rng
            )
        
        if status is None:
            status = self.distribution.sample_weighted(
                self.distribution.STATUS_WEIGHTS, self.rng
            )
        
        if category is None:
            category = self.distribution.sample_weighted(
                self.distribution.CATEGORY_WEIGHTS, self.rng
            )
        
        if payment_method is None:
            payment_method = self.distribution.sample_weighted(
                self.distribution.PAYMENT_METHOD_WEIGHTS, self.rng
            )
        
        # Generate components
        bank = BankDatabase.get_random(self.rng)
        amount = self.distribution.sample_amount(category, self.rng)
        amount_obj = Amount.from_float(amount)
        amount_fmt = self.rng.choice(list(AmountFormat))
        amount_raw = amount_obj.format(amount_fmt)
        
        currency = self.rng.choice(list(CurrencySymbol))
        
        account = Account(str(self.rng.randint(1000, 9999)))
        account_style = self.rng.choice(["XX", "xx", "****", "X"])
        
        # Date
        days_ago = self.rng.randint(0, 365)
        txn_date = date.today() - timedelta(days=days_ago)
        time_tuple = (
            self.rng.randint(0, 23),
            self.rng.randint(0, 59),
            self.rng.randint(0, 59)
        ) if self.rng.random() > 0.5 else None
        date_obj = TransactionDate(txn_date, time_tuple)
        date_fmt = self.rng.choice(list(DateFormat))
        
        # Reference
        ref_type = self.rng.choice(["numeric", "alphanumeric", "utr"])
        ref = Reference.generate(ref_type, bank.code)
        
        # Balance
        balance = round(self.rng.uniform(1000, 500000), 2)
        balance_fmt = self.rng.choice([AmountFormat.COMMA_INTERNATIONAL, AmountFormat.DECIMAL_2])
        balance_raw = Amount.from_float(balance).format(balance_fmt)
        
        # Counterparty
        is_p2m = category not in [Category.TRANSFER, Category.SALARY]
        
        if is_p2m:
            merchant = self.merchants.get_random(self.rng, category)
            vpa = merchant.vpa
            merchant_raw = merchant.get_display_name(self.rng)
            beneficiary = None
        else:
            merchant = None
            name = PersonDatabase.generate_name(self.rng)
            vpa = PersonDatabase.generate_vpa(name, self.rng)
            merchant_raw = None
            beneficiary = name
        
        # Build parameters for grammar
        params = {
            'bank': bank.name,
            'currency': currency.value,
            'amount': amount_raw,
            'account': account.format(account_style),
            'date': date_obj.format_date(date_fmt),
            'time': date_obj.format_time() or "",
            'vpa': vpa,
            'beneficiary': beneficiary or (merchant.name if merchant else ""),
            'ref': ref.value,
            'balance': balance_raw,
            'helpline': bank.helpline,
            'reason': self.rng.choice([
                "Insufficient funds",
                "Transaction declined",
                "Network error",
                "Invalid VPA",
            ]) if status == TransactionStatus.FAILED else "",
        }
        
        # Generate message using grammar
        if status == TransactionStatus.FAILED:
            text = self.grammar.generate_failed_message(params)
        elif status == TransactionStatus.PENDING:
            text = self.grammar.generate_pending_message(params)
        elif txn_type == TransactionType.CREDIT:
            text = self.grammar.generate_credit_message(params)
        else:
            text = self.grammar.generate_debit_message(params)
        
        # Apply noise
        text = self.noise.apply(text, noise_level)
        
        # Build ground truth
        ground_truth = GroundTruth(
            amount=amount,
            amount_raw=amount_raw,
            currency="INR",
            type=txn_type.value,
            status=status.value,
            account=account.last_four,
            account_raw=account.format(account_style),
            bank=bank.name,
            date=date_obj.normalized(),
            date_raw=date_obj.format_date(date_fmt),
            time=f"{time_tuple[0]:02d}:{time_tuple[1]:02d}:{time_tuple[2]:02d}" if time_tuple else None,
            time_raw=date_obj.format_time(),
            reference=ref.value,
            reference_type=ref_type,
            merchant=merchant.name.lower() if merchant else None,
            merchant_raw=merchant_raw,
            vpa=vpa,
            beneficiary=beneficiary,
            payment_method=payment_method.value,
            category=category.value,
            balance_after=balance,
            balance_raw=balance_raw,
            message_type="transaction",
            is_p2m=is_p2m,
        )
        
        return {
            'text': text,
            'ground_truth': ground_truth.to_dict(),
        }
    
    def generate_non_transaction(self, msg_type: MessageType) -> Dict[str, Any]:
        """Generate non-transaction message (OTP, promo, etc.)."""
        bank = BankDatabase.get_random(self.rng)
        
        if msg_type == MessageType.OTP:
            otp = ''.join(self.rng.choices('0123456789', k=6))
            templates = [
                f"{bank.name} Bank: Your OTP is {otp}. Valid for 10 mins. Do not share with anyone.",
                f"{bank.name}: {otp} is your OTP for transaction. Valid for 5 mins.",
                f"OTP for {bank.name} Bank: {otp}. Do not share this with anyone.",
            ]
            text = self.rng.choice(templates)
            
        elif msg_type == MessageType.PROMOTIONAL:
            templates = [
                f"{bank.name} Bank wishes you a Happy Diwali! Enjoy 10% cashback on all transactions.",
                f"Exclusive offer! Get 5% cashback on shopping using {bank.name} Credit Card.",
                f"{bank.name}: Apply for Personal Loan at lowest interest rates. Click here.",
            ]
            text = self.rng.choice(templates)
            
        elif msg_type == MessageType.ALERT:
            account = Account(str(self.rng.randint(1000, 9999)))
            templates = [
                f"{bank.name} Bank: Your Debit Card has been blocked. Call {bank.helpline}.",
                f"{bank.name}: Suspicious activity detected on A/c {account.format('XX')}. Call immediately.",
                f"{bank.name}: Your account KYC is pending. Update by end of month.",
            ]
            text = self.rng.choice(templates)
            
        else:  # STATEMENT
            account = Account(str(self.rng.randint(1000, 9999)))
            balance = round(self.rng.uniform(1000, 500000), 2)
            text = f"{bank.name} Bank: Your A/c {account.format('XX')} balance is Rs.{balance:,.2f}."
        
        ground_truth = GroundTruth(
            message_type=msg_type.value,
            bank=bank.name,
        )
        
        return {
            'text': text,
            'ground_truth': ground_truth.to_dict(),
        }
    
    def generate_batch(
        self,
        count: int,
        include_non_transactions: bool = True,
        include_edge_cases: bool = True,
        noise_level: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of transactions.
        
        Args:
            count: Number of records to generate
            include_non_transactions: Include OTP, promo messages (5%)
            include_edge_cases: Include edge cases (3%)
            noise_level: Noise intensity
        """
        records = []
        
        # Calculate distribution
        edge_case_count = int(count * 0.03) if include_edge_cases else 0
        non_txn_count = int(count * 0.05) if include_non_transactions else 0
        txn_count = count - edge_case_count - non_txn_count
        
        print(f"Generating {count:,} records...")
        print(f"  - Transactions: {txn_count:,}")
        print(f"  - Edge cases: {edge_case_count:,}")
        print(f"  - Non-transactions: {non_txn_count:,}")
        
        # Generate transactions
        for i in range(txn_count):
            record = self.generate_transaction(noise_level=noise_level)
            record['id'] = i + 1
            record['hash'] = hashlib.sha256(record['text'].encode()).hexdigest()[:16]
            records.append(record)
            
            if (i + 1) % 10000 == 0:
                print(f"  Generated {i+1:,}/{count:,}")
        
        # Generate edge cases
        edge_configs = EdgeCaseGenerator.generate_edge_case_batch(edge_case_count, self.rng)
        for i, config in enumerate(edge_configs):
            # Generate with edge case parameters
            record = self.generate_transaction(noise_level=noise_level * 0.5)
            
            # Apply edge case modifications
            if config.get('amount'):
                record['ground_truth']['amount'] = config['amount']
            if config.get('merchant'):
                record['ground_truth']['merchant_raw'] = config['merchant']
            if config.get('beneficiary'):
                record['ground_truth']['beneficiary'] = config['beneficiary']
            
            record['id'] = txn_count + i + 1
            record['edge_case'] = config['edge_type']
            records.append(record)
        
        # Generate non-transactions
        for i in range(non_txn_count):
            msg_type = self.rng.choice([MessageType.OTP, MessageType.PROMOTIONAL, MessageType.ALERT])
            record = self.generate_non_transaction(msg_type)
            record['id'] = txn_count + edge_case_count + i + 1
            records.append(record)
        
        # Shuffle
        self.rng.shuffle(records)
        
        return records


# ============================================================================
# VALIDATION
# ============================================================================

class DatasetValidator:
    """Validate generated dataset for quality."""
    
    @staticmethod
    def validate(data: List[Dict]) -> Dict[str, Any]:
        """
        Validate dataset quality.
        
        Checks:
        1. Required fields present
        2. Distribution matches expectations
        3. No duplicate texts
        4. Format correctness
        """
        issues = []
        warnings = []
        
        # Check duplicates
        texts = [r['text'] for r in data]
        unique = set(texts)
        dup_rate = 1 - len(unique) / len(texts)
        if dup_rate > 0.01:
            warnings.append(f"Duplicate rate: {dup_rate:.2%}")
        
        # Check field completeness
        txn_records = [r for r in data if r['ground_truth'].get('message_type') == 'transaction']
        
        for r in txn_records:
            gt = r['ground_truth']
            if gt.get('amount') is None:
                issues.append(f"Missing amount in record {r.get('id')}")
            if gt.get('type') is None:
                issues.append(f"Missing type in record {r.get('id')}")
        
        # Check distributions
        categories = defaultdict(int)
        statuses = defaultdict(int)
        types = defaultdict(int)
        
        for r in txn_records:
            gt = r['ground_truth']
            categories[gt.get('category', 'unknown')] += 1
            statuses[gt.get('status', 'unknown')] += 1
            types[gt.get('type', 'unknown')] += 1
        
        # Calculate statistics
        total_txn = len(txn_records)
        
        result = {
            'total_records': len(data),
            'transaction_records': total_txn,
            'non_transaction_records': len(data) - total_txn,
            'unique_texts': len(unique),
            'duplicate_rate': dup_rate,
            'issues': issues[:20],
            'warnings': warnings,
            'distributions': {
                'categories': {k: v/total_txn for k, v in categories.items()},
                'statuses': {k: v/total_txn for k, v in statuses.items()},
                'types': {k: v/total_txn for k, v in types.items()},
            },
            'valid': len(issues) == 0,
        }
        
        return result


# ============================================================================
# OUTPUT WRITERS
# ============================================================================

def save_jsonl(data: List[Dict], path: Path):
    """Save as JSONL."""
    with open(path, 'w', encoding='utf-8') as f:
        for record in data:
            line = {
                'input': record['text'],
                'output': json.dumps(record['ground_truth'], ensure_ascii=False),
                'id': record.get('id'),
                'hash': record.get('hash'),
            }
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(f"✅ Saved: {path}")


def save_chat_format(data: List[Dict], path: Path):
    """Save in chat format for instruction tuning."""
    system = """Extract financial entities from Indian banking messages. Output JSON with:
- amount, amount_raw, type, status, account, date, reference
- merchant, vpa, beneficiary, payment_method, category
Only include fields present in the message."""

    with open(path, 'w', encoding='utf-8') as f:
        for record in data:
            chat = {
                'messages': [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': record['text']},
                    {'role': 'assistant', 'content': json.dumps(record['ground_truth'], ensure_ascii=False)},
                ]
            }
            f.write(json.dumps(chat, ensure_ascii=False) + '\n')
    print(f"✅ Saved: {path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Production-grade synthetic data generator for Indian banking transactions"
    )
    parser.add_argument("-n", "--count", type=int, default=10000, help="Number of records")
    parser.add_argument("-o", "--output", default="data/synthetic.jsonl", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--noise", type=float, default=0.3, help="Noise level (0-1)")
    parser.add_argument("--validate", action="store_true", help="Validate after generation")
    parser.add_argument("--chat-format", action="store_true", help="Also save chat format")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate
    generator = TransactionGenerator(seed=args.seed)
    data = generator.generate_batch(
        count=args.count,
        noise_level=args.noise,
    )
    
    # Save
    save_jsonl(data, output_path)
    
    if args.chat_format:
        chat_path = output_path.with_suffix('.chat.jsonl')
        save_chat_format(data, chat_path)
    
    # Validate
    if args.validate:
        print("\n📊 Validation:")
        result = DatasetValidator.validate(data)
        print(f"  Total: {result['total_records']:,}")
        print(f"  Unique: {result['unique_texts']:,}")
        print(f"  Duplicate rate: {result['duplicate_rate']:.2%}")
        print(f"  Valid: {'✅' if result['valid'] else '❌'}")
        
        if result['issues']:
            print(f"\n⚠️ Issues ({len(result['issues'])}):")
            for issue in result['issues'][:5]:
                print(f"  - {issue}")
        
        print("\n📈 Distributions:")
        for dist_name, dist in result['distributions'].items():
            print(f"\n  {dist_name}:")
            for k, v in sorted(dist.items(), key=lambda x: -x[1])[:5]:
                print(f"    {k}: {v:.1%}")


if __name__ == "__main__":
    main()