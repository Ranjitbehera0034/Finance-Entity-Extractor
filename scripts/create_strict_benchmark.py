"""
Create Strict Transaction Benchmark.

Only includes real transaction alerts with clear patterns.
Excludes marketing, bill notifications, and investment updates.

Author: Ranjit Behera
"""

import json
import re
import random
from pathlib import Path
from collections import defaultdict

CORPUS_FILE = Path("data/corpus/emails/financial_emails.jsonl")
BENCHMARK_FILE = Path("data/benchmark/strict_benchmark.json")

# Transaction patterns that indicate real transactions
TRANSACTION_PATTERNS = [
    r'has been debited',
    r'has been credited', 
    r'is debited from',
    r'is credited to',
    r'Rs\.\s*[\d,]+.*debited',
    r'Rs\.\s*[\d,]+.*credited',
    r'INR\s*[\d,]+.*debited',
    r'INR\s*[\d,]+.*credited',
    r'UPI transaction reference',
    r'UPI Ref',
    r'IMPS Ref',
    r'NEFT Ref',
]

# Exclude patterns (marketing, bills, investments)
EXCLUDE_PATTERNS = [
    r'welcome to your',
    r'greetings of the day',
    r'unsubscribe from',
    r'skills that will get you',
    r'daily digest',
    r'top picks',
    r'mutual fund nav',
    r'market update',
    r'job opportunity',
    r'margin statement',
    r'password reset',
]


def is_transaction_email(body: str) -> bool:
    """Check if email is a real transaction alert."""
    body_lower = body.lower()
    
    # Must match at least one transaction pattern
    has_transaction = any(re.search(p, body, re.IGNORECASE) for p in TRANSACTION_PATTERNS)
    
    # Must not match exclude patterns
    has_exclude = any(re.search(p, body_lower) for p in EXCLUDE_PATTERNS)
    
    return has_transaction and not has_exclude


def detect_bank(body: str, sender: str = "") -> str:
    """Detect bank from email."""
    text = (body + " " + sender).lower()
    
    # Priority order (more specific first)
    if 'hdfc bank' in text or 'hdfcbank' in text:
        return 'hdfc'
    elif 'icici bank' in text:
        return 'icici'
    elif 'state bank' in text or 'sbi:' in text:
        return 'sbi'
    elif 'axis bank' in text:
        return 'axis'
    elif 'kotak' in text:
        return 'kotak'
    
    return ''


def extract_entities(body: str, bank: str) -> dict:
    """Extract entities from transaction email."""
    entities = {
        'amount': '',
        'type': '',
        'date': '',
        'account': '',
        'reference': '',
        'merchant': '',
        'bank': bank
    }
    
    # Amount
    match = re.search(r'Rs\.?\s*([\d,]+\.?\d*)', body, re.IGNORECASE)
    if match:
        entities['amount'] = match.group(1).replace(',', '')
    else:
        match = re.search(r'INR\s*([\d,]+\.?\d*)', body, re.IGNORECASE)
        if match:
            entities['amount'] = match.group(1).replace(',', '')
    
    # Type
    body_lower = body.lower()
    if 'debited' in body_lower:
        entities['type'] = 'debit'
    elif 'credited' in body_lower:
        entities['type'] = 'credit'
    
    # Account (4 digits after XX or **)
    match = re.search(r'(?:XX|X|\*\*|account\s*)(\d{4})', body, re.IGNORECASE)
    if match:
        entities['account'] = match.group(1)
    
    # Date
    match = re.search(r'on\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', body)
    if match:
        entities['date'] = match.group(1)
    
    # Reference (12+ digit number)
    ref_patterns = [
        r'reference number is\s*(\d{10,})',
        r'(?:Ref(?:erence)?[:\s.]*|UPI\s*Ref[:\s]*|IMPS\s*Ref[:\s]*)(\d{10,})',
    ]
    for pattern in ref_patterns:
        match = re.search(pattern, body, re.IGNORECASE)
        if match:
            entities['reference'] = match.group(1)
            break
    
    # Merchant from VPA
    match = re.search(r'VPA[:\s]+\S+\s+([A-Z][A-Za-z\s]+?)(?:\s+on|\s+\d)', body)
    if match:
        entities['merchant'] = match.group(1).strip().lower()
    
    return entities


def create_strict_benchmark():
    """Create strictly filtered benchmark."""
    print("=" * 60)
    print("📊 CREATING STRICT TRANSACTION BENCHMARK")
    print("=" * 60)
    
    bank_transactions = defaultdict(list)
    
    with open(CORPUS_FILE, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                body = data.get('body', '')
                sender = data.get('sender', '')
                
                # Strict filtering
                if not is_transaction_email(body):
                    continue
                
                if len(body) < 50:
                    continue
                
                # Detect bank
                bank = detect_bank(body, sender)
                if not bank:
                    continue
                
                # Extract entities
                entities = extract_entities(body, bank)
                
                # Must have amount, type, and reference
                if entities['amount'] and entities['type'] and entities['reference']:
                    bank_transactions[bank].append({
                        'text': body,
                        'expected_entities': entities,
                        'subject': data.get('subject', ''),
                        'verified': True
                    })
            except:
                continue
    
    print("\n📊 Strict transactions per bank:")
    for bank, txns in sorted(bank_transactions.items()):
        print(f"   {bank.upper():10} {len(txns):4} transactions")
    
    # Sample and deduplicate
    random.seed(42)
    benchmark = []
    
    for bank, txns in bank_transactions.items():
        # Deduplicate by reference
        seen_refs = set()
        unique = []
        for t in txns:
            ref = t['expected_entities']['reference']
            if ref not in seen_refs:
                seen_refs.add(ref)
                unique.append(t)
        
        sampled = random.sample(unique, min(15, len(unique)))
        benchmark.extend(sampled)
    
    for i, s in enumerate(benchmark):
        s['id'] = i + 1
    
    random.shuffle(benchmark)
    
    # Save
    BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_FILE, 'w') as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(benchmark)} samples to {BENCHMARK_FILE}")
    
    # Stats
    bank_counts = defaultdict(int)
    for s in benchmark:
        bank_counts[s['expected_entities']['bank']] += 1
    
    print("\n📊 Benchmark composition:")
    for bank, count in sorted(bank_counts.items()):
        print(f"   {bank.upper():10} {count:3} samples")
    
    # Show samples
    print("\n📧 Sample transaction:")
    if benchmark:
        s = benchmark[0]
        print(f"   Bank: {s['expected_entities']['bank'].upper()}")
        print(f"   Amount: {s['expected_entities']['amount']}")
        print(f"   Type: {s['expected_entities']['type']}")
        print(f"   Reference: {s['expected_entities']['reference']}")
        print(f"   Text: {s['text'][:150]}...")
    
    return benchmark


if __name__ == "__main__":
    create_strict_benchmark()
