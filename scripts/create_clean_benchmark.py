"""
Create Multi-Bank Clean Benchmark.

Creates a high-quality benchmark from real transaction emails
across multiple banks: HDFC, ICICI, SBI, Axis, Kotak, PhonePe, GPay, Paytm.

Author: Ranjit Behera
"""

import json
import re
import random
from pathlib import Path
from collections import defaultdict

CORPUS_FILE = Path("data/corpus/emails/financial_emails.jsonl")
BENCHMARK_FILE = Path("data/benchmark/multi_bank_benchmark.json")


def detect_bank(body: str, sender: str = "") -> str:
    """Detect bank from email content and sender."""
    text = (body + " " + sender).lower()
    
    if 'hdfc' in text:
        return 'hdfc'
    elif 'icici' in text:
        return 'icici'
    elif 'sbi' in text or 'state bank' in text:
        return 'sbi'
    elif 'axis' in text:
        return 'axis'
    elif 'kotak' in text:
        return 'kotak'
    elif 'phonepe' in text:
        return 'phonepe'
    elif 'gpay' in text or 'google pay' in text:
        return 'gpay'
    elif 'paytm' in text:
        return 'paytm'
    
    return ''


def extract_entities(body: str, bank: str) -> dict:
    """Extract entities from email based on bank format."""
    entities = {
        'amount': '',
        'type': '',
        'date': '',
        'account': '',
        'reference': '',
        'merchant': '',
        'bank': bank
    }
    
    # Amount patterns (works across banks)
    amount_patterns = [
        r'Rs\.?\s*([\d,]+\.?\d*)',
        r'INR\s*([\d,]+\.?\d*)',
        r'₹\s*([\d,]+\.?\d*)',
    ]
    for pattern in amount_patterns:
        match = re.search(pattern, body, re.IGNORECASE)
        if match:
            entities['amount'] = match.group(1).replace(',', '')
            break
    
    # Type detection
    body_lower = body.lower()
    if any(x in body_lower for x in ['debited', 'sent', 'paid', 'payment of']):
        entities['type'] = 'debit'
    elif any(x in body_lower for x in ['credited', 'received', 'added']):
        entities['type'] = 'credit'
    
    # Bank-specific patterns
    if bank == 'hdfc':
        # HDFC: "from account 3545" or "A/c **3545"
        match = re.search(r'(?:account|A/c\s*\**)(\d{4})', body)
        if match:
            entities['account'] = match.group(1)
        
        # Date: "on 22-12-25"
        match = re.search(r'on\s*(\d{1,2}-\d{1,2}-\d{2,4})', body)
        if match:
            entities['date'] = match.group(1)
        
        # Reference: "reference number is 535680069988"
        match = re.search(r'reference number is\s*(\d{10,})', body)
        if match:
            entities['reference'] = match.group(1)
    
    elif bank == 'icici':
        # ICICI: "A/c XX5061" or "Acct XX4872"
        match = re.search(r'(?:A/c|Acct)\s*XX(\d{4})', body, re.IGNORECASE)
        if match:
            entities['account'] = match.group(1)
        
        # Date: "on 11112025" or "on 13 Nov 2025"
        match = re.search(r'on\s*(\d{8}|\d{1,2}\s+\w+\s+\d{4}|\d{1,2}-\d{1,2}-\d{2,4})', body)
        if match:
            entities['date'] = match.group(1)
        
        # Reference: "Ref:230788137103" or "IMPS Ref 928612436713"
        match = re.search(r'Ref[:\s]*(\d{10,})', body, re.IGNORECASE)
        if match:
            entities['reference'] = match.group(1)
    
    elif bank == 'sbi':
        # SBI: "a/c XX9666" or "A/c X2771"
        match = re.search(r'[Aa]/c\s*X+(\d{4})', body)
        if match:
            entities['account'] = match.group(1)
        
        # Date
        match = re.search(r'on\s*(\d{1,2}-\d{1,2}-\d{2,4}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}\s+\w+\s+\d{4})', body)
        if match:
            entities['date'] = match.group(1)
        
        # Reference
        match = re.search(r'Ref\s*(\d{10,})', body, re.IGNORECASE)
        if match:
            entities['reference'] = match.group(1)
    
    elif bank == 'axis':
        # Axis: "Acct XX4185" or "A/c XX3041"
        match = re.search(r'(?:Acct|A/c)\s*XX(\d{4})', body, re.IGNORECASE)
        if match:
            entities['account'] = match.group(1)
        
        # Date
        match = re.search(r'on\s*(\d{8}|\d{1,2}-\d{1,2}-\d{4}|\d{1,2}/\d{1,2}/\d{4})', body)
        if match:
            entities['date'] = match.group(1)
        
        # Reference
        match = re.search(r'Ref\s*(\d{10,})', body, re.IGNORECASE)
        if match:
            entities['reference'] = match.group(1)
    
    elif bank == 'kotak':
        # Kotak: "A/c XX6934" or "A/c 9817"
        match = re.search(r'A/c\s*(?:XX)?(\d{4})', body, re.IGNORECASE)
        if match:
            entities['account'] = match.group(1)
        
        # Date
        match = re.search(r'on\s*(\d{8}|\d{1,2}-\d{1,2}-\d{2,4}|\d{1,2}\s+\w+\s+\d{4})', body)
        if match:
            entities['date'] = match.group(1)
        
        # Reference
        match = re.search(r'Ref[:\s.]*(\d{10,})', body, re.IGNORECASE)
        if match:
            entities['reference'] = match.group(1)
    
    elif bank in ['phonepe', 'gpay', 'paytm']:
        # Payment apps: various patterns
        match = re.search(r'(?:a/c|account)\s*(?:XX)?(\d{4})', body, re.IGNORECASE)
        if match:
            entities['account'] = match.group(1)
        
        # Date
        match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+\w+\s+\d{4})', body)
        if match:
            entities['date'] = match.group(1)
        
        # Reference/Txn ID
        match = re.search(r'(?:Ref|Txn\s*ID)[:\s]*(\d{10,})', body, re.IGNORECASE)
        if match:
            entities['reference'] = match.group(1)
    
    # Merchant from VPA (works across banks)
    vpa_match = re.search(r'VPA[:\s]+\S+\s+([A-Z][A-Za-z\s]+?)(?:\s+on|\s+\d|$)', body)
    if vpa_match:
        merchant = vpa_match.group(1).strip().lower()
        if len(merchant) > 2 and len(merchant) < 50:
            entities['merchant'] = merchant
    
    # Also try UPI: pattern
    if not entities['merchant']:
        upi_match = re.search(r'UPI[:\s-]+([A-Za-z]+)', body)
        if upi_match:
            entities['merchant'] = upi_match.group(1).lower()
    
    return entities


def create_multi_bank_benchmark():
    """Create benchmark with multiple banks."""
    print("=" * 60)
    print("📊 CREATING MULTI-BANK BENCHMARK")
    print("=" * 60)
    
    # Collect transactions by bank
    bank_transactions = defaultdict(list)
    
    with open(CORPUS_FILE, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                body = data.get('body', '')
                sender = data.get('sender', '')
                
                # Must have transaction keywords
                body_lower = body.lower()
                has_transaction = any(x in body_lower for x in 
                    ['debited', 'credited', 'received', 'sent', 'paid', 'payment'])
                has_amount = any(x in body_lower for x in ['rs.', 'rs ', 'inr', '₹'])
                
                if not (has_transaction and has_amount and len(body) > 50):
                    continue
                
                # Detect bank
                bank = detect_bank(body, sender)
                if not bank:
                    continue
                
                # Extract entities
                entities = extract_entities(body, bank)
                
                # Only include if we have good extraction
                if entities['amount'] and entities['type']:
                    bank_transactions[bank].append({
                        'text': body,
                        'expected_entities': entities,
                        'subject': data.get('subject', ''),
                        'verified': True
                    })
            except:
                continue
    
    print("\n📊 Transactions found by bank:")
    for bank, txns in sorted(bank_transactions.items()):
        print(f"   {bank.upper():10} {len(txns):4} transactions")
    
    # Sample from each bank (max 20 per bank)
    random.seed(42)
    benchmark = []
    
    for bank, txns in bank_transactions.items():
        # Deduplicate by reference if available
        seen_refs = set()
        unique_txns = []
        for t in txns:
            ref = t['expected_entities'].get('reference', '')
            if ref:
                if ref not in seen_refs:
                    seen_refs.add(ref)
                    unique_txns.append(t)
            else:
                unique_txns.append(t)
        
        # Sample
        sampled = random.sample(unique_txns, min(20, len(unique_txns)))
        benchmark.extend(sampled)
    
    # Add IDs
    for i, sample in enumerate(benchmark):
        sample['id'] = i + 1
    
    # Shuffle
    random.shuffle(benchmark)
    
    # Save
    BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_FILE, 'w') as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(benchmark)} samples to {BENCHMARK_FILE}")
    
    # Stats by bank
    bank_counts = defaultdict(int)
    for s in benchmark:
        bank_counts[s['expected_entities']['bank']] += 1
    
    print("\n📊 Benchmark composition:")
    for bank, count in sorted(bank_counts.items()):
        print(f"   {bank.upper():10} {count:3} samples")
    
    # Show sample from each bank
    print("\n📧 Sample from each bank:")
    shown_banks = set()
    for s in benchmark:
        bank = s['expected_entities']['bank']
        if bank not in shown_banks:
            shown_banks.add(bank)
            print(f"\n   [{bank.upper()}]")
            print(f"   Amount: {s['expected_entities']['amount']}")
            print(f"   Type: {s['expected_entities']['type']}")
            print(f"   Text: {s['text'][:100]}...")
            if len(shown_banks) >= 4:
                break
    
    return benchmark


if __name__ == "__main__":
    create_multi_bank_benchmark()
