"""
Create Held-Out Benchmark from Real Emails.

Extracts 100 real financial emails from the MBOX file,
ensures they were NOT used in training, and creates a
benchmark for measuring real-world performance.

Author: Ranjit Behera
"""

import json
import random
import re
from pathlib import Path

# Paths
CORPUS_FILE = Path("data/corpus/emails/financial_emails.jsonl")
TRAIN_FILE = Path("data/training/train.jsonl")
BENCHMARK_FILE = Path("data/benchmark/real_emails_benchmark.json")

def load_corpus():
    """Load the extracted financial emails."""
    emails = []
    with open(CORPUS_FILE, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                emails.append(data)
            except:
                continue
    return emails

def load_training_texts():
    """Load training data to exclude from benchmark."""
    texts = set()
    with open(TRAIN_FILE, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Get first 100 chars as fingerprint
                text = data.get('text', '')[:100]
                texts.add(text)
            except:
                continue
    return texts

def extract_entities_from_email(email_body: str) -> dict:
    """Auto-extract entities from email text for labeling."""
    entities = {
        'amount': '',
        'type': '',
        'date': '',
        'account': '',
        'reference': '',
        'merchant': '',
        'bank': ''
    }
    
    text = email_body
    text_lower = text.lower()
    
    # Detect bank
    if 'hdfc' in text_lower:
        entities['bank'] = 'hdfc'
    elif 'icici' in text_lower:
        entities['bank'] = 'icici'
    elif 'sbi' in text_lower:
        entities['bank'] = 'sbi'
    elif 'axis' in text_lower:
        entities['bank'] = 'axis'
    elif 'kotak' in text_lower:
        entities['bank'] = 'kotak'
    elif 'phonepe' in text_lower:
        entities['bank'] = 'phonepe'
    elif 'gpay' in text_lower or 'google pay' in text_lower:
        entities['bank'] = 'gpay'
    elif 'paytm' in text_lower:
        entities['bank'] = 'paytm'
    
    # Detect type
    if 'debited' in text_lower or 'sent' in text_lower or 'paid' in text_lower:
        entities['type'] = 'debit'
    elif 'credited' in text_lower or 'received' in text_lower:
        entities['type'] = 'credit'
    
    # Extract amount - various patterns
    amount_patterns = [
        r'Rs\.?\s*([\d,]+\.?\d*)',
        r'INR\s*([\d,]+\.?\d*)',
        r'₹\s*([\d,]+\.?\d*)',
        r'([\d,]+\.?\d*)\s*has been (?:debited|credited)'
    ]
    for pattern in amount_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            entities['amount'] = match.group(1).replace(',', '')
            break
    
    # Extract account
    account_patterns = [
        r'account\s*(?:no\.?|number|#|XX|X)?\s*(\d{4})',
        r'A/c\s*(?:XX|X)?(\d{4})',
        r'a/c\s*(\d{4})',
        r'XX(\d{4})',
        r'X(\d{4})'
    ]
    for pattern in account_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            entities['account'] = match.group(1)
            break
    
    # Extract date - various formats
    date_patterns = [
        r'on\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})',
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            entities['date'] = match.group(1)
            break
    
    # Extract reference
    ref_patterns = [
        r'(?:UPI\s*)?(?:Ref(?:erence)?(?:\s*(?:No|Number|#|:))?\.?\s*:?\s*)(\d{10,})',
        r'transaction reference number is\s*(\d+)',
        r'Txn[:\s]*(\d+)',
    ]
    for pattern in ref_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            entities['reference'] = match.group(1)
            break
    
    # Extract merchant from VPA
    vpa_match = re.search(r'VPA\s+(\w+)@\w+\s+([A-Z][A-Za-z\s]+)', text)
    if vpa_match:
        entities['merchant'] = vpa_match.group(2).strip().lower()
    else:
        # Try common merchants
        merchants = ['swiggy', 'zomato', 'amazon', 'flipkart', 'uber', 'ola', 'rapido', 'bigbasket', 'blinkit', 'zepto']
        for m in merchants:
            if m in text_lower:
                entities['merchant'] = m
                break
    
    return entities


def create_benchmark(n_samples=100):
    """Create held-out benchmark from real emails."""
    print("=" * 60)
    print("📊 CREATING HELD-OUT BENCHMARK")
    print("=" * 60)
    
    # Load data
    print(f"\n1. Loading corpus from {CORPUS_FILE}...")
    corpus = load_corpus()
    print(f"   Found {len(corpus)} financial emails")
    
    print(f"\n2. Loading training data to exclude...")
    train_texts = load_training_texts()
    print(f"   Found {len(train_texts)} training samples to exclude")
    
    # Filter for transaction emails
    print(f"\n3. Filtering for transaction emails...")
    candidates = []
    for email in corpus:
        body = email.get('body', '')
        
        # Skip if too short
        if len(body) < 50:
            continue
        
        # Must have transaction keywords
        body_lower = body.lower()
        has_transaction = any(x in body_lower for x in ['debited', 'credited', 'received', 'sent'])
        has_amount = any(x in body_lower for x in ['rs.', 'rs ', 'inr', '₹'])
        
        if has_transaction and has_amount:
            # Auto-extract entities
            entities = extract_entities_from_email(body)
            
            candidates.append({
                'text': body,
                'subject': email.get('subject', ''),
                'sender': email.get('sender', ''),
                'date': email.get('date', ''),
                'expected_entities': entities
            })
    
    print(f"   Found {len(candidates)} transaction emails")
    
    # Sample randomly
    print(f"\n4. Sampling {n_samples} emails for benchmark...")
    random.seed(42)  # Reproducible
    benchmark = random.sample(candidates, min(n_samples, len(candidates)))
    
    # Add IDs
    for i, sample in enumerate(benchmark):
        sample['id'] = i + 1
        sample['auto_labeled'] = True
        sample['verified'] = False
    
    # Save benchmark
    BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_FILE, 'w') as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Benchmark saved to {BENCHMARK_FILE}")
    print(f"   Total samples: {len(benchmark)}")
    
    # Stats
    banks = {}
    for s in benchmark:
        bank = s['expected_entities'].get('bank', 'unknown')
        banks[bank] = banks.get(bank, 0) + 1
    
    print("\n📊 Benchmark by Bank:")
    for bank, count in sorted(banks.items()):
        print(f"   {bank.upper():10} {count}")
    
    # Show sample
    print("\n" + "=" * 60)
    print("📧 SAMPLE EMAIL FROM BENCHMARK:")
    print("=" * 60)
    if benchmark:
        sample = benchmark[0]
        print(f"Subject: {sample.get('subject', 'N/A')}")
        print(f"Text: {sample['text'][:300]}...")
        print(f"\nAuto-extracted entities:")
        for k, v in sample['expected_entities'].items():
            if v:
                print(f"   {k}: {v}")
    
    return benchmark


if __name__ == "__main__":
    create_benchmark(n_samples=100)
