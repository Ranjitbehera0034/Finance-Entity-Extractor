"""
Generate ICICI-Specific Training Data.

Creates 100 ICICI Bank email samples to improve model accuracy.

Author: Ranjit Behera
"""

import json
import random
from pathlib import Path
from datetime import datetime, timedelta

OUTPUT_FILE = Path("data/synthetic/icici_samples.jsonl")

# ICICI-specific email templates (based on real formats)
ICICI_TEMPLATES = [
    # Template 1: Standard debit
    "Dear Customer, Rs.{amount} is debited from A/c XX{account} for UPI txn to VPA-{merchant}@{suffix} on {date}. Ref:{reference}",
    
    # Template 2: Credit
    "INR {amount} credited to ICICI Bank A/c {account} on {date} from {sender}. UPI Ref: {reference}",
    
    # Template 3: Credit with IMPS
    "ICICI Bank Acct XX{account} credited with INR {amount} on {date}. IMPS Ref {reference}.",
    
    # Template 4: Debit with merchant
    "INR {amount} debited from ICICI Bank A/c {account} on {date} to {merchant} via UPI. Ref: {reference}",
    
    # Template 5: Alert format
    "Alert: Rs {amount} debited from your ICICI Bank account ending {account} on {date}. UPI Ref No. {reference}. Not you? Call 1800-200-3344",
    
    # Template 6: Credit received
    "Dear Customer, your ICICI Bank A/c XX{account} has been credited with Rs.{amount} on {date}. Ref: {reference}",
    
    # Template 7: Transaction alert
    "ICICI Bank: Rs.{amount} has been debited from your account XX{account} on {date} for UPI payment to {merchant}. Ref:{reference}",
    
    # Template 8: Balance info
    "Dear Customer, INR {amount} debited from ICICI A/c XX{account} on {date}. Info: UPI-{merchant}. Avl Bal: Rs.{balance}",
    
    # Template 9: Statement format
    "Rs {amount} has been credited to your ICICI Bank account {account} on {date} via NEFT. Reference: {reference}",
    
    # Template 10: Fund transfer
    "ICICI Bank: Fund Transfer of Rs.{amount} to {merchant} successful. A/c: XX{account}. Date: {date}. Ref: {reference}"
]

VPA_SUFFIXES = ['ybl', 'paytm', 'okicici', 'okhdfcbank', 'axl', 'sbi', 'icici']

MERCHANTS = [
    ('swiggy', 'food'), ('zomato', 'food'), ('amazon', 'shopping'),
    ('flipkart', 'shopping'), ('uber', 'transport'), ('ola', 'transport'),
    ('rapido', 'transport'), ('bigbasket', 'grocery'), ('blinkit', 'grocery'),
    ('zepto', 'grocery'), ('dmart', 'grocery'), ('jio', 'bills'),
    ('airtel', 'bills'), ('electricity', 'bills'), ('water', 'bills'),
    ('myntra', 'shopping'), ('ajio', 'shopping'), ('bookmyshow', 'entertainment'),
    ('netflix', 'entertainment'), ('hotstar', 'entertainment')
]

SENDERS = [
    'Amit Kumar', 'Priya Singh', 'Rahul Sharma', 'Neha Gupta',
    'Suresh Patel', 'Anita Verma', 'Cashback', 'Refund - Amazon',
    'Interest Credit', 'Dividend - Mutual Fund', 'Salary'
]

DATE_FORMATS = [
    '%d-%m-%Y', '%d/%m/%Y', '%d-%m-%y', '%d %b %Y', 
    '%d%m%Y', '%d %B %Y'
]

def generate_random_date():
    """Generate random date in past 3 months."""
    days_ago = random.randint(1, 90)
    d = datetime.now() - timedelta(days=days_ago)
    fmt = random.choice(DATE_FORMATS)
    return d.strftime(fmt)

def generate_reference():
    """Generate 12-digit reference number."""
    return ''.join([str(random.randint(0, 9)) for _ in range(12)])

def generate_account():
    """Generate 4-digit account number."""
    return str(random.randint(1000, 9999))

def generate_amount(is_high=False):
    """Generate realistic amount."""
    if is_high:
        return str(random.randint(5000, 50000))
    else:
        amounts = [
            round(random.uniform(50, 500), 2),
            round(random.uniform(100, 2000), 2),
            random.randint(100, 5000),
            round(random.uniform(500, 3000), 2)
        ]
        return str(random.choice(amounts))

def generate_icici_samples(n_samples=100):
    """Generate ICICI-specific training samples."""
    samples = []
    
    for i in range(n_samples):
        template = random.choice(ICICI_TEMPLATES)
        is_credit = 'credited' in template.lower() or 'credit' in template.lower()
        
        merchant_info = random.choice(MERCHANTS)
        merchant = merchant_info[0]
        category = merchant_info[1]
        
        account = generate_account()
        reference = generate_reference()
        amount = generate_amount(is_high=random.random() < 0.2)
        date = generate_random_date()
        balance = str(random.randint(10000, 500000))
        sender = random.choice(SENDERS)
        suffix = random.choice(VPA_SUFFIXES)
        
        # Fill template
        text = template.format(
            amount=amount,
            account=account,
            reference=reference,
            date=date,
            merchant=merchant,
            category=category,
            balance=balance,
            sender=sender,
            suffix=suffix
        )
        
        # Create entities
        entities = {
            'amount': amount.replace(',', ''),
            'type': 'credit' if is_credit else 'debit',
            'date': date,
            'account': account,
            'reference': reference
        }
        
        if not is_credit:
            entities['merchant'] = merchant
            entities['category'] = category
        
        # Create training format (prompt/completion)
        prompt = f"""Extract financial entities from this ICICI Bank email:

{text}

Extract: amount, type, date, account, reference{', merchant, category' if not is_credit else ''}
Output JSON:"""

        completion = json.dumps(entities, indent=2)
        
        sample = {
            'prompt': prompt,
            'completion': completion,
            'bank': 'icici',
            'entities': entities
        }
        samples.append(sample)
    
    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"✅ Generated {len(samples)} ICICI samples")
    print(f"   Saved to {OUTPUT_FILE}")
    
    # Show sample
    print("\n📧 Sample:")
    print(samples[0]['prompt'][:300])
    
    return samples


if __name__ == "__main__":
    generate_icici_samples(100)
