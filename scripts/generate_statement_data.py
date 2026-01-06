"""
Generate synthetic bank statement training data for Phase 2.

Creates realistic bank statement row samples for all supported banks
with proper labeling, ready for training.

Author: Ranjit Behera
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Seed for reproducibility
random.seed(42)

# Bank-specific row formats
BANK_FORMATS = {
    "hdfc": [
        "{date} | {desc} | {debit} | {credit} | {balance}",
        "{date}  {desc}  DR:{debit}  BAL:{balance}",
        "{date} {desc} Withdrawal: Rs.{debit} Balance: Rs.{balance}",
    ],
    "icici": [
        "{date} | {desc} | {debit} | {credit} | {balance}",
        "{date}//ICICI//{desc}//AMT:{amount}//BAL:{balance}",
        "{date} - {desc} - Rs.{amount} - Bal: Rs.{balance}",
    ],
    "sbi": [
        "{date} | {desc} | {debit} | {credit} | {balance}",
        "SBI/{date}/{desc}/Rs.{amount}/Bal.{balance}",
        "{date} {desc} Amount: {amount} Available: {balance}",
    ],
    "axis": [
        "{date} | {desc} | {debit} | {credit} | {balance}",
        "AXIS-{date}-{desc}-{amount}-{balance}",
        "{date} >> {desc} >> Rs {amount} >> Bal Rs {balance}",
    ],
    "kotak": [
        "{date} | {desc} | {debit} | {credit} | {balance}",
        "{date} KOTAK {desc} {amount} BAL:{balance}",
        "{date}: {desc} | Amount: Rs.{amount} | Balance: Rs.{balance}",
    ],
}

# Transaction descriptions by category
DESCRIPTIONS = {
    "upi": [
        "UPI-{merchant}@ybl-PAYMENT",
        "UPI/{merchant}/{ref}",
        "UPI-TRANSFER-{merchant}",
        "IMPS/P2P/{name}",
        "UPI-{phone}-{merchant}",
        "{merchant}@paytm-UPI",
        "UPI-{name}-{ref}",
    ],
    "neft": [
        "NEFT-{name}-{ref}",
        "NEFT CR-{bank}-{name}",
        "NEFT/TRANSFER/{account}/{name}",
        "NEFT-INW-{ref}-{name}",
        "RTGS-{bank}-{name}-{ref}",
    ],
    "card": [
        "POS {merchant} {city}",
        "ATM WDL {city} {ref}",
        "CARD TXN-{merchant}",
        "ECOM/{merchant}/ONLINE",
        "CC PAYMENT-{card}",
        "DEBIT CARD {merchant} {city}",
    ],
    "emi": [
        "EMI-{merchant}-{ref}",
        "AUTO-DEBIT-{merchant}",
        "SB/AUTODR/{merchant}",
        "LOAN EMI-{ref}",
        "ECS/{merchant}/PAYMENT",
    ],
    "bill": [
        "BILL PAY-{merchant}",
        "AUTOPAY-{merchant}-{ref}",
        "{merchant} BILL PAYMENT",
        "BBPS/{merchant}/{ref}",
        "ELECTRICITY-{merchant}",
        "MOBILE RECHARGE-{phone}",
    ],
    "transfer": [
        "FT-{name}-{account}",
        "SELF TRF-{account}",
        "FUND TRANSFER-{name}",
        "A/C TRANSFER-{ref}",
        "INT TRF-{name}",
    ],
    "salary": [
        "SALARY-{company}",
        "SAL-{month}-{company}",
        "NEFT-SALARY-{company}",
        "PAYROLL-{company}-{ref}",
    ],
    "cash": [
        "CASH DEP-{city}",
        "ATM DEP-{ref}",
        "CASH DEPOSIT-BR:{branch}",
        "CDM DEPOSIT-{city}",
    ],
}

MERCHANTS = [
    "Amazon", "Flipkart", "Swiggy", "Zomato", "Uber", "Ola",
    "BigBasket", "Zepto", "Blinkit", "Dunzo", "Netflix", "Spotify",
    "JioMart", "Myntra", "Nykaa", "BookMyShow", "MakeMyTrip",
    "PhonePe", "Paytm", "HDFC Life", "ICICI Pru", "Airtel", "Jio",
    "Vodafone", "BSNL", "TATA Play", "Hotstar", "Prime Video",
    "GooglePlay", "AppStore", "LinkedIn", "Udemy", "Coursera",
    "LIC", "SBI Life", "Bajaj Finserv", "Tata Capital"
]

NAMES = [
    "Rahul Kumar", "Priya Sharma", "Amit Singh", "Neha Patel",
    "Vikram Reddy", "Anjali Gupta", "Ravi Verma", "Pooja Joshi",
    "Suresh Menon", "Kavita Nair", "Arun Krishnan", "Deepa Iyer",
    "Sanjay Kapoor", "Meera Rao", "Kiran Hegde", "Anita Desai"
]

COMPANIES = [
    "TCS", "Infosys", "Wipro", "HCL Tech", "Tech Mahindra",
    "Accenture", "IBM India", "Microsoft", "Google", "Amazon",
    "Flipkart", "Paytm", "Reliance", "HDFC Bank", "ICICI Bank"
]

BANKS = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "YES", "IDFC", "RBL"]
CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Pune", "Hyderabad", "Kolkata", "Ahmedabad"]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Category mapping
CATEGORY_MAP = {
    "upi": ["food", "shopping", "transport", "bills", "transfer"],
    "neft": ["transfer", "salary", "investment"],
    "card": ["shopping", "food", "travel", "entertainment"],
    "emi": ["loan", "shopping"],
    "bill": ["bills", "utilities", "telecom"],
    "transfer": ["transfer"],
    "salary": ["salary"],
    "cash": ["cash"],
}


def random_date(days_back: int = 180) -> str:
    """Generate random date."""
    days_ago = random.randint(0, days_back)
    date = datetime.now() - timedelta(days=days_ago)
    
    formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", 
        "%d %b %Y", "%d %b %y", "%Y-%m-%d"
    ]
    return date.strftime(random.choice(formats))


def random_amount(min_val: int = 50, max_val: int = 50000) -> str:
    """Generate random amount."""
    amount = random.uniform(min_val, max_val)
    
    if random.random() < 0.4:
        return f"{amount:,.2f}"
    else:
        return f"{int(amount):,}"


def random_balance() -> str:
    """Generate random balance."""
    balance = random.uniform(10000, 500000)
    return f"{balance:,.2f}"


def random_ref() -> str:
    """Generate random reference number."""
    length = random.choice([8, 10, 12])
    return ''.join(str(random.randint(0, 9)) for _ in range(length))


def random_account() -> str:
    """Generate random account suffix."""
    return ''.join(str(random.randint(0, 9)) for _ in range(4))


def random_phone() -> str:
    """Generate random phone number."""
    return ''.join(str(random.randint(0, 9)) for _ in range(10))


def random_card() -> str:
    """Generate random card suffix."""
    return "XX" + ''.join(str(random.randint(0, 9)) for _ in range(4))


def generate_description(txn_type: str) -> tuple:
    """Generate description and metadata."""
    templates = DESCRIPTIONS[txn_type]
    template = random.choice(templates)
    
    merchant = random.choice(MERCHANTS)
    name = random.choice(NAMES)
    
    desc = template.format(
        merchant=merchant,
        name=name,
        ref=random_ref(),
        bank=random.choice(BANKS),
        account=random_account(),
        city=random.choice(CITIES),
        phone=random_phone(),
        card=random_card(),
        company=random.choice(COMPANIES),
        month=random.choice(MONTHS),
        branch=f"{random.randint(1, 99):02d}"
    )
    
    # Determine merchant for category
    if txn_type in ["upi", "card", "emi", "bill"]:
        detected_merchant = merchant.lower()
    else:
        detected_merchant = None
    
    # Determine category
    categories = CATEGORY_MAP.get(txn_type, ["other"])
    category = random.choice(categories)
    
    return desc, detected_merchant, category


def generate_row(bank: str) -> Dict[str, Any]:
    """Generate a single bank statement row."""
    # Transaction type weights
    txn_types = ["upi"] * 40 + ["card"] * 20 + ["neft"] * 15 + ["bill"] * 10 + \
                ["emi"] * 5 + ["transfer"] * 5 + ["salary"] * 3 + ["cash"] * 2
    txn_type = random.choice(txn_types)
    
    # Generate data
    date = random_date()
    desc, merchant, category = generate_description(txn_type)
    
    # Debit or credit
    is_credit = txn_type in ["salary", "cash", "neft"] and random.random() < 0.7
    
    if txn_type == "salary":
        amount = random_amount(30000, 150000)
    elif txn_type == "emi":
        amount = random_amount(5000, 30000)
    elif txn_type == "cash" and is_credit:
        amount = random_amount(5000, 50000)
    else:
        amount = random_amount(50, 25000)
    
    balance = random_balance()
    
    # Format based on bank
    formats = BANK_FORMATS.get(bank, BANK_FORMATS["hdfc"])
    template = random.choice(formats)
    
    if is_credit:
        raw_text = template.format(
            date=date, desc=desc, 
            debit="", credit=amount, 
            balance=balance, amount=amount
        )
    else:
        raw_text = template.format(
            date=date, desc=desc,
            debit=amount, credit="",
            balance=balance, amount=amount
        )
    
    # Build entities
    entities = {
        "date": date,
        "description": desc,
        "amount": amount.replace(",", ""),
        "type": "credit" if is_credit else "debit",
        "balance": balance.replace(",", ""),
    }
    
    if merchant:
        entities["merchant"] = merchant
    entities["category"] = category
    
    return {
        "raw_text": raw_text,
        "date": date,
        "description": desc,
        "debit": None if is_credit else amount,
        "credit": amount if is_credit else None,
        "balance": balance,
        "bank": bank,
        "labeled": True,
        "entities": entities
    }


def generate_training_data(
    samples_per_bank: int = 100,
    output_dir: str = "data/training"
) -> Dict[str, Any]:
    """Generate complete training dataset."""
    
    banks = ["hdfc", "icici", "sbi", "axis", "kotak"]
    all_samples = []
    
    for bank in banks:
        for _ in range(samples_per_bank):
            row = generate_row(bank)
            all_samples.append(row)
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Convert to training format with [BANK_STATEMENT] prefix
    training_data = []
    for sample in all_samples:
        prompt = f"[BANK_STATEMENT] Extract financial entities from this bank statement row:\n\n{sample['raw_text']}"
        completion = json.dumps(sample['entities'], indent=2)
        training_data.append({
            "prompt": prompt,
            "completion": completion
        })
    
    # Split train/valid
    split_idx = int(len(training_data) * 0.9)
    train_data = training_data[:split_idx]
    valid_data = training_data[split_idx:]
    
    # Save files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_file = output_path / "statement_train.jsonl"
    valid_file = output_path / "statement_valid.jsonl"
    
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(valid_file, 'w') as f:
        for item in valid_data:
            f.write(json.dumps(item) + '\n')
    
    # Also save raw samples for reference
    samples_file = output_path / "statement_samples.json"
    with open(samples_file, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    return {
        "total_samples": len(all_samples),
        "train_samples": len(train_data),
        "valid_samples": len(valid_data),
        "banks": banks,
        "train_file": str(train_file),
        "valid_file": str(valid_file),
        "samples_file": str(samples_file)
    }


def main():
    """Generate Phase 2 training data."""
    print("📄 Generating Phase 2: Bank Statement Training Data")
    print("=" * 60)
    
    result = generate_training_data(samples_per_bank=100)
    
    print(f"\n✅ Generated {result['total_samples']} samples")
    print(f"   Banks: {', '.join(b.upper() for b in result['banks'])}")
    print(f"   Train: {result['train_samples']} samples")
    print(f"   Valid: {result['valid_samples']} samples")
    print(f"\n📁 Files created:")
    print(f"   {result['train_file']}")
    print(f"   {result['valid_file']}")
    print(f"   {result['samples_file']}")
    
    # Show sample
    print("\n📋 Sample training entry:")
    with open(result['train_file']) as f:
        sample = json.loads(f.readline())
    print(f"   Prompt: {sample['prompt'][:80]}...")
    print(f"   Completion: {sample['completion'][:60]}...")


if __name__ == "__main__":
    main()
