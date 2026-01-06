"""
Payment App Statement Generator for Phase 3.

Generates synthetic training data for PhonePe, GPay, and Paytm
statement formats with proper prefixes and entity labeling.

Supported Apps:
    - PhonePe: [PHONEPE] prefix
    - GPay: [GPAY] prefix  
    - Paytm: [PAYTM] prefix

Example:
    >>> from scripts.generate_payment_app_data import generate_all
    >>> result = generate_all(samples_per_app=300)

Author: Ranjit Behera
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Seed for reproducibility
random.seed(42)

# PhonePe statement formats
PHONEPE_FORMATS = [
    # Transaction history format
    "{date} | {type_text} | {merchant} | ₹{amount} | {status}",
    "{date} {time} | {merchant} | {type_text} Rs.{amount} | Txn ID: {ref}",
    "PhonePe: {type_text} of ₹{amount} to {merchant} on {date}. UPI Ref: {ref}",
    "{date} - {merchant}@ybl - ₹{amount} - {status} - Ref: {ref}",
    "Transaction: {type_text} | Amount: ₹{amount} | To: {merchant} | {date}",
    "{type_text}: ₹{amount} | {merchant} | {date} {time} | ID: {ref}",
]

# GPay statement formats
GPAY_FORMATS = [
    # Google Pay export format
    "{date},{merchant},{amount},{type_text},{status},{upi_id},{ref}",
    "Google Pay: {type_text} of ₹{amount} to {merchant}. {date}. Ref {ref}",
    "{date} | {merchant} | ₹{amount} {type_text} | UPI: {upi_id} | {ref}",
    "You {action} ₹{amount} {direction} {merchant}. {date}. UPI Ref: {ref}. -Google Pay",
    "GPay Transaction: {date} | {merchant} | {type_text} ₹{amount} | Ref: {ref}",
    "{date} {time} - {type_text} - {merchant} - Rs {amount} - {ref}",
]

# Paytm statement formats
PAYTM_FORMATS = [
    # Paytm history format
    "{date} | {merchant} | {type_text} | ₹{amount} | {wallet_balance}",
    "Paytm: {type_text} of Rs.{amount} to {merchant}. {date}. Order ID: {ref}",
    "{date} {time} | {type_text} ₹{amount} | {merchant} | Paytm | Ref: {ref}",
    "You {action} Rs.{amount} to {merchant} using Paytm on {date}. ID: {ref}",
    "Transaction: {date} | {merchant} | Rs {amount} | Type: {type_text} | {status}",
    "Paytm Wallet: {type_text} Rs.{amount} | {merchant} | Balance: ₹{wallet_balance} | {date}",
]

# Merchants by category
MERCHANTS_BY_CATEGORY = {
    "food": [
        "Swiggy", "Zomato", "Dominos", "McDonalds", "KFC", "Pizza Hut",
        "Burger King", "Starbucks", "Cafe Coffee Day", "Subway",
        "Behrouz Biryani", "Faasos", "Box8", "EatFit", "Haldirams"
    ],
    "shopping": [
        "Amazon", "Flipkart", "Myntra", "Ajio", "Nykaa", "Meesho",
        "Snapdeal", "Shopclues", "Tata Cliq", "FirstCry",
        "Bewakoof", "Urbanic", "Shein", "H&M", "Zara"
    ],
    "grocery": [
        "BigBasket", "Zepto", "Blinkit", "Dunzo", "JioMart",
        "Amazon Fresh", "Swiggy Instamart", "DMart Ready",
        "Grofers", "Nature's Basket", "Spencer's", "More Supermarket"
    ],
    "transport": [
        "Uber", "Ola", "Rapido", "BluSmart", "IRCTC",
        "RedBus", "AbhiBus", "MakeMyTrip", "Goibibo", "Yatra",
        "Cleartrip", "EaseMyTrip", "IndiGo", "SpiceJet", "Air India"
    ],
    "bills": [
        "Airtel", "Jio", "Vodafone Idea", "BSNL", "ACT Fibernet",
        "Tata Power", "Adani Electricity", "MSEB", "BESCOM",
        "Mahanagar Gas", "Indraprastha Gas", "Gujarat Gas"
    ],
    "entertainment": [
        "Netflix", "Amazon Prime", "Hotstar", "Zee5", "SonyLiv",
        "Spotify", "Gaana", "JioSaavn", "Apple Music", "YouTube Premium",
        "BookMyShow", "PVR", "INOX", "Carnival Cinemas"
    ],
    "recharge": [
        "Airtel Prepaid", "Jio Prepaid", "Vi Prepaid", "BSNL Mobile",
        "Airtel DTH", "Tata Play", "Dish TV", "d2h", "Sun Direct"
    ],
    "transfer": [
        "Self Transfer", "Rahul Kumar", "Priya Sharma", "Amit Singh",
        "Neha Patel", "Vikram Reddy", "Bank Transfer", "UPI Transfer"
    ],
    "investment": [
        "Zerodha", "Groww", "Upstox", "Angel One", "5paisa",
        "Coin by Zerodha", "Kuvera", "INDmoney", "ET Money",
        "Paytm Money", "PhonePe Mutual Funds", "Scripbox"
    ],
    "insurance": [
        "LIC", "HDFC Life", "ICICI Pru", "SBI Life", "Max Life",
        "Bajaj Allianz", "Tata AIA", "PolicyBazaar", "Digit Insurance"
    ],
}

# UPI IDs by app
UPI_SUFFIXES = {
    "phonepe": ["@ybl", "@ibl", "@axl"],
    "gpay": ["@okaxis", "@okhdfcbank", "@okicici", "@oksbi"],
    "paytm": ["@paytm", "@pthdfc", "@ptaxis", "@ptsbi"],
}

# Status options
STATUSES = ["Success", "Successful", "Completed", "Done", "Processed"]
FAILED_STATUSES = ["Failed", "Declined", "Cancelled", "Pending"]


def random_date(days_back: int = 180) -> Tuple[str, str]:
    """Generate random date and time."""
    days_ago = random.randint(0, days_back)
    dt = datetime.now() - timedelta(days=days_ago)
    
    date_formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%d %b %Y", "%d %b, %Y",
        "%Y-%m-%d", "%d-%m-%y", "%b %d, %Y"
    ]
    time_formats = ["%H:%M", "%I:%M %p", "%H:%M:%S"]
    
    date_str = dt.strftime(random.choice(date_formats))
    time_str = dt.strftime(random.choice(time_formats))
    
    return date_str, time_str


def random_amount(category: str = None) -> str:
    """Generate random amount based on category."""
    ranges = {
        "food": (50, 2000),
        "shopping": (200, 15000),
        "grocery": (100, 5000),
        "transport": (50, 5000),
        "bills": (200, 10000),
        "entertainment": (99, 1500),
        "recharge": (100, 2000),
        "transfer": (500, 50000),
        "investment": (500, 50000),
        "insurance": (1000, 30000),
    }
    
    min_val, max_val = ranges.get(category, (50, 10000))
    amount = random.uniform(min_val, max_val)
    
    if random.random() < 0.4:
        return f"{amount:,.2f}"
    else:
        return f"{int(amount):,}"


def random_ref(prefix: str = "") -> str:
    """Generate random reference number."""
    length = random.choice([10, 12, 14, 16])
    ref = ''.join(str(random.randint(0, 9)) for _ in range(length))
    return f"{prefix}{ref}" if prefix else ref


def random_wallet_balance() -> str:
    """Generate random wallet balance."""
    balance = random.uniform(100, 10000)
    return f"{balance:,.2f}"


def generate_phonepe_row() -> Dict[str, Any]:
    """Generate a PhonePe statement row."""
    category = random.choice(list(MERCHANTS_BY_CATEGORY.keys()))
    merchant = random.choice(MERCHANTS_BY_CATEGORY[category])
    is_credit = category == "transfer" and random.random() < 0.3
    
    date_str, time_str = random_date()
    amount = random_amount(category)
    ref = random_ref()
    status = random.choice(STATUSES)
    upi_suffix = random.choice(UPI_SUFFIXES["phonepe"])
    
    type_text = "Received" if is_credit else "Paid"
    
    template = random.choice(PHONEPE_FORMATS)
    raw_text = template.format(
        date=date_str,
        time=time_str,
        merchant=merchant,
        amount=amount,
        type_text=type_text,
        status=status,
        ref=ref,
        upi_id=f"{merchant.lower().replace(' ', '')}{upi_suffix}"
    )
    
    entities = {
        "date": date_str,
        "amount": amount.replace(",", ""),
        "type": "credit" if is_credit else "debit",
        "merchant": merchant.lower(),
        "category": category,
        "reference": ref,
        "status": status.lower(),
    }
    
    return {
        "app": "phonepe",
        "prefix": "[PHONEPE]",
        "raw_text": raw_text,
        "labeled": True,
        "entities": entities
    }


def generate_gpay_row() -> Dict[str, Any]:
    """Generate a GPay statement row."""
    category = random.choice(list(MERCHANTS_BY_CATEGORY.keys()))
    merchant = random.choice(MERCHANTS_BY_CATEGORY[category])
    is_credit = category == "transfer" and random.random() < 0.3
    
    date_str, time_str = random_date()
    amount = random_amount(category)
    ref = random_ref()
    status = random.choice(STATUSES)
    upi_suffix = random.choice(UPI_SUFFIXES["gpay"])
    upi_id = f"{merchant.lower().replace(' ', '')}{upi_suffix}"
    
    type_text = "Credit" if is_credit else "Debit"
    action = "received" if is_credit else "paid"
    direction = "from" if is_credit else "to"
    
    template = random.choice(GPAY_FORMATS)
    raw_text = template.format(
        date=date_str,
        time=time_str,
        merchant=merchant,
        amount=amount,
        type_text=type_text,
        status=status,
        ref=ref,
        upi_id=upi_id,
        action=action,
        direction=direction
    )
    
    entities = {
        "date": date_str,
        "amount": amount.replace(",", ""),
        "type": "credit" if is_credit else "debit",
        "merchant": merchant.lower(),
        "category": category,
        "reference": ref,
    }
    
    return {
        "app": "gpay",
        "prefix": "[GPAY]",
        "raw_text": raw_text,
        "labeled": True,
        "entities": entities
    }


def generate_paytm_row() -> Dict[str, Any]:
    """Generate a Paytm statement row."""
    category = random.choice(list(MERCHANTS_BY_CATEGORY.keys()))
    merchant = random.choice(MERCHANTS_BY_CATEGORY[category])
    is_credit = category == "transfer" and random.random() < 0.3
    
    date_str, time_str = random_date()
    amount = random_amount(category)
    ref = random_ref("ORD")
    status = random.choice(STATUSES)
    wallet_balance = random_wallet_balance()
    
    type_text = "Credit" if is_credit else "Debit"
    action = "received" if is_credit else "sent"
    
    template = random.choice(PAYTM_FORMATS)
    raw_text = template.format(
        date=date_str,
        time=time_str,
        merchant=merchant,
        amount=amount,
        type_text=type_text,
        status=status,
        ref=ref,
        wallet_balance=wallet_balance,
        action=action
    )
    
    entities = {
        "date": date_str,
        "amount": amount.replace(",", ""),
        "type": "credit" if is_credit else "debit",
        "merchant": merchant.lower(),
        "category": category,
        "reference": ref,
    }
    
    if "Wallet" in template:
        entities["wallet_balance"] = wallet_balance.replace(",", "")
    
    return {
        "app": "paytm",
        "prefix": "[PAYTM]",
        "raw_text": raw_text,
        "labeled": True,
        "entities": entities
    }


def generate_all(
    samples_per_app: int = 300,
    output_dir: str = "data/training"
) -> Dict[str, Any]:
    """
    Generate complete training dataset for all payment apps.
    
    Args:
        samples_per_app: Number of samples per app.
        output_dir: Output directory for JSONL files.
    
    Returns:
        Summary dictionary with stats.
    """
    generators = {
        "phonepe": generate_phonepe_row,
        "gpay": generate_gpay_row,
        "paytm": generate_paytm_row,
    }
    
    all_samples = []
    
    for app, generator in generators.items():
        for _ in range(samples_per_app):
            sample = generator()
            all_samples.append(sample)
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Convert to training format with app-specific prefix
    training_data = []
    for sample in all_samples:
        prefix = sample["prefix"]
        prompt = f"{prefix} Extract financial entities from this payment app statement:\n\n{sample['raw_text']}"
        completion = json.dumps(sample["entities"], indent=2)
        training_data.append({
            "prompt": prompt,
            "completion": completion,
            "app": sample["app"]  # Keep for analysis
        })
    
    # Split train/valid
    split_idx = int(len(training_data) * 0.9)
    train_data = training_data[:split_idx]
    valid_data = training_data[split_idx:]
    
    # Save files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_file = output_path / "payment_apps_train.jsonl"
    valid_file = output_path / "payment_apps_valid.jsonl"
    
    # Remove app field before saving (just for tracking)
    for filepath, data in [(train_file, train_data), (valid_file, valid_data)]:
        with open(filepath, 'w') as f:
            for item in data:
                save_item = {"prompt": item["prompt"], "completion": item["completion"]}
                f.write(json.dumps(save_item) + '\n')
    
    # Save raw samples for reference
    samples_file = output_path / "payment_apps_samples.json"
    with open(samples_file, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    # Stats by app
    app_counts = {}
    for sample in all_samples:
        app = sample["app"]
        app_counts[app] = app_counts.get(app, 0) + 1
    
    return {
        "total_samples": len(all_samples),
        "train_samples": len(train_data),
        "valid_samples": len(valid_data),
        "by_app": app_counts,
        "train_file": str(train_file),
        "valid_file": str(valid_file),
        "samples_file": str(samples_file)
    }


def main():
    """Generate Phase 3 training data."""
    print("💳 Generating Phase 3: Payment App Statement Data")
    print("=" * 60)
    
    result = generate_all(samples_per_app=300)
    
    print(f"\n✅ Generated {result['total_samples']} samples")
    print(f"\n📱 By App:")
    for app, count in result['by_app'].items():
        prefix = {"phonepe": "[PHONEPE]", "gpay": "[GPAY]", "paytm": "[PAYTM]"}[app]
        print(f"   {app.upper():10} {prefix:12} {count} samples")
    
    print(f"\n📊 Split:")
    print(f"   Train: {result['train_samples']} samples")
    print(f"   Valid: {result['valid_samples']} samples")
    
    print(f"\n📁 Files created:")
    print(f"   {result['train_file']}")
    print(f"   {result['valid_file']}")
    print(f"   {result['samples_file']}")
    
    # Show sample
    print("\n📋 Sample entries:")
    with open(result['train_file']) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            sample = json.loads(line)
            print(f"\n   [{i+1}] {sample['prompt'][:80]}...")


if __name__ == "__main__":
    main()
