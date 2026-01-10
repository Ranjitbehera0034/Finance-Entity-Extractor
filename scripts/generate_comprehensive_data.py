"""
Generate Comprehensive Multi-Bank Training Data.

Uses realistic templates for HDFC, ICICI, SBI, Axis, Kotak
to generate training samples for all transaction types.

Author: Ranjit Behera
"""

import json
import random
from pathlib import Path
from datetime import datetime, timedelta

OUTPUT_FILE = Path("data/synthetic/multi_bank_comprehensive.jsonl")

# Bank-specific email templates
TEMPLATES = {
    "hdfc": {
        "upi_debit": [
            "HDFC BANK Dear Customer,\nRs.{amount} has been debited from account {account} to VPA {merchant}@{vpa_suffix} {merchant_name} on {date}.\nYour UPI transaction reference number is {reference}.\nIf you did not authorize this transaction, please call 18002586161.",
            "Dear Customer, INR {amount} has been debited from your HDFC Bank Account XX{account} on {date} by UPI.\n\nTransaction Details:\nAmount: INR {amount}\nUPI Reference: {reference}\nBeneficiary: {merchant}@{vpa_suffix}\nRemarks: {remarks}\nAvailable Balance: INR {balance}",
        ],
        "upi_credit": [
            "HDFC BANK Dear Customer,\nRs.{amount} has been credited to account {account} from VPA {sender}@{vpa_suffix} {sender_name} on {date}.\nYour UPI transaction reference number is {reference}.",
            "Dear Customer, INR {amount} has been credited to your HDFC Bank Account XX{account} on {date} by UPI.\n\nUPI Reference: {reference}\nSender: {sender}@{vpa_suffix}\nRemarks: {remarks}\nAvailable Balance: INR {balance}",
        ],
        "neft_credit": [
            "Rs.{amount} has been credited to your HDFC Bank A/c XX{account} on {date} through NEFT.\n\nNEFT Reference: HDFC{ref_prefix}N{reference}\nSender Name: {sender_name}\nSender Bank: {sender_bank}\nRemarks: {remarks}\nAvailable Balance: Rs.{balance}",
        ],
        "atm": [
            "Rs.{amount} has been withdrawn from your HDFC Bank Account XX{account} at ATM.\n\nDate: {date}\nATM ID: HDFC{atm_id}\nLocation: {location}\nReference: ATM{reference}\nAvailable Balance: Rs.{balance}",
        ],
    },
    "icici": {
        "upi_debit": [
            "Dear Customer, Rs. {amount} has been debited from your ICICI Bank Account ending with {account} on {date} at {time}.\n\nMode: UPI\nRef No: {reference}\nTo VPA: {merchant}@{vpa_suffix}\nNarration: {remarks}\nUpdated Balance: Rs. {balance}",
            "Rs.{amount} has been debited from your ICICI Bank Account {account} for UPI txn to VPA-{merchant}@{vpa_suffix} on {date}. Ref:{reference}",
        ],
        "upi_credit": [
            "Dear Customer, Rs. {amount} has been credited to your ICICI Bank Account ending with {account} on {date} at {time}.\n\nMode: UPI\nRef No: {reference}\nFrom VPA: {sender}@{vpa_suffix}\nNarration: {remarks}\nUpdated Balance: Rs. {balance}",
            "INR {amount} credited to ICICI Bank A/c {account} on {date} from {sender_name}. UPI Ref: {reference}",
        ],
        "imps_credit": [
            "ICICI Bank Acct XX{account} credited with INR {amount} on {date}. IMPS Ref {reference}.",
        ],
        "neft_credit": [
            "Rs.{amount} has been credited to your ICICI Bank Account {account} via NEFT.\n\nDate: {date}\nUTR: ICIC{ref_prefix}N{reference}\nSender: {sender_name}\nSender A/c: XXXXXX{sender_acc}\nBank: {sender_bank}\nNarration: {remarks}\nBalance: Rs.{balance}",
        ],
    },
    "sbi": {
        "upi_debit": [
            "SBI: Rs.{amount} debited from a/c XX{account} on {date}. UPI txn to {merchant}@{vpa_suffix}. If not done by you,call 1800112211. Ref: {reference}",
            "Dear SBI User, Rs {amount} debited from A/c X{account} on {date} by UPI {merchant_name}. Ref {reference}. If not you, fwd SMS to 9223766666",
        ],
        "upi_credit": [
            "Dear SBI User, Rs {amount} credited to A/c X{account} on {date}. UPI Ref {reference}.",
            "Rs {amount} credited to SBI A/c {account} on {date}. IMPS from {sender_name}. Balance: Rs.{balance}",
        ],
        "neft_credit": [
            "Dear Customer, Rs.{amount} has been credited to your SBI Account XXXXXXX{account} on {date} through NEFT.\n\nUTR No: SBIN{ref_prefix}{reference}\nRemitter: {sender_name}\nRemitter Bank: {sender_bank}\nPurpose: {remarks}\nBalance: Rs.{balance}",
        ],
    },
    "axis": {
        "upi_debit": [
            "Rs.{amount} debited from Axis Bank A/c {account} on {date} to VPA {merchant}@{vpa_suffix}. UPI Ref:{reference}",
            "Axis Bank Acct XX{account} debited for Rs.{amount} on {date}. UPI:{merchant_name}. Ref {reference}. Not you? SMS FREEZE to 5676782",
            "Dear Customer, INR {amount} debited from Axis A/c XX{account} on {date}. Info:UPI-{merchant_name}. Bal:Rs.{balance}",
        ],
        "upi_credit": [
            "Rs.{amount} credited to Axis Bank A/c {account} on {date} from {sender_name}. Ref:{reference}",
            "Dear Customer, INR {amount} credited to Axis A/c XX{account} on {date}. Avl Bal:Rs.{balance}",
        ],
        "imps_credit": [
            "INR {amount} has been credited to your Axis Bank Account XX{account} via IMPS on {date} at {time}.\n\nIMPS Ref No: {reference}\nSender: {sender_name}\nSender A/c: XXXXXX{sender_acc}\nSender Bank: {sender_bank}\nRemarks: {remarks}\nAvl Bal: INR {balance}",
        ],
    },
    "kotak": {
        "upi_debit": [
            "INR {amount} debited from Kotak A/c {account} on {date} to VPA:{merchant}@{vpa_suffix}. Ref No.{reference}",
            "Kotak Bank: Rs.{amount} debited from A/c XX{account} on {date} towards UPI-{merchant_name}. Avl Bal Rs.{balance}. Ref {reference}",
            "Rs {amount} debited via UPI from Kotak Bank A/c XX{account} on {date}. Merchant:{merchant_name}. If not you call 18002740110",
        ],
        "upi_credit": [
            "INR {amount} credited to Kotak A/c {account} on {date} from {sender_name}. Ref:{reference}",
            "Rs {amount} credited to Kotak Bank A/c XX{account} on {date}. Info:{remarks}. Balance:Rs.{balance}",
        ],
    },
}

# Merchants and categories
MERCHANTS = [
    ("swiggy", "food", "SWIGGY INDIA"),
    ("zomato", "food", "ZOMATO MEDIA"),
    ("amazon", "shopping", "AMAZON SELLER"),
    ("flipkart", "shopping", "FLIPKART PAYMENTS"),
    ("myntra", "shopping", "MYNTRA DESIGNS"),
    ("uber", "transport", "UBER INDIA"),
    ("ola", "transport", "OLA CABS"),
    ("rapido", "transport", "RAPIDO BIKE"),
    ("bigbasket", "grocery", "BIGBASKET"),
    ("blinkit", "grocery", "BLINKIT QUICK"),
    ("zepto", "grocery", "ZEPTO NOW"),
    ("dmart", "grocery", "DMART RETAIL"),
    ("jio", "bills", "RELIANCE JIO"),
    ("airtel", "bills", "BHARTI AIRTEL"),
    ("electricity", "bills", "ELECTRICITY BOARD"),
    ("water", "bills", "WATER BOARD"),
    ("netflix", "entertainment", "NETFLIX SERVICES"),
    ("hotstar", "entertainment", "DISNEY HOTSTAR"),
    ("bookmyshow", "entertainment", "BOOKMYSHOW"),
    ("makemytrip", "travel", "MAKEMYTRIP"),
]

VPA_SUFFIXES = ["ybl", "paytm", "okicici", "okhdfcbank", "axl", "sbi", "icici", "kotak"]

SENDERS = [
    ("amit.kumar", "AMIT KUMAR"),
    ("priya.singh", "PRIYA SINGH"),
    ("rahul.sharma", "RAHUL SHARMA"),
    ("neha.gupta", "NEHA GUPTA"),
    ("suresh.patel", "SURESH PATEL"),
    ("anita.verma", "ANITA VERMA"),
]

SENDER_BANKS = ["HDFC BANK", "ICICI BANK", "SBI", "AXIS BANK", "KOTAK BANK", "PNB"]

LOCATIONS = [
    "MG Road, Bangalore",
    "Connaught Place, Delhi",
    "Bandra West, Mumbai",
    "Hitech City, Hyderabad",
    "Anna Nagar, Chennai",
]

REMARKS = [
    "Food order", "Shopping", "Bill payment", "Rent share",
    "Grocery", "Transport", "Subscription", "Birthday gift",
    "Salary transfer", "Medical", "Education", "Investment",
]

DATE_FORMATS = ["%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d %b %Y", "%d%m%Y"]


def generate_date():
    """Generate random date in past 90 days."""
    days_ago = random.randint(1, 90)
    d = datetime.now() - timedelta(days=days_ago)
    fmt = random.choice(DATE_FORMATS)
    return d.strftime(fmt)

def generate_time():
    """Generate random time."""
    h = random.randint(6, 23)
    m = random.randint(0, 59)
    s = random.randint(0, 59)
    return f"{h:02d}:{m:02d}:{s:02d}"

def generate_reference():
    """Generate 12-digit reference."""
    return ''.join([str(random.randint(0, 9)) for _ in range(12)])

def generate_account():
    """Generate 4-digit account."""
    return str(random.randint(1000, 9999))

def generate_amount():
    """Generate realistic amount."""
    options = [
        round(random.uniform(50, 500), 2),
        round(random.uniform(100, 2000), 2),
        random.randint(100, 5000),
        random.randint(500, 15000),
        round(random.uniform(1000, 10000), 2),
    ]
    return str(random.choice(options))

def generate_balance():
    """Generate balance."""
    return str(random.randint(10000, 500000))


def generate_samples(n_per_bank=100):
    """Generate comprehensive multi-bank samples."""
    samples = []
    
    for bank, templates in TEMPLATES.items():
        bank_samples = 0
        
        for txn_type, template_list in templates.items():
            for _ in range(n_per_bank // len(templates)):
                template = random.choice(template_list)
                
                merchant_info = random.choice(MERCHANTS)
                sender_info = random.choice(SENDERS)
                
                is_credit = "credit" in txn_type
                
                # Fill template
                data = {
                    "amount": generate_amount(),
                    "account": generate_account(),
                    "date": generate_date(),
                    "time": generate_time(),
                    "reference": generate_reference(),
                    "balance": generate_balance(),
                    "remarks": random.choice(REMARKS),
                    "vpa_suffix": random.choice(VPA_SUFFIXES),
                    "ref_prefix": f"260{random.randint(10, 99)}",
                    "atm_id": f"000{random.randint(1000, 9999)}",
                    "location": random.choice(LOCATIONS),
                    "sender_bank": random.choice(SENDER_BANKS),
                    "sender_acc": str(random.randint(1000, 9999)),
                }
                
                if is_credit:
                    data["sender"] = sender_info[0]
                    data["sender_name"] = sender_info[1]
                else:
                    data["merchant"] = merchant_info[0]
                    data["merchant_name"] = merchant_info[2]
                
                try:
                    email_text = template.format(**data)
                except KeyError:
                    continue
                
                # Create entities
                entities = {
                    "amount": data["amount"].replace(",", ""),
                    "type": "credit" if is_credit else "debit",
                    "date": data["date"],
                    "account": data["account"],
                    "reference": data["reference"],
                    "bank": bank,
                }
                
                if not is_credit:
                    entities["merchant"] = merchant_info[0]
                    entities["category"] = merchant_info[1]
                
                # Create training format
                prompt = f"""Extract financial entities from this {bank.upper()} Bank email:

{email_text}

Extract: amount, type, date, account, reference{', merchant, category' if not is_credit else ''}
Output JSON:"""
                
                completion = json.dumps(entities, indent=2)
                
                samples.append({
                    "prompt": prompt,
                    "completion": completion,
                    "bank": bank,
                    "txn_type": txn_type
                })
                bank_samples += 1
        
        print(f"   {bank.upper():10} {bank_samples} samples")
    
    return samples


def main():
    print("=" * 60)
    print("📊 GENERATING COMPREHENSIVE MULTI-BANK DATA")
    print("=" * 60)
    
    print("\nGenerating samples per bank:")
    samples = generate_samples(n_per_bank=100)
    
    # Shuffle
    random.seed(42)
    random.shuffle(samples)
    
    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"\n✅ Total samples: {len(samples)}")
    print(f"   Saved to {OUTPUT_FILE}")
    
    # Stats
    from collections import Counter
    bank_counts = Counter(s['bank'] for s in samples)
    type_counts = Counter(s['txn_type'] for s in samples)
    
    print("\n📊 By transaction type:")
    for t, c in sorted(type_counts.items()):
        print(f"   {t:15} {c}")


if __name__ == "__main__":
    main()
