"""
Bank Email Templates Generator
Generates synthetic training data for different bank formats.
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta
import re


# ============================================
# BANK EMAIL TEMPLATES
# ============================================

BANK_TEMPLATES = {
    "hdfc": {
        "debit": [
            "HDFC Bank: Rs.{amount} has been debited from A/c **{account} on {date} to VPA {vpa} (UPI Ref No {ref}). Avl Bal Rs.{balance}",
            "Dear Customer, INR {amount} debited from A/c {account} on {date}. Info: UPI-{merchant}. If not done by you, call 18002586161",
            "Rs {amount} debited from HDFC Bank A/c XX{account} on {date} for UPI txn. Ref {ref}. Not you? Call 18002586161",
        ],
        "credit": [
            "HDFC Bank: Rs.{amount} has been credited to A/c **{account} on {date} from VPA {vpa} (UPI Ref No {ref}). Avl Bal Rs.{balance}",
            "Dear Customer, INR {amount} credited to A/c {account} on {date}. Info: UPI-{sender}. Ref: {ref}",
            "Rs {amount} credited to HDFC Bank A/c XX{account} on {date}. UPI Ref {ref}. Thank you for banking with us.",
        ],
    },
    "icici": {
        "debit": [
            "ICICI Bank Acct XX{account} debited with INR {amount} on {date}. IMPS Ref {ref}. Call 18602662 if not done by you.",
            "INR {amount} debited from ICICI Bank A/c {account} on {date} to {merchant} via UPI. Ref: {ref}",
            "Dear Customer, Rs.{amount} is debited from A/c XX{account} for UPI txn to VPA-{vpa} on {date}. Ref:{ref}",
        ],
        "credit": [
            "ICICI Bank Acct XX{account} credited with INR {amount} on {date}. IMPS Ref {ref}.",
            "INR {amount} credited to ICICI Bank A/c {account} on {date} from {sender}. UPI Ref: {ref}",
            "Dear Customer, Rs.{amount} is credited to A/c XX{account} on {date}. Ref:{ref}. Avl Bal: Rs.{balance}",
        ],
    },
    "sbi": {
        "debit": [
            "Dear SBI User, Rs {amount} debited from A/c X{account} on {date} by UPI {merchant}. Ref {ref}. If not you, fwd SMS to 9223766666",
            "SBI: Rs.{amount} debited from a/c XX{account} on {date}. UPI txn to {vpa}. If not done by you,call 1800112211",
            "Rs {amount} debited from SBI A/c {account} on {date} via UPI. Ref No {ref}. Balance: Rs.{balance}",
        ],
        "credit": [
            "Dear SBI User, Rs {amount} credited to A/c X{account} on {date}. UPI Ref {ref}.",
            "SBI: Rs.{amount} credited to a/c XX{account} on {date}. Info: {sender}. Ref:{ref}",
            "Rs {amount} credited to SBI A/c {account} on {date}. IMPS from {sender}. Balance: Rs.{balance}",
        ],
    },
    "axis": {
        "debit": [
            "Axis Bank Acct XX{account} debited for Rs.{amount} on {date}. UPI:{merchant}. Ref {ref}. Not you? SMS FREEZE to 5676782",
            "Rs.{amount} debited from Axis Bank A/c {account} on {date} to VPA {vpa}. UPI Ref:{ref}",
            "Dear Customer, INR {amount} debited from Axis A/c XX{account} on {date}. Info:UPI-{merchant}. Bal:Rs.{balance}",
        ],
        "credit": [
            "Axis Bank Acct XX{account} credited for Rs.{amount} on {date}. UPI Ref {ref}.",
            "Rs.{amount} credited to Axis Bank A/c {account} on {date} from {sender}. Ref:{ref}",
            "Dear Customer, INR {amount} credited to Axis A/c XX{account} on {date}. Avl Bal:Rs.{balance}",
        ],
    },
    "kotak": {
        "debit": [
            "Kotak Bank: Rs.{amount} debited from A/c XX{account} on {date} towards UPI-{merchant}. Avl Bal Rs.{balance}. Ref {ref}",
            "INR {amount} debited from Kotak A/c {account} on {date} to VPA:{vpa}. Ref No.{ref}",
            "Rs {amount} debited via UPI from Kotak Bank A/c XX{account} on {date}. Merchant:{merchant}. If not you call 18002740110",
        ],
        "credit": [
            "Kotak Bank: Rs.{amount} credited to A/c XX{account} on {date}. UPI Ref {ref}. Avl Bal Rs.{balance}",
            "INR {amount} credited to Kotak A/c {account} on {date} from {sender}. Ref:{ref}",
            "Rs {amount} credited to Kotak Bank A/c XX{account} on {date}. Info:{sender}. Balance:Rs.{balance}",
        ],
    },
    "phonepe": {
        "debit": [
            "You paid Rs.{amount} to {merchant} from {bank} Bank a/c XX{account}. Txn ID: {ref}. {date}",
            "PhonePe: Rs.{amount} sent to {vpa} on {date}. Ref: {ref}. Check app for details.",
            "Payment of Rs.{amount} to {merchant} successful! {date}. UPI Ref: {ref}",
        ],
        "credit": [
            "You received Rs.{amount} from {sender} to {bank} Bank a/c XX{account}. {date}. Txn ID: {ref}",
            "PhonePe: Rs.{amount} received from {sender} on {date}. Ref: {ref}",
            "Rs.{amount} credited to your PhonePe wallet from {sender}. {date}",
        ],
    },
    "gpay": {
        "debit": [
            "You paid ₹{amount} to {merchant}. {date}. UPI Ref: {ref}. -Google Pay",
            "₹{amount} sent to {vpa} from {bank} Bank XX{account}. Txn: {ref}. {date}",
            "Google Pay: Payment of ₹{amount} to {merchant} successful. {date}. Ref {ref}",
        ],
        "credit": [
            "You received ₹{amount} from {sender}. {date}. UPI Ref: {ref}. -Google Pay",
            "₹{amount} received from {sender} to {bank} Bank XX{account}. {date}",
            "Google Pay: ₹{amount} credited from {sender}. Ref: {ref}",
        ],
    },
    "paytm": {
        "debit": [
            "Paid Rs.{amount} to {merchant} via Paytm UPI. {date}. Txn ID: {ref}",
            "Rs {amount} debited from your {bank} Bank a/c XX{account} to {vpa}. Paytm Ref: {ref}",
            "Paytm: You sent Rs.{amount} to {merchant}. {date}. Order ID: {ref}",
        ],
        "credit": [
            "Received Rs.{amount} from {sender} via Paytm UPI. {date}. Txn ID: {ref}",
            "Rs {amount} credited to your {bank} Bank a/c XX{account} from {sender}. Ref: {ref}",
            "Paytm: Rs.{amount} added to wallet from {sender}. {date}",
        ],
    },
}

# Common merchants and VPAs
MERCHANTS = {
    "food": [
        ("Swiggy", "swiggy@ybl"),
        ("Zomato", "zomato@paytm"),
        ("Dominos", "dominos@icici"),
        ("McDonalds", "mcd@hdfcbank"),
        ("KFC", "kfc@axisbank"),
        ("Starbucks", "starbucks@ybl"),
    ],
    "shopping": [
        ("Amazon", "amazon@apl"),
        ("Flipkart", "flipkart@axisb"),
        ("Myntra", "myntra@ybl"),
        ("Ajio", "ajio@icici"),
        ("Nykaa", "nykaa@paytm"),
        ("BigBasket", "bigbasket@ybl"),
    ],
    "transport": [
        ("Uber", "uber@axisbank"),
        ("Ola", "ola@ybl"),
        ("Rapido", "rapido@icici"),
        ("Metro", "metro@sbi"),
    ],
    "bills": [
        ("Airtel", "airtel@paytm"),
        ("Jio", "jio@icici"),
        ("Electricity", "bescom@ybl"),
        ("Water", "bwssb@sbi"),
    ],
    "grocery": [
        ("Zepto", "zepto@ybl"),
        ("Blinkit", "blinkit@icici"),
        ("DMart", "dmart@hdfcbank"),
        ("BigBazaar", "bigbazaar@ybl"),
    ],
}

# Sample sender names for credits
SENDERS = [
    "Rahul Sharma", "Priya Singh", "Amit Kumar", "Neha Gupta",
    "Salary - ACME Corp", "Refund - Amazon", "Cashback",
    "Interest Credit", "Dividend - Mutual Fund",
]


class BankEmailGenerator:
    """Generate synthetic bank email training data."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
    
    def generate_amount(self, min_val: int = 50, max_val: int = 50000) -> str:
        """Generate realistic transaction amount."""
        # More small transactions than large ones
        if random.random() < 0.6:
            amount = random.randint(min_val, 2000)
        elif random.random() < 0.85:
            amount = random.randint(2000, 10000)
        else:
            amount = random.randint(10000, max_val)
        
        # Sometimes add decimals
        if random.random() < 0.3:
            amount = amount + random.randint(1, 99) / 100
        
        return f"{amount:.2f}" if isinstance(amount, float) else str(amount)
    
    def generate_date(self, days_back: int = 90) -> str:
        """Generate random date in various formats."""
        days_ago = random.randint(0, days_back)
        date = datetime.now() - timedelta(days=days_ago)
        
        formats = [
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%d-%m-%y",
            "%d %b %Y",
            "%d%m%Y",
        ]
        return date.strftime(random.choice(formats))
    
    def generate_reference(self) -> str:
        """Generate UPI reference number."""
        return str(random.randint(100000000000, 999999999999))
    
    def generate_account(self) -> str:
        """Generate masked account number."""
        return str(random.randint(1000, 9999))
    
    def generate_balance(self, amount: str, txn_type: str) -> str:
        """Generate plausible balance."""
        amt = float(amount.replace(",", ""))
        if txn_type == "debit":
            balance = random.randint(int(amt * 2), int(amt * 50))
        else:
            balance = random.randint(int(amt), int(amt * 30))
        return f"{balance:.2f}"
    
    def generate_email(self, bank: str, txn_type: str) -> Dict:
        """Generate a single synthetic email."""
        templates = BANK_TEMPLATES.get(bank, BANK_TEMPLATES["hdfc"])
        template = random.choice(templates[txn_type])
        
        # Pick merchant/sender
        if txn_type == "debit":
            category = random.choice(list(MERCHANTS.keys()))
            merchant_name, vpa = random.choice(MERCHANTS[category])
        else:
            merchant_name = random.choice(SENDERS)
            vpa = f"{merchant_name.lower().replace(' ', '')}@upi"
            category = "transfer"
        
        amount = self.generate_amount()
        date = self.generate_date()
        ref = self.generate_reference()
        account = self.generate_account()
        balance = self.generate_balance(amount, txn_type)
        
        # Fill template
        email_body = template.format(
            amount=amount,
            date=date,
            account=account,
            vpa=vpa,
            merchant=merchant_name,
            sender=merchant_name,
            ref=ref,
            balance=balance,
            bank=bank.upper(),
        )
        
        # Expected extraction
        entities = {
            "amount": amount.replace(",", ""),
            "type": txn_type,
            "date": date,
            "account": account,
            "reference": ref,
        }
        
        # Add category for debit only
        if txn_type == "debit":
            entities["merchant"] = merchant_name.lower()
            entities["category"] = category
        
        return {
            "bank": bank,
            "subject": f"Transaction Alert - {bank.upper()} Bank",
            "body": email_body,
            "entities": entities,
            "raw_text": email_body,
        }
    
    def generate_dataset(
        self,
        num_samples: int = 1500,
        banks: List[str] = None,
        output_file: str = None
    ) -> List[Dict]:
        """Generate a full training dataset."""
        
        if banks is None:
            banks = list(BANK_TEMPLATES.keys())
        
        samples_per_bank = num_samples // len(banks)
        dataset = []
        
        print(f"🏧 Generating {num_samples} samples across {len(banks)} banks...")
        
        for bank in banks:
            # 60% debit, 40% credit
            debit_count = int(samples_per_bank * 0.6)
            credit_count = samples_per_bank - debit_count
            
            for _ in range(debit_count):
                dataset.append(self.generate_email(bank, "debit"))
            
            for _ in range(credit_count):
                dataset.append(self.generate_email(bank, "credit"))
            
            print(f"  ✅ {bank.upper()}: {samples_per_bank} samples")
        
        # Shuffle
        random.shuffle(dataset)
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(dataset, f, indent=2)
            print(f"\n💾 Saved to: {output_path}")
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total samples: {len(dataset)}")
        print(f"   Banks covered: {', '.join(banks)}")
        
        return dataset
    
    def convert_to_training_format(self, dataset: List[Dict], output_file: str):
        """Convert dataset to JSONL training format."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            for item in dataset:
                training_example = {
                    "prompt": f"Extract financial entities from this email:\n\nSubject: {item['subject']}\n\nBody: {item['body']}",
                    "completion": json.dumps(item["entities"], indent=2)
                }
                f.write(json.dumps(training_example) + "\n")
        
        print(f"💾 Training file saved: {output_path}")


if __name__ == "__main__":
    generator = BankEmailGenerator()
    
    # Generate full dataset
    dataset = generator.generate_dataset(
        num_samples=1500,
        output_file="data/synthetic/bank_emails.json"
    )
    
    # Convert to training format
    generator.convert_to_training_format(
        dataset,
        "data/training/synthetic_train.jsonl"
    )
    
    # Show samples
    print("\n📧 Sample emails:")
    for sample in dataset[:3]:
        print(f"\n[{sample['bank'].upper()}] {sample['entities']['type'].upper()}")
        print(f"  {sample['body'][:100]}...")
        print(f"  → {sample['entities']}")
