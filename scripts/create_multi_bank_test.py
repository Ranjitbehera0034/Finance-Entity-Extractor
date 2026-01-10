"""
Create Multi-Bank Test Benchmark from Synthetic Data.

Creates test samples from the generated data to test v8.
Excludes training data by using different random seed.

Author: Ranjit Behera
"""

import json
import random
from pathlib import Path
from scripts.generate_comprehensive_data import (
    TEMPLATES, MERCHANTS, VPA_SUFFIXES, SENDERS, SENDER_BANKS,
    LOCATIONS, REMARKS, generate_date, generate_time, generate_reference,
    generate_account, generate_amount, generate_balance
)

BENCHMARK_FILE = Path("data/benchmark/multi_bank_test.json")


def generate_test_samples(n_per_bank=10):
    """Generate test samples with different random seed."""
    random.seed(123)  # Different from training
    samples = []
    
    for bank, templates in TEMPLATES.items():
        for txn_type, template_list in templates.items():
            for _ in range(n_per_bank // len(templates)):
                template = random.choice(template_list)
                
                merchant_info = random.choice(MERCHANTS)
                sender_info = random.choice(SENDERS)
                
                is_credit = "credit" in txn_type
                
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
                
                samples.append({
                    "id": len(samples) + 1,
                    "text": email_text,
                    "expected_entities": entities,
                    "txn_type": txn_type,
                    "verified": True
                })
    
    return samples


def main():
    print("=" * 60)
    print("📊 CREATING MULTI-BANK TEST BENCHMARK")
    print("=" * 60)
    
    samples = generate_test_samples(n_per_bank=10)
    random.shuffle(samples)
    
    # Save
    BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_FILE, 'w') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(samples)} test samples to {BENCHMARK_FILE}")
    
    # Stats
    from collections import Counter
    bank_counts = Counter(s['expected_entities']['bank'] for s in samples)
    
    print("\n📊 By bank:")
    for bank, count in sorted(bank_counts.items()):
        print(f"   {bank.upper():10} {count}")


if __name__ == "__main__":
    main()
