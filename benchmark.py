#!/usr/bin/env python3
"""
FinEE Benchmark Script
======================

Run this to verify accuracy on your own data.

Usage:
    python benchmark.py                    # Run built-in tests
    python benchmark.py --file data.jsonl  # Test on your data
    python benchmark.py --torture          # Run edge case tests

Author: Ranjit Behera
"""

import json
import time
import argparse
from typing import Dict, List, Any
from dataclasses import dataclass

try:
    from finee import extract, FinEE
    from finee.schema import ExtractionConfig
except ImportError:
    print("Install finee first: pip install finee")
    exit(1)


@dataclass
class BenchmarkResult:
    total: int = 0
    correct: int = 0
    field_accuracy: Dict[str, float] = None
    avg_latency_ms: float = 0
    
    def __post_init__(self):
        if self.field_accuracy is None:
            self.field_accuracy = {}


# ============================================================================
# BUILT-IN BENCHMARK DATA
# ============================================================================

BENCHMARK_DATA = [
    # HDFC Bank
    {
        "text": "HDFC Bank: Rs.2500.00 debited from A/c XX3545 on 28-12-2025 to VPA swiggy@ybl. UPI Ref: 534567891234",
        "expected": {"amount": 2500.0, "type": "debit", "account": "3545", "merchant": "Swiggy", "category": "food"}
    },
    {
        "text": "HDFC: INR 15000 credited to A/c 9876 on 15-01-2025. NEFT from RAHUL SHARMA. Ref: HDFC25011512345",
        "expected": {"amount": 15000.0, "type": "credit", "account": "9876"}
    },
    # ICICI Bank
    {
        "text": "ICICI: Rs.1,250.50 debited from Acct XX4321 on 10-01-25 to amazon@apl. Ref: 987654321012",
        "expected": {"amount": 1250.50, "type": "debit", "account": "4321", "merchant": "Amazon", "category": "shopping"}
    },
    # SBI
    {
        "text": "SBI: Rs.350 debited from a/c XX1234 on 10-01-25. UPI txn to zomato@paytm. Ref: 456789012345",
        "expected": {"amount": 350.0, "type": "debit", "account": "1234", "merchant": "Zomato", "category": "food"}
    },
    # Axis Bank
    {
        "text": "Axis Bank: INR 800.00 debited from A/c 5678 on 05-01-2025. Info: UPI-UBER. Bal: Rs.12,500",
        "expected": {"amount": 800.0, "type": "debit", "account": "5678", "merchant": "Uber", "category": "transport"}
    },
    # Kotak
    {
        "text": "Rs.2000 credited to Kotak A/c XX4321 on 20-01-2025 from rahul.sharma@okicici. Ref: 321654987012",
        "expected": {"amount": 2000.0, "type": "credit", "account": "4321"}
    },
    # Payment Apps
    {
        "text": "PhonePe: Paid Rs.150 to swiggy@ybl from A/c XX1234. UPI Ref: 123456789012",
        "expected": {"amount": 150.0, "type": "debit", "merchant": "Swiggy", "category": "food"}
    },
    {
        "text": "GPay: Sent Rs.500 to uber@paytm from HDFC Bank XX9876. Txn ID: GPY987654321",
        "expected": {"amount": 500.0, "type": "debit", "merchant": "Uber", "category": "transport"}
    },
]


# ============================================================================
# TORTURE TEST DATA (Edge Cases)
# ============================================================================

TORTURE_TESTS = [
    # Missing spaces
    {
        "text": "Rs.500.00debited from HDFC A/c1234 on01-01-25",
        "expected": {"amount": 500.0, "type": "debit", "account": "1234"},
        "difficulty": "Missing spaces"
    },
    # Weird formatting
    {
        "text": "HDFC:Rs 2,500/-debited A/c XX3545 dt:28/12/25 VPA-swiggy@ybl Ref534567891234",
        "expected": {"amount": 2500.0, "type": "debit", "account": "3545"},
        "difficulty": "Non-standard formatting"
    },
    # Mixed case
    {
        "text": "Your A/C XXXX1234 is DEBITED for RS. 1500 on 15-JAN-25. VPA: SWIGGY@YBL",
        "expected": {"amount": 1500.0, "type": "debit", "account": "1234"},
        "difficulty": "Mixed case"
    },
    # Truncated SMS
    {
        "text": "Rs.2500 debited from A/c...3545 to swi...",
        "expected": {"amount": 2500.0, "type": "debit"},
        "difficulty": "Truncated message"
    },
    # Extra noise
    {
        "text": "ALERT! Dear Customer, Rs.500.00 has been debited from your account XX1234 on 01-01-2025. For disputes call 1800-XXX-XXXX. Ignore if done by you.",
        "expected": {"amount": 500.0, "type": "debit", "account": "1234"},
        "difficulty": "Extra noise/marketing"
    },
    # Multiple amounts
    {
        "text": "Rs.500 debited from A/c 1234. Bal: Rs.15,000. Min due: Rs.2000",
        "expected": {"amount": 500.0, "type": "debit", "account": "1234"},
        "difficulty": "Multiple amounts (balance, due)"
    },
    # Unicode symbols
    {
        "text": "₹2,500 debited from A/c •••• 3545 on 28-12-25",
        "expected": {"amount": 2500.0, "type": "debit", "account": "3545"},
        "difficulty": "Unicode symbols (₹, •)"
    },
    # Lakhs notation
    {
        "text": "INR 1.5 Lakh credited to your A/c 9876 on 15-01-25",
        "expected": {"amount": 150000.0, "type": "credit", "account": "9876"},
        "difficulty": "Lakhs notation"
    },
]


def normalize(val):
    """Normalize value for comparison."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if hasattr(val, 'value'):  # Enum
        return val.value.lower()
    return str(val).lower().strip()


def compare(expected: Dict, result) -> Dict[str, bool]:
    """Compare expected vs actual."""
    matches = {}
    for field, exp_val in expected.items():
        actual_val = getattr(result, field, None)
        exp_norm = normalize(exp_val)
        act_norm = normalize(actual_val)
        matches[field] = exp_norm == act_norm
    return matches


def run_benchmark(data: List[Dict], name: str = "Benchmark") -> BenchmarkResult:
    """Run benchmark on dataset."""
    result = BenchmarkResult()
    result.total = len(data)
    
    field_correct = {}
    field_total = {}
    latencies = []
    
    print(f"\n{'='*70}")
    print(f"📊 {name} ({len(data)} samples)")
    print(f"{'='*70}\n")
    
    for i, sample in enumerate(data):
        text = sample["text"]
        expected = sample["expected"]
        difficulty = sample.get("difficulty", "")
        
        start = time.time()
        r = extract(text)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        
        matches = compare(expected, r)
        all_match = all(matches.values())
        
        if all_match:
            result.correct += 1
            status = "✅"
        else:
            status = "❌"
            
        # Track field accuracy
        for field, matched in matches.items():
            if field not in field_total:
                field_total[field] = 0
                field_correct[field] = 0
            field_total[field] += 1
            if matched:
                field_correct[field] += 1
        
        # Print result
        if difficulty:
            print(f"{status} [{difficulty}]")
        else:
            print(f"{status} Sample {i+1}")
            
        if not all_match:
            print(f"   Input: {text[:60]}...")
            for field, matched in matches.items():
                if not matched:
                    actual = getattr(r, field, None)
                    exp = expected[field]
                    print(f"   {field}: expected={exp}, got={actual}")
        print()
    
    # Calculate field accuracy
    result.field_accuracy = {
        field: field_correct[field] / field_total[field] * 100
        for field in field_total
    }
    result.avg_latency_ms = sum(latencies) / len(latencies)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"📈 SUMMARY: {name}")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {result.correct}/{result.total} ({result.correct/result.total*100:.1f}%)")
    print(f"Average Latency: {result.avg_latency_ms:.2f}ms")
    print(f"\nField Accuracy:")
    for field, acc in sorted(result.field_accuracy.items()):
        status = "✅" if acc >= 90 else "⚠️" if acc >= 70 else "❌"
        print(f"  {field:12} {acc:5.1f}% {status}")
    print(f"{'='*70}\n")
    
    return result


def run_user_file(filepath: str) -> BenchmarkResult:
    """Run benchmark on user's JSONL file."""
    data = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return run_benchmark(data, f"User Data ({filepath})")


def main():
    parser = argparse.ArgumentParser(description="FinEE Benchmark")
    parser.add_argument("--file", "-f", help="Path to JSONL file with test data")
    parser.add_argument("--torture", "-t", action="store_true", help="Run torture tests (edge cases)")
    parser.add_argument("--all", "-a", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🏦 FinEE BENCHMARK SUITE")
    print("="*70)
    print("Testing extraction accuracy on Indian banking messages...")
    
    if args.file:
        run_user_file(args.file)
    elif args.torture:
        run_benchmark(TORTURE_TESTS, "Torture Tests (Edge Cases)")
    elif args.all:
        run_benchmark(BENCHMARK_DATA, "Standard Benchmark")
        run_benchmark(TORTURE_TESTS, "Torture Tests (Edge Cases)")
    else:
        run_benchmark(BENCHMARK_DATA, "Standard Benchmark")
    
    print("\n✅ Benchmark complete!")
    print("To test on your own data:")
    print('  python benchmark.py --file your_data.jsonl')
    print("\nJSONL format:")
    print('  {"text": "Rs.500 debited...", "expected": {"amount": 500, "type": "debit"}}')


if __name__ == "__main__":
    main()
