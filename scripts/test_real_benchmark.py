"""
Test v6 Model on Real Email Benchmark.

Runs the model on 100 real emails from your MBOX
and measures accuracy per field.

Author: Ranjit Behera
"""

import json
import subprocess
import sys
import re
from pathlib import Path
from collections import defaultdict

MODEL_PATH = "models/base/phi3-finance-base"
ADAPTER_PATH = "models/adapters/finance-lora-v7"
BENCHMARK_FILE = "data/benchmark/real_emails_benchmark.json"


def generate(prompt: str) -> str:
    """Generate response using mlx_lm.generate."""
    cmd = [
        sys.executable, "-m", "mlx_lm.generate",
        "--model", MODEL_PATH,
        "--adapter-path", ADAPTER_PATH,
        "--prompt", prompt,
        "--max-tokens", "200"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.stdout
    except Exception as e:
        return f"Error: {e}"


def parse_json_from_output(output: str) -> dict:
    """Extract JSON from model output."""
    try:
        match = re.search(r'\{[^{}]+\}', output, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return {}


def normalize_value(val: str) -> str:
    """Normalize a value for comparison."""
    if not val:
        return ''
    val = str(val).lower().strip()
    val = val.replace(',', '').replace('.00', '').rstrip('0').rstrip('.')
    return val


def run_real_benchmark(limit: int = 20):
    """Run benchmark on real emails."""
    print("=" * 70)
    print("🧪 REAL EMAIL BENCHMARK - finance-lora-v6")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Adapter: {ADAPTER_PATH}")
    print()
    
    # Load benchmark
    with open(BENCHMARK_FILE) as f:
        benchmark = json.load(f)
    
    # Filter for good candidates (have amount and bank detected)
    good_samples = [s for s in benchmark 
                   if s['expected_entities'].get('amount') 
                   and s['expected_entities'].get('bank')]
    
    if limit:
        good_samples = good_samples[:limit]
    
    print(f"Testing {len(good_samples)} real emails with auto-extracted labels...")
    print()
    
    # Track results
    field_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    bank_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    results = []
    
    for i, sample in enumerate(good_samples):
        text = sample['text']
        expected = sample['expected_entities']
        bank = expected.get('bank', 'unknown')
        
        # Create prompt
        prompt = f"""Extract financial entities from this email:

{text[:500]}

Extract: amount, type, date, account, reference, merchant
Output JSON:"""
        
        # Generate
        output = generate(prompt)
        predicted = parse_json_from_output(output)
        
        # Compare fields
        sample_correct = 0
        sample_total = 0
        
        for field in ['amount', 'type', 'date', 'account', 'reference']:
            exp_val = normalize_value(expected.get(field, ''))
            pred_val = normalize_value(predicted.get(field, ''))
            
            if exp_val:  # Only count if we have expected value
                field_stats[field]['total'] += 1
                sample_total += 1
                if exp_val == pred_val:
                    field_stats[field]['correct'] += 1
                    sample_correct += 1
        
        # Track bank accuracy
        bank_stats[bank]['total'] += 1
        if sample_total > 0 and sample_correct == sample_total:
            bank_stats[bank]['correct'] += 1
        
        results.append({
            'id': sample['id'],
            'expected': expected,
            'predicted': predicted,
            'accuracy': sample_correct / sample_total if sample_total > 0 else 0
        })
        
        # Progress
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(good_samples)}...")
    
    # Print results
    print()
    print("=" * 70)
    print("📈 RESULTS BY FIELD (on REAL emails)")
    print("=" * 70)
    
    for field in ['amount', 'type', 'date', 'account', 'reference']:
        if field in field_stats:
            stats = field_stats[field]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            status = "✅" if acc >= 90 else "⚠️" if acc >= 70 else "❌"
            print(f"  {field:12} {stats['correct']:3}/{stats['total']:3} = {acc:5.1f}% {status}")
    
    print()
    print("=" * 70)
    print("📈 RESULTS BY BANK")
    print("=" * 70)
    
    for bank in sorted(bank_stats.keys()):
        stats = bank_stats[bank]
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        status = "✅" if acc >= 80 else "⚠️" if acc >= 50 else "❌"
        print(f"  {bank.upper():12} {stats['correct']:3}/{stats['total']:3} = {acc:5.1f}% {status}")
    
    # Show some failures
    failures = [r for r in results if r['accuracy'] < 1.0][:3]
    if failures:
        print()
        print("=" * 70)
        print("❌ SAMPLE FAILURES (for debugging)")
        print("=" * 70)
        for f in failures:
            print(f"\n  ID {f['id']}:")
            print(f"    Expected: {f['expected']}")
            print(f"    Predicted: {f['predicted']}")
    
    print()
    print("=" * 70)
    print("✅ Real Email Benchmark Complete!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=15, help='Number of samples to test')
    args = parser.parse_args()
    
    run_real_benchmark(limit=args.limit)
