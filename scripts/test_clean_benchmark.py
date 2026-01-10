"""
Test Model on Clean UPI Benchmark.

Author: Ranjit Behera
"""

import json
import subprocess
import sys
import re
from collections import defaultdict

MODEL_PATH = "models/base/phi3-finance-base"
ADAPTER_PATH = "models/adapters/finance-lora-v7"
BENCHMARK_FILE = "data/benchmark/clean_upi_benchmark.json"


def generate(prompt: str) -> str:
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
    try:
        match = re.search(r'\{[^{}]+\}', output, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return {}


def normalize(val: str) -> str:
    if not val:
        return ''
    val = str(val).lower().strip()
    val = val.replace(',', '').replace('.00', '').rstrip('0').rstrip('.')
    return val


def run_test(limit: int = 20):
    print("=" * 70)
    print("🧪 CLEAN UPI BENCHMARK TEST - v7")
    print("=" * 70)
    
    with open(BENCHMARK_FILE) as f:
        benchmark = json.load(f)
    
    if limit:
        benchmark = benchmark[:limit]
    
    print(f"Testing {len(benchmark)} clean HDFC UPI emails...")
    
    field_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for i, sample in enumerate(benchmark):
        text = sample['text']
        expected = sample['expected_entities']
        
        prompt = f"""Extract financial entities from this HDFC Bank email:

{text[:500]}

Extract: amount, type, date, account, reference, merchant
Output JSON:"""
        
        output = generate(prompt)
        predicted = parse_json_from_output(output)
        
        for field in ['amount', 'type', 'date', 'account', 'reference']:
            exp_val = normalize(expected.get(field, ''))
            pred_val = normalize(predicted.get(field, ''))
            
            if exp_val:
                field_stats[field]['total'] += 1
                if exp_val == pred_val:
                    field_stats[field]['correct'] += 1
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(benchmark)}...")
    
    print()
    print("=" * 70)
    print("📈 CLEAN UPI BENCHMARK RESULTS")
    print("=" * 70)
    
    total_correct = 0
    total_fields = 0
    
    for field in ['amount', 'type', 'date', 'account', 'reference']:
        stats = field_stats[field]
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        status = "✅" if acc >= 90 else "⚠️" if acc >= 70 else "❌"
        print(f"  {field:12} {stats['correct']:3}/{stats['total']:3} = {acc:5.1f}% {status}")
        total_correct += stats['correct']
        total_fields += stats['total']
    
    overall = total_correct / total_fields * 100 if total_fields > 0 else 0
    print(f"\n  {'OVERALL':12} {total_correct:3}/{total_fields:3} = {overall:5.1f}%")
    
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()
    run_test(limit=args.limit)
