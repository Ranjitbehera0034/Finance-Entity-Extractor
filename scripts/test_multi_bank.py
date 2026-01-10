"""
Test Model on Multi-Bank Benchmark.

Author: Ranjit Behera
"""

import json
import subprocess
import sys
import re
from collections import defaultdict

MODEL_PATH = "models/base/phi3-finance-base"
ADAPTER_PATH = "models/adapters/finee-adapter-v1"  # Updated for V1.0
BENCHMARK_FILE = "data/benchmark/multi_bank_comprehensive.jsonl"  # Updated to generated file


def generate(prompt: str, adapter_path: str = ADAPTER_PATH) -> str:
    cmd = [
        sys.executable, "-m", "mlx_lm.generate",
        "--model", MODEL_PATH,
        "--adapter-path", adapter_path,
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


def run_test(limit: int = None, adapter_path: str = ADAPTER_PATH):
    print("=" * 70)
    print("🧪 MULTI-BANK BENCHMARK TEST - v1.0")
    print("=" * 70)
    
    # Handle JSONL format (lines of JSON)
    benchmark = []
    try:
        with open(BENCHMARK_FILE) as f:
            if BENCHMARK_FILE.endswith('.jsonl'):
                for line in f:
                    if line.strip():
                        benchmark.append(json.loads(line))
            else:
                benchmark = json.load(f)
    except FileNotFoundError:
        print(f"Benchmark file not found: {BENCHMARK_FILE}")
        return
    
    if limit:
        benchmark = benchmark[:limit]
    
    print(f"Testing {len(benchmark)} samples across multiple banks...")
    
    field_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    bank_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'fields_correct': 0, 'fields_total': 0})
    
    for i, sample in enumerate(benchmark):
        # Handle difference between generation format and test format if any
        # The generator produces: "prompt", "completion", "bank", "txn_type"
        # The old benchmark expected: "text", "expected_entities"
        
        if 'prompt' in sample and 'completion' in sample:
            # Parse completion JSON string to dict
            try:
                expected = json.loads(sample['completion'])
            except:
                expected = sample['completion']
            
            # Extract text from prompt (it's embedded in the prompt string)
            # The prompt format is: ...\n\n{email_text}\n\n...
            # A simple way is to use the raw text if available, but here we only have prompt.
            # We can pass the prompt DIRECTLY to the model if we trained on it!
            # BUT wait, generate() appends its own prompt template?
            # No, mlx_lm.generate takes a prompt. 
            # If we pass sample['prompt'], it's the FULL prompt including "Output JSON:".
            prompt = sample['prompt']
            
            # For reporting/debugging, we want the bank name
            bank = sample.get('bank', 'unknown')
            
        else:
            # Fallback to old format
            text = sample['text']
            expected = sample['expected_entities']
            bank = expected.get('bank', 'unknown')
            
            prompt = f"""Extract financial entities from this {bank.upper()} Bank email:

{text[:500]}

Extract: amount, type, date, account, reference, merchant
Output JSON:"""
        
        output = generate(prompt, adapter_path)
        predicted = parse_json_from_output(output)
        
        sample_correct = 0
        sample_total = 0
        
        # Define fields to check
        fields_to_check = ['amount', 'type', 'date', 'account', 'reference']
        if 'merchant' in expected: fields_to_check.append('merchant')
        
        for field in fields_to_check:
            exp_val = normalize(expected.get(field, ''))
            pred_val = normalize(predicted.get(field, ''))
            
            if exp_val:
                field_stats[field]['total'] += 1
                bank_stats[bank]['fields_total'] += 1
                sample_total += 1
                
                if exp_val == pred_val:
                    field_stats[field]['correct'] += 1
                    bank_stats[bank]['fields_correct'] += 1
                    sample_correct += 1
        
        # Track bank-level (all fields match)
        bank_stats[bank]['total'] += 1
        if sample_total > 0 and sample_correct == sample_total:
            bank_stats[bank]['correct'] += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(benchmark)}...")
    
    print()
    print("=" * 70)
    print("📈 RESULTS BY FIELD")
    print("=" * 70)
    
    total_correct = 0
    total_fields = 0
    
    for field in sorted(field_stats.keys()):
        stats = field_stats[field]
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        status = "✅" if acc >= 90 else "⚠️" if acc >= 70 else "❌"
        print(f"  {field:12} {stats['correct']:3}/{stats['total']:3} = {acc:5.1f}% {status}")
        total_correct += stats['correct']
        total_fields += stats['total']
    
    overall = total_correct / total_fields * 100 if total_fields > 0 else 0
    print(f"\n  {'OVERALL':12} {total_correct:3}/{total_fields:3} = {overall:5.1f}%")
    
    print()
    print("=" * 70)
    print("📈 RESULTS BY BANK (Field-Level Accuracy)")
    print("=" * 70)
    
    for bank in sorted(bank_stats.keys()):
        stats = bank_stats[bank]
        field_acc = stats['fields_correct'] / stats['fields_total'] * 100 if stats['fields_total'] > 0 else 0
        # full_acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        
        status = "✅" if field_acc >= 90 else "⚠️" if field_acc >= 70 else "❌"
        print(f"  {bank.upper():10} Fields: {stats['fields_correct']:3}/{stats['fields_total']:3} = {field_acc:5.1f}% | Full Match: {stats['correct']}/{stats['total']} {status}")
    
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=40, help='Number of samples')
    parser.add_argument('--adapter', type=str, default=ADAPTER_PATH, help='Adapter path')
    args = parser.parse_args()
    run_test(limit=args.limit, adapter_path=args.adapter)
