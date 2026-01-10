"""
Quantitative Evaluation for finance-lora-v6.
Tests the model on the full test dataset and computes accuracy metrics.

Author: Ranjit Behera
"""

import json
import subprocess
import sys
import re
from pathlib import Path
from collections import defaultdict

MODEL_PATH = "models/base/phi3-finance-base"
ADAPTER_PATH = "models/adapters/finance-lora-v6"
TEST_FILE = "data/synthetic/test_emails.json"


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
        # Find JSON in output
        match = re.search(r'\{[^{}]+\}', output, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return {}


def compare_entities(expected: dict, predicted: dict) -> dict:
    """Compare expected vs predicted entities."""
    fields = ['amount', 'type', 'date', 'account', 'reference', 'merchant']
    
    results = {}
    for field in fields:
        exp_val = str(expected.get(field, '')).lower().strip()
        pred_val = str(predicted.get(field, '')).lower().strip()
        
        if exp_val:
            # Normalize amount (remove commas, trailing zeros)
            if field == 'amount':
                exp_val = exp_val.replace(',', '').rstrip('0').rstrip('.')
                pred_val = pred_val.replace(',', '').rstrip('0').rstrip('.')
            
            results[field] = {
                'expected': exp_val,
                'predicted': pred_val,
                'correct': exp_val == pred_val
            }
    
    return results


def run_evaluation(limit: int = None):
    """Run evaluation on test dataset."""
    print("=" * 70)
    print("📊 QUANTITATIVE EVALUATION - finance-lora-v6")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Adapter: {ADAPTER_PATH}")
    print()
    
    # Load test data
    with open(TEST_FILE) as f:
        test_data = json.load(f)
    
    if limit:
        test_data = test_data[:limit]
    
    print(f"Testing {len(test_data)} samples...")
    print()
    
    # Track results by bank and field
    bank_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    field_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for i, sample in enumerate(test_data):
        bank = sample.get('bank', 'unknown')
        body = sample.get('body', sample.get('raw_text', ''))
        expected = sample.get('entities', {})
        
        # Create prompt
        prompt = f"""Extract financial entities from this email:

{body}

Extract: amount, type, date, account, reference, merchant
Output JSON:"""
        
        # Generate prediction
        output = generate(prompt)
        predicted = parse_json_from_output(output)
        
        # Compare
        comparison = compare_entities(expected, predicted)
        
        # Update stats
        sample_correct = 0
        sample_total = 0
        for field, result in comparison.items():
            field_stats[field]['total'] += 1
            if result['correct']:
                field_stats[field]['correct'] += 1
                sample_correct += 1
            sample_total += 1
        
        if sample_total > 0:
            bank_stats[bank]['total'] += 1
            if sample_correct == sample_total:
                bank_stats[bank]['correct'] += 1
        
        # Progress
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(test_data)}...")
    
    # Print results
    print()
    print("=" * 70)
    print("📈 RESULTS BY BANK")
    print("=" * 70)
    
    total_correct = 0
    total_samples = 0
    
    for bank in sorted(bank_stats.keys()):
        stats = bank_stats[bank]
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        status = "✅" if acc >= 90 else "⚠️" if acc >= 70 else "❌"
        print(f"  {bank.upper():12} {stats['correct']:3}/{stats['total']:3} = {acc:5.1f}% {status}")
        total_correct += stats['correct']
        total_samples += stats['total']
    
    overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"\n  {'OVERALL':12} {total_correct:3}/{total_samples:3} = {overall_acc:5.1f}%")
    
    print()
    print("=" * 70)
    print("📈 RESULTS BY FIELD")
    print("=" * 70)
    
    for field in ['amount', 'type', 'date', 'account', 'reference', 'merchant']:
        if field in field_stats:
            stats = field_stats[field]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            status = "✅" if acc >= 90 else "⚠️" if acc >= 70 else "❌"
            print(f"  {field:12} {stats['correct']:3}/{stats['total']:3} = {acc:5.1f}% {status}")
    
    print()
    print("=" * 70)
    print("✅ Evaluation Complete!")
    print("=" * 70)
    
    # Save results
    results = {
        'model': MODEL_PATH,
        'adapter': ADAPTER_PATH,
        'total_samples': total_samples,
        'overall_accuracy': overall_acc,
        'by_bank': {k: v for k, v in bank_stats.items()},
        'by_field': {k: v for k, v in field_stats.items()}
    }
    
    with open('evaluation_results_v6.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to evaluation_results_v6.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=20, help='Number of samples to test')
    args = parser.parse_args()
    
    run_evaluation(limit=args.limit)
