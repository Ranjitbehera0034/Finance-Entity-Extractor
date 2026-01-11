#!/usr/bin/env python3
"""
Step 3: Baseline Test
=====================

Tests the current finee extractor on the cleaned training data
to establish a baseline before fine-tuning.

This answers: "How good is our Regex engine on real data?"

Usage:
    python step3_baseline.py --input step2_training_ready.csv
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from finee import extract
    from finee.schema import ExtractionResult
except ImportError:
    print("❌ finee not installed!")
    print("   Run: pip install finee")
    sys.exit(1)


def extract_and_analyze(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract entities from a message and analyze results."""
    body = str(row.get('body', ''))
    
    try:
        result = extract(body)
        
        return {
            # Original data
            'timestamp': row.get('timestamp', ''),
            'sender': row.get('sender', ''),
            'body': body[:200] + '...' if len(body) > 200 else body,
            'source': row.get('source', ''),
            
            # Extracted fields
            'extracted_amount': result.amount,
            'extracted_type': result.type.value if result.type else None,
            'extracted_account': result.account,
            'extracted_date': result.date,
            'extracted_reference': result.reference,
            'extracted_vpa': result.vpa,
            'extracted_merchant': result.merchant,
            'extracted_category': result.category.value if result.category else None,
            'extracted_confidence': result.confidence.value if result.confidence else None,
            'extracted_confidence_score': result.confidence_score,
            
            # Quality metrics
            'has_amount': result.amount is not None,
            'has_type': result.type is not None,
            'has_merchant': result.merchant is not None,
            'has_category': result.category is not None,
            'fields_extracted': sum([
                result.amount is not None,
                result.type is not None,
                result.account is not None,
                result.date is not None,
                result.reference is not None,
                result.merchant is not None,
                result.category is not None,
            ]),
            
            # Processing info
            'processing_time_ms': result.processing_time_ms,
            'extraction_success': result.amount is not None and result.type is not None,
        }
    except Exception as e:
        return {
            'timestamp': row.get('timestamp', ''),
            'sender': row.get('sender', ''),
            'body': body[:200],
            'source': row.get('source', ''),
            'extraction_error': str(e),
            'extraction_success': False,
        }


def run_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Run baseline extraction on all rows."""
    print("=" * 60)
    print("📊 STEP 3: BASELINE TEST")
    print("=" * 60)
    print(f"\nTesting finee extractor on {len(df):,} messages...")
    print("(This tests Regex-only mode, no LLM)\n")
    
    results = []
    success_count = 0
    
    for i, (_, row) in enumerate(df.iterrows()):
        result = extract_and_analyze(row.to_dict())
        results.append(result)
        
        if result.get('extraction_success'):
            success_count += 1
        
        # Progress every 100
        if (i + 1) % 100 == 0:
            pct = 100 * success_count / (i + 1)
            print(f"   Processed {i+1:,}/{len(df):,} ({pct:.1f}% success rate)")
    
    return pd.DataFrame(results)


def analyze_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze extraction results."""
    total = len(results_df)
    
    # Core metrics
    success_count = results_df['extraction_success'].sum()
    has_amount = results_df['has_amount'].sum()
    has_type = results_df['has_type'].sum()
    has_merchant = results_df['has_merchant'].sum()
    has_category = results_df['has_category'].sum()
    
    # Confidence distribution
    confidence_counts = results_df['extracted_confidence'].value_counts().to_dict()
    
    # Type distribution
    type_counts = results_df['extracted_type'].value_counts().to_dict()
    
    # Category distribution
    category_counts = results_df['extracted_category'].value_counts().to_dict()
    
    # Top merchants
    merchant_counts = results_df['extracted_merchant'].value_counts().head(20).to_dict()
    
    # Performance
    avg_time = results_df['processing_time_ms'].mean()
    
    analysis = {
        'total_messages': total,
        'extraction_success_rate': 100 * success_count / total,
        'field_coverage': {
            'amount': 100 * has_amount / total,
            'type': 100 * has_type / total,
            'merchant': 100 * has_merchant / total,
            'category': 100 * has_category / total,
        },
        'confidence_distribution': confidence_counts,
        'type_distribution': type_counts,
        'category_distribution': category_counts,
        'top_merchants': merchant_counts,
        'avg_processing_time_ms': avg_time,
        'timestamp': datetime.now().isoformat(),
    }
    
    return analysis


def print_analysis(analysis: Dict[str, Any]) -> None:
    """Print analysis results."""
    print("\n" + "=" * 60)
    print("📈 BASELINE RESULTS")
    print("=" * 60)
    
    print(f"\n📊 COVERAGE:")
    print(f"   Total messages:      {analysis['total_messages']:,}")
    print(f"   Extraction success:  {analysis['extraction_success_rate']:.1f}%")
    
    print(f"\n📋 FIELD COVERAGE:")
    for field, pct in analysis['field_coverage'].items():
        status = "✅" if pct >= 80 else "⚠️" if pct >= 50 else "❌"
        print(f"   {field:12} {pct:5.1f}% {status}")
    
    print(f"\n📊 CONFIDENCE DISTRIBUTION:")
    for level, count in sorted(analysis['confidence_distribution'].items(), key=lambda x: -x[1]):
        if level:
            pct = 100 * count / analysis['total_messages']
            print(f"   {level:10} {count:,} ({pct:.1f}%)")
    
    print(f"\n💳 TRANSACTION TYPES:")
    for txn_type, count in sorted(analysis['type_distribution'].items(), key=lambda x: -x[1]):
        if txn_type:
            pct = 100 * count / analysis['total_messages']
            print(f"   {txn_type:10} {count:,} ({pct:.1f}%)")
    
    print(f"\n🏪 TOP 10 MERCHANTS:")
    for i, (merchant, count) in enumerate(list(analysis['top_merchants'].items())[:10]):
        if merchant:
            print(f"   {i+1:2}. {merchant:20} {count:,}")
    
    print(f"\n⚡ PERFORMANCE:")
    print(f"   Avg processing time: {analysis['avg_processing_time_ms']:.2f}ms")
    print(f"   Throughput: ~{1000/analysis['avg_processing_time_ms']:.0f} msg/sec")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Step 3: Baseline extraction test")
    parser.add_argument("--input", "-i", default="data/pipeline/step2_training_ready.csv",
                       help="Input CSV from step 2")
    parser.add_argument("--output", "-o", default="data/pipeline/step3_baseline_results.csv",
                       help="Output CSV with extraction results")
    parser.add_argument("--limit", "-n", type=int, default=None,
                       help="Limit number of rows to process (for testing)")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        print(f"   Run step2_filter.py first!")
        return
    
    # Load data
    print(f"\n📂 Loading: {input_path}")
    df = pd.read_csv(input_path)
    
    if args.limit:
        df = df.head(args.limit)
        print(f"   (Limited to {args.limit} rows for testing)")
    
    print(f"   Loaded {len(df):,} records")
    
    # Run baseline
    results_df = run_baseline(df)
    
    # Analyze
    analysis = analyze_results(results_df)
    print_analysis(analysis)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved extraction results to: {output_path}")
    
    # Save analysis as JSON
    analysis_path = output_path.parent / "step3_baseline_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"   Analysis saved to: {analysis_path}")
    
    # Summary
    success_rate = analysis['extraction_success_rate']
    if success_rate >= 80:
        print(f"\n🎉 Great! {success_rate:.1f}% success rate. Regex is working well!")
    elif success_rate >= 50:
        print(f"\n⚠️ {success_rate:.1f}% success rate. Room for improvement.")
        print("   Consider adding more regex patterns or enabling LLM mode.")
    else:
        print(f"\n❌ Low {success_rate:.1f}% success rate.")
        print("   Your data may have unusual formats. Review failed extractions.")


if __name__ == "__main__":
    main()
