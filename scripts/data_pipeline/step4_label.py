#!/usr/bin/env python3
"""
Step 4: Create Labeled Training Data
=====================================

Processes the clean SMS data and creates training labels.
Extracts:
- amount, type, account, date, reference (regex)
- beneficiary_name (from SMS pattern)
- Detects merchant vs P2P transactions

Usage:
    python step4_label.py --input step2_sms_clean.csv --output step4_labeled.csv
"""

import argparse
import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


# ============================================================================
# ICICI BANK SMS PATTERNS (Dominant format in data)
# ============================================================================

ICICI_DEBIT_PATTERN = re.compile(
    r'ICICI Bank Acc?t?\s*XX?(\d+)\s+debited\s+(?:for\s+)?Rs\.?\s*([\d,]+(?:\.\d{2})?)\s+on\s+(\d{1,2}-[A-Za-z]{3}-\d{2,4})[\s;]+([A-Za-z0-9\s]+?)\s+credited\.\s*UPI[:\s]*(\d+)',
    re.IGNORECASE
)

ICICI_CREDIT_PATTERN = re.compile(
    r'(?:Dear Customer,?\s*)?Acc?t?\s*XX?(\d+)\s+(?:is\s+)?credited\s+(?:with\s+)?Rs\.?\s*([\d,]+(?:\.\d{2})?)\s+on\s+(\d{1,2}-[A-Za-z]{3}-\d{2,4})\s+from\s+([A-Za-z0-9\s]+?)[\.\s]+UPI[:\s]*(\d+)',
    re.IGNORECASE
)

# Generic amount pattern
AMOUNT_PATTERN = re.compile(r'Rs\.?\s*([\d,]+(?:\.\d{1,2})?)', re.IGNORECASE)
DATE_PATTERN = re.compile(r'(\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})')
UPI_REF_PATTERN = re.compile(r'UPI[:\s]*(\d{12,16})', re.IGNORECASE)
ACCOUNT_PATTERN = re.compile(r'XX?(\d{3,4})', re.IGNORECASE)

# Merchant detection keywords (P2M vs P2P)
MERCHANT_KEYWORDS = {
    'swiggy', 'zomato', 'uber', 'ola', 'amazon', 'flipkart', 'paytm', 
    'phonepe', 'google', 'youtube', 'netflix', 'spotify', 'airtel',
    'jio', 'vodafone', 'bsnl', 'electricity', 'gas', 'water', 'bill',
    'store', 'mart', 'shop', 'restaurant', 'hotel', 'hospital', 'clinic',
    'pharmacy', 'petrol', 'fuel', 'charging', 'parking', 'toll', 'metro',
    'railway', 'flight', 'bus', 'cab', 'taxi', 'rent', 'insurance',
    'zepto', 'bigbasket', 'blinkit', 'instamart', 'dunzo', 'myntra',
    'ajio', 'nykaa', 'tata', 'reliance', 'dmart', 'more', 'grofers'
}


def is_merchant(beneficiary: str) -> bool:
    """Determine if beneficiary is a merchant (P2M) or person (P2P)."""
    if not beneficiary:
        return False
    
    name_lower = beneficiary.lower().strip()
    
    # Check against known merchant keywords
    for keyword in MERCHANT_KEYWORDS:
        if keyword in name_lower:
            return True
    
    # Heuristics for P2M vs P2P:
    # - Merchants often have Ltd, Pvt, Inc, Store, Shop
    # - Person names are usually 2-3 words
    
    if any(x in name_lower for x in ['ltd', 'pvt', 'inc', 'llp', 'corp', 
                                      'store', 'shop', 'mart', 'services',
                                      'limited', 'private']):
        return True
    
    # All caps names with numbers are likely merchants
    if beneficiary.isupper() and any(c.isdigit() for c in beneficiary):
        return True
    
    return False


def normalize_beneficiary(name: str) -> str:
    """Clean up beneficiary name."""
    if not name:
        return ""
    
    # Remove trailing/leading whitespace
    name = name.strip()
    
    # Remove common suffixes
    name = re.sub(r'\s+credited\.?$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+debited\.?$', '', name, flags=re.IGNORECASE)
    
    # Title case if all uppercase
    if name.isupper():
        name = name.title()
    
    return name.strip()


def extract_from_sms(body: str) -> Dict[str, Any]:
    """Extract all fields from SMS body."""
    result = {
        'amount': None,
        'type': None,
        'account': None,
        'date': None,
        'reference': None,
        'beneficiary': None,
        'is_merchant': False,
        'category': None,
        'extraction_method': None
    }
    
    # Try ICICI debit pattern
    match = ICICI_DEBIT_PATTERN.search(body)
    if match:
        result['account'] = match.group(1)
        result['amount'] = float(match.group(2).replace(',', ''))
        result['date'] = match.group(3)
        result['beneficiary'] = normalize_beneficiary(match.group(4))
        result['reference'] = match.group(5)
        result['type'] = 'debit'
        result['is_merchant'] = is_merchant(result['beneficiary'])
        result['extraction_method'] = 'icici_debit_pattern'
        return result
    
    # Try ICICI credit pattern
    match = ICICI_CREDIT_PATTERN.search(body)
    if match:
        result['account'] = match.group(1)
        result['amount'] = float(match.group(2).replace(',', ''))
        result['date'] = match.group(3)
        result['beneficiary'] = normalize_beneficiary(match.group(4))
        result['reference'] = match.group(5)
        result['type'] = 'credit'
        result['is_merchant'] = is_merchant(result['beneficiary'])
        result['extraction_method'] = 'icici_credit_pattern'
        return result
    
    # Fallback: generic extraction
    # Amount
    amount_match = AMOUNT_PATTERN.search(body)
    if amount_match:
        try:
            result['amount'] = float(amount_match.group(1).replace(',', ''))
        except:
            pass
    
    # Type
    if re.search(r'\bdebit', body, re.IGNORECASE):
        result['type'] = 'debit'
    elif re.search(r'\bcredit', body, re.IGNORECASE):
        result['type'] = 'credit'
    
    # Account
    acc_match = ACCOUNT_PATTERN.search(body)
    if acc_match:
        result['account'] = acc_match.group(1)
    
    # Date
    date_match = DATE_PATTERN.search(body)
    if date_match:
        result['date'] = date_match.group(1)
    
    # Reference
    ref_match = UPI_REF_PATTERN.search(body)
    if ref_match:
        result['reference'] = ref_match.group(1)
    
    result['extraction_method'] = 'generic_fallback'
    return result


def create_training_label(row: Dict[str, Any], extraction: Dict[str, Any]) -> Dict[str, Any]:
    """Create a training label with ground truth."""
    body = str(row.get('body', ''))
    
    # Build ground truth JSON (what we want the model to output)
    ground_truth = {
        'amount': extraction['amount'],
        'type': extraction['type'],
        'account': extraction['account'],
        'date': extraction['date'],
        'reference': extraction['reference'],
        'beneficiary': extraction['beneficiary'],
        'is_p2m': extraction['is_merchant'],
    }
    
    # Remove None values
    ground_truth = {k: v for k, v in ground_truth.items() if v is not None}
    
    return {
        # Original data
        'timestamp': row.get('timestamp', ''),
        'sender': row.get('sender', ''),
        'body': body,
        'source': row.get('source', ''),
        
        # Extracted fields
        **{f'extracted_{k}': v for k, v in extraction.items()},
        
        # Training label (JSON format for LLM fine-tuning)
        'ground_truth_json': json.dumps(ground_truth, ensure_ascii=False),
        
        # Quality flags
        'has_amount': extraction['amount'] is not None,
        'has_type': extraction['type'] is not None,
        'has_beneficiary': extraction['beneficiary'] is not None and len(extraction['beneficiary']) > 0,
        'complete_extraction': all([
            extraction['amount'] is not None,
            extraction['type'] is not None,
            extraction['reference'] is not None
        ]),
    }


def label_data(df: pd.DataFrame) -> pd.DataFrame:
    """Label all data for training."""
    print("=" * 60)
    print("🏷️ STEP 4: CREATING LABELED TRAINING DATA")
    print("=" * 60)
    
    results = []
    complete_count = 0
    
    for i, (_, row) in enumerate(df.iterrows()):
        body = str(row.get('body', ''))
        extraction = extract_from_sms(body)
        label = create_training_label(row.to_dict(), extraction)
        results.append(label)
        
        if label['complete_extraction']:
            complete_count += 1
        
        if (i + 1) % 500 == 0:
            print(f"   Processed {i+1:,}/{len(df):,} ({100*complete_count/(i+1):.1f}% complete)")
    
    result_df = pd.DataFrame(results)
    
    print(f"\n📊 LABELING RESULTS:")
    print(f"   Total records: {len(result_df):,}")
    print(f"   Complete extractions: {complete_count:,} ({100*complete_count/len(result_df):.1f}%)")
    print(f"   Has amount: {result_df['has_amount'].sum():,}")
    print(f"   Has type: {result_df['has_type'].sum():,}")
    print(f"   Has beneficiary: {result_df['has_beneficiary'].sum():,}")
    
    # Show breakdown by extraction method
    print(f"\n📋 EXTRACTION METHODS:")
    method_counts = result_df['extracted_extraction_method'].value_counts()
    for method, count in method_counts.items():
        print(f"   {method}: {count:,}")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Step 4: Create labeled training data")
    parser.add_argument("--input", "-i", default="data/pipeline/step2_sms_clean.csv",
                       help="Input CSV from step 2 (SMS only)")
    parser.add_argument("--output", "-o", default="data/pipeline/step4_labeled.csv",
                       help="Output CSV with labels")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    # Load data
    print(f"\n📂 Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df):,} records")
    
    # Label data
    labeled_df = label_data(df)
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved labeled data to: {output_path}")
    
    # Also save training-ready JSONL (for LLM fine-tuning)
    jsonl_path = output_path.parent / "step4_training.jsonl"
    with open(jsonl_path, 'w') as f:
        for _, row in labeled_df[labeled_df['complete_extraction']].iterrows():
            training_example = {
                'input': row['body'],
                'output': row['ground_truth_json']
            }
            f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
    
    print(f"   JSONL for LLM training: {jsonl_path}")


if __name__ == "__main__":
    main()
