#!/usr/bin/env python3
"""
Step 2: Garbage Filter
======================

Filters out noise from unified data:
- OTPs / Verification codes
- Login alerts
- Marketing spam
- Bill reminders (non-transaction)
- Password reset

Keeps only actual transaction messages.

Usage:
    python step2_filter.py --input step1_unified.csv --output step2_training_ready.csv
"""

import argparse
import re
import pandas as pd
from pathlib import Path
from typing import Tuple


# ============================================================================
# GARBAGE PATTERNS (Messages to REMOVE)
# ============================================================================

GARBAGE_PATTERNS = [
    # OTPs and verification codes
    r'\bOTP\b',
    r'\bone.time.password\b',
    r'\bverification.code\b',
    r'\bverify.your\b',
    r'\bconfirm.your.identity\b',
    r'\bcvv\b',
    r'\bpin\b.*\bdo.not.share\b',
    r'\b\d{4,6}\b.*\bexpires?\b',
    r'\bvalid.for.\d+.min',
    
    # Login/Security alerts
    r'\blogin.alert\b',
    r'\blogged.in\b',
    r'\bnew.device\b',
    r'\bnew.login\b',
    r'\bsecurity.alert\b',
    r'\bsign.in.attempt\b',
    r'\bpassword.changed\b',
    r'\bpassword.reset\b',
    r'\baccount.locked\b',
    r'\bunusual.activity\b',
    
    # Marketing / Promotional
    r'\b\d+%.off\b',
    r'\bdiscount\b',
    r'\bcashback.offer\b',
    r'\bflat.rs\b.*\boff\b',
    r'\bget.upto\b',
    r'\bwin.upto\b',
    r'\bexclusive.offer\b',
    r'\blimited.time\b',
    r'\bspecial.offer\b',
    r'\bsale.live\b',
    r'\bshop.now\b',
    r'\bbuy.now\b',
    r'\bsubscribe\b',
    r'\bunsubscribe\b',
    r'\bnewsletter\b',
    
    # Bill reminders (not actual transactions)
    r'\bbill.due\b',
    r'\bbill.reminder\b',
    r'\bdue.date\b',
    r'\bpay.before\b',
    r'\bauto.debit.scheduled\b',
    r'\bemi.due\b',
    r'\bpayment.reminder\b',
    r'\bminimum.amount.due\b',
    r'\boutstanding.balance\b',
    
    # Account statements / Summaries
    r'\baccount.statement\b',
    r'\bmonthly.statement\b',
    r'\be.statement\b',
    r'\bstatement.ready\b',
    r'\bdownload.statement\b',
    
    # Delivery / Shipping (Not finance)
    r'\bout.for.delivery\b',
    r'\bdelivered.to\b',
    r'\bshipment.update\b',
    r'\btracking.number\b',
    r'\border.confirmed\b',
    r'\border.placed\b',
    r'\border.shipped\b',
    
    # App notifications
    r'\brate.your.experience\b',
    r'\bleave.a.review\b',
    r'\bupdate.available\b',
    r'\bapp.update\b',
    r'\bdownload.app\b',
    r'\binstall.app\b',
    
    # Generic noise
    r'\bclick.here\b',
    r'\bvisit.us\b',
    r'\bcall.us\b',
    r'\bcontact.us\b',
    r'\bfollow.us\b',
    r'\bjoin.us\b',
    r'\blearn.more\b',
    r'\bread.more\b',
]

# ============================================================================
# KEEP PATTERNS (Messages to KEEP - actual transactions)
# ============================================================================

KEEP_PATTERNS = [
    # Transaction keywords
    r'\bdebited?\b',
    r'\bcredited?\b',
    r'\btransferred?\b',
    r'\bwithdra(?:wn|wal)\b',
    r'\bdeposited?\b',
    r'\breceived?\b',
    r'\bsent\b',
    r'\bpaid\b',
    r'\bpurchase\b',
    r'\btransaction\b',
    r'\btxn\b',
    r'\bupi\b',
    r'\bneft\b',
    r'\bimps\b',
    r'\brtgs\b',
    
    # Amount patterns
    r'rs\.?\s*[\d,]+',
    r'inr\s*[\d,]+',
    r'₹\s*[\d,]+',
    
    # Account references
    r'a/c\s*\w+',
    r'acct?\s*\w+',
    r'account\s*\w+',
    
    # Reference numbers
    r'ref\.?\s*:?\s*\d{10,}',
    r'upi.?ref\s*:?\s*\d{10,}',
]


def is_garbage(text: str) -> bool:
    """Check if message is garbage (should be removed)."""
    text_lower = text.lower()
    
    for pattern in GARBAGE_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    
    return False


def is_transaction(text: str) -> bool:
    """Check if message is a transaction (should be kept)."""
    text_lower = text.lower()
    
    matches = 0
    for pattern in KEEP_PATTERNS:
        if re.search(pattern, text_lower):
            matches += 1
    
    # Require at least 2 transaction indicators
    return matches >= 2


def classify_message(text: str) -> Tuple[str, str]:
    """
    Classify a message as 'keep', 'garbage', or 'uncertain'.
    
    Returns: (classification, reason)
    """
    if not text or len(text.strip()) < 20:
        return 'garbage', 'too_short'
    
    # First check garbage patterns (high confidence negative)
    if is_garbage(text):
        return 'garbage', 'matched_garbage_pattern'
    
    # Then check transaction patterns (high confidence positive)
    if is_transaction(text):
        return 'keep', 'matched_transaction_pattern'
    
    # Messages with amounts but no clear transaction type
    if re.search(r'(?:rs\.?|inr|₹)\s*[\d,]+', text.lower()):
        return 'uncertain', 'has_amount_no_transaction_type'
    
    # Default: uncertain
    return 'uncertain', 'no_strong_signals'


def filter_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filter DataFrame into keep/garbage/uncertain.
    
    Returns: (keep_df, garbage_df, uncertain_df)
    """
    print("=" * 60)
    print("🧹 STEP 2: GARBAGE FILTER")
    print("=" * 60)
    
    results = []
    for _, row in df.iterrows():
        body = row.get('body', '')
        classification, reason = classify_message(str(body))
        results.append({
            **row.to_dict(),
            'classification': classification,
            'filter_reason': reason
        })
    
    result_df = pd.DataFrame(results)
    
    keep_df = result_df[result_df['classification'] == 'keep'].copy()
    garbage_df = result_df[result_df['classification'] == 'garbage'].copy()
    uncertain_df = result_df[result_df['classification'] == 'uncertain'].copy()
    
    # Print summary
    total = len(df)
    print(f"\n📊 FILTER RESULTS:")
    print(f"   Total input:  {total:,}")
    print(f"   ✅ Keep:      {len(keep_df):,} ({100*len(keep_df)/total:.1f}%)")
    print(f"   ❌ Garbage:   {len(garbage_df):,} ({100*len(garbage_df)/total:.1f}%)")
    print(f"   ❓ Uncertain: {len(uncertain_df):,} ({100*len(uncertain_df)/total:.1f}%)")
    
    # Show reason breakdown for garbage
    if len(garbage_df) > 0:
        print(f"\n📋 Garbage Reasons:")
        for reason, count in garbage_df['filter_reason'].value_counts().items():
            print(f"   {reason}: {count:,}")
    
    return keep_df, garbage_df, uncertain_df


def main():
    parser = argparse.ArgumentParser(description="Step 2: Filter garbage messages")
    parser.add_argument("--input", "-i", default="data/pipeline/step1_unified.csv",
                       help="Input CSV from step 1")
    parser.add_argument("--output", "-o", default="data/pipeline/step2_training_ready.csv",
                       help="Output CSV with clean data")
    parser.add_argument("--save-all", action="store_true",
                       help="Also save garbage and uncertain to separate files")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        print(f"   Run step1_unify.py first!")
        return
    
    # Load data
    print(f"\n📂 Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df):,} records")
    
    # Filter
    keep_df, garbage_df, uncertain_df = filter_data(df)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Drop helper columns before saving
    keep_df = keep_df.drop(columns=['classification', 'filter_reason'])
    keep_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved clean data to: {output_path}")
    print(f"   Records: {len(keep_df):,}")
    
    if args.save_all:
        garbage_path = output_path.parent / "step2_garbage.csv"
        uncertain_path = output_path.parent / "step2_uncertain.csv"
        
        garbage_df.to_csv(garbage_path, index=False)
        uncertain_df.to_csv(uncertain_path, index=False)
        print(f"\n📁 Also saved:")
        print(f"   Garbage: {garbage_path}")
        print(f"   Uncertain: {uncertain_path}")
    
    print("\nNext: python scripts/data_pipeline/step3_baseline.py")


if __name__ == "__main__":
    main()
