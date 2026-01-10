"""
Claude API Auto-Labeling Script.

Uses Claude API to automatically label financial emails for training.
Generates high-quality labels at scale.

Author: Ranjit Behera

Usage:
    export ANTHROPIC_API_KEY="your-key-here"
    python scripts/claude_labeling.py --limit 100
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("⚠️ anthropic package not installed. Run: pip install anthropic")

CORPUS_FILE = Path("data/corpus/emails/financial_emails.jsonl")
OUTPUT_FILE = Path("data/synthetic/claude_labeled.jsonl")

EXTRACTION_PROMPT = """You are a financial entity extraction expert. Extract structured data from this Indian bank email.

EMAIL:
{email_text}

Extract the following fields (return empty string if not found):
- amount: The transaction amount (numbers only, no currency symbols)
- type: "credit" or "debit"
- date: Transaction date (keep original format)
- account: Last 4 digits of account number
- reference: UPI/NEFT/IMPS reference number (12+ digits)
- merchant: Merchant/recipient name (lowercase)
- bank: Bank name (hdfc/icici/sbi/axis/kotak/phonepe/gpay/paytm)

Respond ONLY with valid JSON, no explanation:"""


def extract_with_claude(email_text: str, client) -> Optional[dict]:
    """Use Claude to extract entities from email."""
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(email_text=email_text[:1000])
                }
            ]
        )
        
        response_text = message.content[0].text
        
        # Parse JSON
        import re
        match = re.search(r'\{[^{}]+\}', response_text, re.DOTALL)
        if match:
            return json.loads(match.group())
        
    except Exception as e:
        print(f"    Error: {e}")
    
    return None


def load_unlabeled_emails(limit: int = 100) -> list:
    """Load emails that need labeling."""
    emails = []
    
    with open(CORPUS_FILE, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                body = data.get('body', '')
                
                # Filter for transaction emails
                body_lower = body.lower()
                has_transaction = any(x in body_lower for x in ['debited', 'credited', 'received', 'paid'])
                has_amount = any(x in body_lower for x in ['rs.', 'rs ', 'inr', '₹'])
                
                if has_transaction and has_amount and len(body) > 50:
                    emails.append({
                        'text': body,
                        'subject': data.get('subject', ''),
                        'sender': data.get('sender', '')
                    })
                    
                    if len(emails) >= limit:
                        break
            except:
                continue
    
    return emails


def run_labeling(limit: int = 100):
    """Run Claude labeling on unlabeled emails."""
    print("=" * 60)
    print("🤖 CLAUDE API AUTO-LABELING")
    print("=" * 60)
    
    if not HAS_ANTHROPIC:
        print("❌ Please install anthropic: pip install anthropic")
        return
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    client = anthropic.Anthropic(api_key=api_key)
    
    print(f"\n1. Loading unlabeled emails (limit: {limit})...")
    emails = load_unlabeled_emails(limit=limit)
    print(f"   Found {len(emails)} candidates")
    
    print(f"\n2. Labeling with Claude...")
    
    labeled = []
    for i, email in enumerate(emails):
        print(f"   [{i+1}/{len(emails)}] Extracting...", end=" ")
        
        entities = extract_with_claude(email['text'], client)
        
        if entities and entities.get('amount'):
            # Create training sample
            prompt = f"""Extract financial entities from this email:

{email['text'][:500]}

Extract: amount, type, date, account, reference, merchant
Output JSON:"""
            
            labeled.append({
                'prompt': prompt,
                'completion': json.dumps(entities, indent=2),
                'source': 'claude_labeled'
            })
            print(f"✅ {entities.get('amount')}")
        else:
            print("❌ No entities found")
        
        # Rate limit
        time.sleep(0.5)
    
    # Save
    print(f"\n3. Saving labeled data...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        for sample in labeled:
            f.write(json.dumps(sample) + '\n')
    
    print(f"   ✅ Saved {len(labeled)} labeled samples to {OUTPUT_FILE}")
    
    # Show sample
    if labeled:
        print("\n📧 Sample:")
        print(labeled[0]['completion'])
    
    return labeled


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=50, help='Number of emails to label')
    args = parser.parse_args()
    
    run_labeling(limit=args.limit)
