"""
RBI/NPCI Corpus Collection Script.

Downloads and processes financial documents from:
- RBI (Reserve Bank of India) circulars
- NPCI (National Payments Corporation of India) docs
- Banking glossaries and FAQs

Author: Ranjit Behera

Usage:
    python scripts/collect_rbi_npci.py
"""

import json
import time
from pathlib import Path
from typing import List, Dict
import re

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("⚠️ Missing dependencies. Run: pip install requests beautifulsoup4")

OUTPUT_DIR = Path("data/corpus/regulatory")

# RBI UPI-related URLs (public pages)
RBI_SOURCES = [
    {
        "name": "RBI UPI FAQ",
        "url": "https://www.rbi.org.in/Scripts/FAQView.aspx?Id=112",
        "type": "faq"
    },
    {
        "name": "RBI Digital Payments",
        "url": "https://www.rbi.org.in/Scripts/PublicationsView.aspx?id=21454",
        "type": "publication"
    }
]

# NPCI sources
NPCI_SOURCES = [
    {
        "name": "NPCI UPI Overview",
        "url": "https://www.npci.org.in/what-we-do/upi/product-overview",
        "type": "overview"
    },
    {
        "name": "NPCI IMPS",
        "url": "https://www.npci.org.in/what-we-do/imps/product-overview",
        "type": "overview"
    }
]

# Indian Banking terminology (expanded)
BANKING_TERMS = """
# UPI (Unified Payments Interface) Terms
UPI: Unified Payments Interface - A real-time payment system developed by NPCI
VPA: Virtual Payment Address - A unique identifier like yourname@bankname
UPI ID: Same as VPA, used to send and receive money
UPI PIN: 4-6 digit PIN used to authorize UPI transactions
Collect Request: Request for payment sent via UPI
Pay Request: Sending money via UPI
Transaction Reference Number: 12-digit unique identifier for UPI transactions
UTR: Unique Transaction Reference - Another term for transaction ID

# IMPS (Immediate Payment Service) Terms
IMPS: Immediate Payment Service - 24x7 instant money transfer
MMID: Mobile Money Identifier - 7-digit number linked to bank account
IFSC: Indian Financial System Code - 11-character bank branch identifier

# NEFT/RTGS Terms
NEFT: National Electronic Funds Transfer - Batch processing fund transfer
RTGS: Real Time Gross Settlement - High value instant transfers (min Rs.2 lakh)
Beneficiary: Person/entity receiving the funds
Remitter: Person/entity sending the funds

# Account Terms
Savings Account: Account for individuals with interest
Current Account: Account for businesses, no interest
Fixed Deposit: Term deposit with fixed interest rate
Recurring Deposit: Monthly deposit scheme
Demat Account: Account to hold securities in electronic form
CASA: Current Account Savings Account ratio

# Card Terms
Debit Card: Card linked to bank account for spending
Credit Card: Card with credit limit for spending
CVV: Card Verification Value - 3-digit security code
EMI: Equated Monthly Installment
POS: Point of Sale terminal

# Loan Terms
Home Loan: Loan for purchasing property
Personal Loan: Unsecured loan for personal use
EMI: Monthly payment amount
Principal: Original loan amount
Interest: Cost of borrowing
ROI: Rate of Interest
Tenure: Loan repayment period

# Investment Terms
Mutual Fund: Pooled investment vehicle
SIP: Systematic Investment Plan
NAV: Net Asset Value
ELSS: Equity Linked Savings Scheme
Dividend: Distribution of profits
Capital Gains: Profit from selling investments

# Tax Terms
TDS: Tax Deducted at Source
PAN: Permanent Account Number
GST: Goods and Services Tax
ITR: Income Tax Return
Form 16: TDS certificate from employer
Form 26AS: Annual tax statement

# Regulatory Bodies
RBI: Reserve Bank of India - Central bank
NPCI: National Payments Corporation of India
SEBI: Securities and Exchange Board of India
IRDAI: Insurance Regulatory and Development Authority
PFRDA: Pension Fund Regulatory and Development Authority

# Digital Payment Apps
PhonePe: UPI-based payment app
GPay: Google Pay - UPI payment app
Paytm: Payment and financial services app
BHIM: Bharat Interface for Money - Government UPI app
Amazon Pay: Amazon's payment service

# Common Transaction Types
Credit: Money added to account
Debit: Money removed from account
Transfer: Moving money between accounts
Bill Payment: Paying utility bills
Recharge: Adding balance to prepaid services
P2P: Person to Person transfer
P2M: Person to Merchant transfer

# Common Merchants
Swiggy: Food delivery platform
Zomato: Food delivery platform
Amazon: E-commerce platform
Flipkart: E-commerce platform
Uber: Ride-hailing service
Ola: Ride-hailing service
Rapido: Bike taxi service
BigBasket: Grocery delivery
Blinkit: Quick commerce
Zepto: Quick commerce
DMart: Retail chain
"""


def fetch_page(url: str) -> str:
    """Fetch and parse webpage content."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove scripts and styles
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text
        
    except Exception as e:
        print(f"    Error fetching {url}: {e}")
        return ""


def collect_corpus():
    """Collect corpus from RBI, NPCI and banking terms."""
    print("=" * 60)
    print("📚 COLLECTING RBI/NPCI CORPUS")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    corpus = []
    
    # 1. Banking Terms (always available)
    print("\n1. Adding banking terminology...")
    terms_lines = BANKING_TERMS.strip().split('\n')
    for line in terms_lines:
        if line.strip() and not line.startswith('#'):
            corpus.append({
                'text': line.strip(),
                'source': 'banking_terms',
                'type': 'glossary'
            })
    print(f"   Added {len(corpus)} term definitions")
    
    # 2. Try fetching RBI pages
    if HAS_DEPS:
        print("\n2. Fetching RBI documents...")
        for source in RBI_SOURCES:
            print(f"   Fetching {source['name']}...", end=" ")
            text = fetch_page(source['url'])
            if text and len(text) > 100:
                corpus.append({
                    'text': text[:10000],  # Limit size
                    'source': source['name'],
                    'url': source['url'],
                    'type': source['type']
                })
                print(f"✅ ({len(text)} chars)")
            else:
                print("❌")
            time.sleep(1)
        
        print("\n3. Fetching NPCI documents...")
        for source in NPCI_SOURCES:
            print(f"   Fetching {source['name']}...", end=" ")
            text = fetch_page(source['url'])
            if text and len(text) > 100:
                corpus.append({
                    'text': text[:10000],
                    'source': source['name'],
                    'url': source['url'],
                    'type': source['type']
                })
                print(f"✅ ({len(text)} chars)")
            else:
                print("❌")
            time.sleep(1)
    else:
        print("\n⚠️ Skipping web scraping (missing requests/beautifulsoup4)")
    
    # Save corpus
    output_file = OUTPUT_DIR / "rbi_npci_corpus.jsonl"
    with open(output_file, 'w') as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Saved {len(corpus)} documents to {output_file}")
    
    # Calculate tokens (rough estimate: 1 token ≈ 4 chars)
    total_chars = sum(len(item['text']) for item in corpus)
    est_tokens = total_chars // 4
    print(f"   Estimated tokens: ~{est_tokens:,}")
    
    return corpus


if __name__ == "__main__":
    collect_corpus()
