"""
Quick Evaluation for finance-lora-v6.
Tests the model with sample prompts from each phase.

Author: Ranjit Behera
"""

import subprocess
import sys

def generate(prompt: str) -> str:
    """Generate response using mlx_lm.generate."""
    cmd = [
        sys.executable, "-m", "mlx_lm.generate",
        "--model", "models/base/phi3-finance-base",
        "--adapter-path", "models/adapters/finance-lora-v6",
        "--prompt", prompt,
        "--max-tokens", "200"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.stdout
    except Exception as e:
        return f"Error: {e}"


def main():
    print("=" * 70)
    print("🧪 FINANCE-LORA-V6 EVALUATION")
    print("=" * 70)
    
    # Test 1: UPI Email (Phase 1)
    print("\n📧 TEST 1: UPI Email Extraction (Phase 1)")
    print("-" * 50)
    upi_prompt = """Extract financial entities from this email:

Subject: Money received! Rs.500.00 credited
From: alerts@hdfcbank.net

Dear Customer,
Rs.500.00 has been credited to your account ending 1234 by UPI.
UPI Ref: 123456789012
From: customer@okaxis
Date: 15-Jan-2024

Extract: amount, sender, receiver, transaction_id, date, bank
Output JSON:"""
    
    print(f"Prompt: {upi_prompt[:100]}...")
    response = generate(upi_prompt)
    print(f"Response:\n{response}")
    
    # Test 2: Bank Statement Row (Phase 2)
    print("\n🏦 TEST 2: Bank Statement Row (Phase 2)")
    print("-" * 50)
    statement_prompt = """[BANK_STATEMENT]
15/01/2024 | UPI/CR/415926537890 | AMOUNT CREDITED FROM ROHIT KUMAR | 2500.00 | 15000.00

Extract: date, description, amount, balance, type
Output JSON:"""
    
    print(f"Prompt: {statement_prompt[:100]}...")
    response = generate(statement_prompt)
    print(f"Response:\n{response}")
    
    # Test 3: PhonePe Statement (Phase 3)
    print("\n📱 TEST 3: PhonePe Statement (Phase 3)")
    print("-" * 50)
    phonepe_prompt = """[PHONEPE]
Transaction Details:
Date: Jan 15, 2024 at 3:45 PM
Paid to: Swiggy
Amount: ₹350.00
Transaction ID: PPT202401153456789012
Status: Success

Extract: date, receiver, amount, transaction_id, status
Output JSON:"""
    
    print(f"Prompt: {phonepe_prompt[:100]}...")
    response = generate(phonepe_prompt)
    print(f"Response:\n{response}")
    
    # Test 4: GPay Statement (Phase 3)
    print("\n📱 TEST 4: GPay Statement (Phase 3)")
    print("-" * 50)
    gpay_prompt = """[GPAY]
You paid ₹1,200.00 to Zomato
Jan 16, 2024
Transaction ID: GPay-TXN-8765432109876543
Status: Completed

Extract: date, receiver, amount, transaction_id, status
Output JSON:"""
    
    print(f"Prompt: {gpay_prompt[:100]}...")
    response = generate(gpay_prompt)
    print(f"Response:\n{response}")
    
    print("\n" + "=" * 70)
    print("✅ Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
