"""
Test Model Generalization on Unseen Banks.

Tests the v8 model on banks that were NEVER in the training set
(Federal Bank, RBL Bank, IndusInd) to evaluate true domain understanding
vs. template memorization.

Author: Ranjit Behera
"""

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "models/released/finance-extractor-v8-pytorch"

def test_unseen():
    print(f"🔄 Loading v8 model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 1. Federal Bank (Unseen)
    federal_email = """
    From: alert@federalbank.co.in
    Subject: Transaction Alert
    
    Dear Customer,
    
    Your Federal Bank Acct XXXXXX1029 is debited for Rs.4,500.50 on 10-Jan-2026.
    Info: AMA*Netflix Subscription.
    Available Bal: Rs. 12,000.00.
    
    If not done by you, forward to phish@federalbank.co.in.
    """
    
    # 2. IndusInd Bank (Unseen - different structure)
    indusid_email = """
    IndusInd Bank Alert:
    INR 12,000.00 credited to your A/c no. 8822 via NEFT from REF-U99228811
    on 10/01/2026. Sender: RAJESH ENTERPRISES.
    Clr Bal: INR 54,000.00.
    """
    
    # 3. Generic/Unknown Format (Extreme generalization test)
    generic_sms = """
    Paid Rs 230 to Chai Point via UPI. Ref 992882211. 10 Jan 5:30 PM.
    """
    
    tests = [
        ("Federal Bank (Debited structure)", federal_email),
        ("IndusInd Bank (Credited structure)", indusid_email),
        ("Generic SMS (Informal)", generic_sms)
    ]
    
    print("\n🧪 TESTING GENERALIZATION ON UNSEEN FORMATS")
    print("=" * 60)
    
    prompt_template = """Extract financial entities from this email:

{text}

Extract: amount, type, date, account, reference, merchant, category
Output JSON:"""

    for name, text in tests:
        print(f"\n📋 Test: {name}")
        print("-" * 40)
        print(text.strip())
        print("-" * 40)
        
        inputs = tokenizer(prompt_template.format(text=text), return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        # Extract just the JSON part
        try:
            # Simple heuristic to find JSON start
            json_part = result.split("Output JSON:")[-1].strip()
            print("🤖 Model Output:")
            print(json_part)
        except:
            print(f"❌ Failed to parse: {result}")

if __name__ == "__main__":
    test_unseen()
