"""
Verify Model Compatibility with Transformers.

Checks if the fused model can be loaded by standard Hugging Face Transformers.

Author: Ranjit Behera
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

MODEL_PATH = "models/released/finance-extractor-v8-pytorch"

def verify_compatibility():
    print(f"🔄 Verifying compatibility for: {MODEL_PATH}")
    
    try:
        # Try loading tokenizer
        print("1. Loading Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        print("   ✅ Tokenizer loaded successfully")
        
        # Try loading model
        print("2. Loading Model (PyTorch)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("   ✅ Model loaded successfully")
        
        # Test generation
        print("3. Testing Inference...")
        prompt = "Extract financial entities from this email:\n\nRs.500 debited from HDFC A/c 1234.\n\nExtract: amount, bank\nOutput JSON:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
            
        print("   ✅ Generation successful")
        print("\n🎉 The model is fully compatible with Hugging Face Transformers!")
        return True
        
    except Exception as e:
        print(f"\n❌ Compatibility verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if verify_compatibility():
        sys.exit(0)
    else:
        sys.exit(1)
