"""
Fine-tune Model v6.
Final task-specific fine-tuning for UPI emails, bank statements,
and payment app statements. Uses the domain-pretrained base.

Author: Ranjit Behera
"""

import subprocess
import argparse
from pathlib import Path

def run_finetune():
    base_model = "models/base/phi3-finance-base"
    data_dir = "data/training"
    adapter_path = "models/adapters/finance-lora-v6"
    
    print(f"🚀 Starting Fine-tuning v6 using {base_model}...")
    
    cmd = [
        "mlx_lm.lora",
        "--model", base_model,
        "--data", data_dir,
        "--train",
        "--iters", "1500",
        "--batch-size", "1",
        "--num-layers", "16",
        "--learning-rate", "1e-5",
        "--adapter-path", adapter_path,
        "--max-seq-length", "1024" # Increased seq length for statements
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Successfully trained v6 adapters at {adapter_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Fine-tuning failed: {e}")

if __name__ == "__main__":
    run_finetune()
