"""
Model Fusion Script.

Fuses the domain-pretrained adapters into the base model to create
a standalone finance-native base model. This model can then be
fine-tuned for specific tasks.

Author: Ranjit Behera
"""

import subprocess
import argparse
from pathlib import Path

def fuse_model(base_model: str, adapter_path: str, output_path: str):
    """Run mlx_lm.fuse to merge adapters into base model."""
    print(f"Merging {adapter_path} into {base_model}...")
    
    cmd = [
        "mlx_lm.fuse",
        "--model", base_model,
        "--adapter-path", adapter_path,
        "--save-path", output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Successfully fused model to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Fusion failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse MLX adapters into base model")
    parser.add_argument("--base", default="models/base/phi3-mini", help="Path to base model")
    parser.add_argument("--adapter", default="models/domain-pretrained/phi3-finance", help="Path to adapters")
    parser.add_argument("--output", default="models/base/phi3-finance-base", help="Output path for fused model")
    
    args = parser.parse_args()
    
    fuse_model(args.base, args.adapter, args.output)
