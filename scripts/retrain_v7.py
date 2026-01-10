"""
Combine Training Data and Retrain v7.

Adds ICICI samples to existing training data and retrains the model.

Author: Ranjit Behera
"""

import json
import subprocess
import random
from pathlib import Path

def combine_and_retrain():
    print("=" * 60)
    print("🔄 COMBINING DATA AND RETRAINING v7")
    print("=" * 60)
    
    # Load existing training data
    train_file = Path("data/training/train.jsonl")
    valid_file = Path("data/training/valid.jsonl")
    icici_file = Path("data/synthetic/icici_samples.jsonl")
    
    print("\n1. Loading existing training data...")
    existing_train = []
    with open(train_file, 'r') as f:
        for line in f:
            existing_train.append(json.loads(line))
    print(f"   Loaded {len(existing_train)} existing samples")
    
    print("\n2. Loading ICICI samples...")
    icici_samples = []
    with open(icici_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Keep prompt/completion format
            icici_samples.append({
                'prompt': data['prompt'],
                'completion': data['completion']
            })
    print(f"   Loaded {len(icici_samples)} ICICI samples")
    
    # Combine
    print("\n3. Combining datasets...")
    combined = existing_train + icici_samples
    random.seed(42)
    random.shuffle(combined)
    
    # Split 90/10
    split_idx = int(len(combined) * 0.9)
    new_train = combined[:split_idx]
    new_valid = combined[split_idx:]
    
    print(f"   New training set: {len(new_train)}")
    print(f"   New validation set: {len(new_valid)}")
    
    # Backup and save
    print("\n4. Saving combined data...")
    
    # Backup original
    train_file.rename(train_file.with_suffix('.jsonl.bak'))
    valid_file.rename(valid_file.with_suffix('.jsonl.bak'))
    
    with open(train_file, 'w') as f:
        for sample in new_train:
            f.write(json.dumps(sample) + '\n')
    
    with open(valid_file, 'w') as f:
        for sample in new_valid:
            f.write(json.dumps(sample) + '\n')
    
    print(f"   Saved to {train_file}")
    print(f"   Saved to {valid_file}")
    
    # Count ICICI in training
    icici_count = sum(1 for s in new_train if 'ICICI' in s.get('text', '').upper())
    print(f"\n   ICICI samples in training: ~{icici_count}")
    
    print("\n✅ Data combined successfully!")
    print("\n5. Starting v7 training...")
    
    # Train v7
    cmd = [
        "mlx_lm.lora",
        "--model", "models/base/phi3-finance-base",
        "--data", "data/training",
        "--train",
        "--iters", "500",  # Quick iteration
        "--batch-size", "1",
        "--num-layers", "16",
        "--learning-rate", "1e-5",
        "--adapter-path", "models/adapters/finance-lora-v7",
        "--max-seq-length", "1024"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ v7 training complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")

if __name__ == "__main__":
    combine_and_retrain()
