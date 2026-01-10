"""
Combine All Data and Retrain v8.

Combines:
- Existing training data (2800 samples)
- Multi-bank comprehensive data (498 samples)

Author: Ranjit Behera
"""

import json
import random
import subprocess
from pathlib import Path

def combine_and_retrain():
    print("=" * 60)
    print("🔄 COMBINING DATA AND RETRAINING v8")
    print("=" * 60)
    
    # Load existing
    train_file = Path("data/training/train.jsonl")
    valid_file = Path("data/training/valid.jsonl")
    multi_bank_file = Path("data/synthetic/multi_bank_comprehensive.jsonl")
    
    print("\n1. Loading existing training data...")
    existing = []
    with open(train_file, 'r') as f:
        for line in f:
            existing.append(json.loads(line))
    print(f"   Loaded {len(existing)} existing samples")
    
    print("\n2. Loading multi-bank comprehensive data...")
    multi_bank = []
    with open(multi_bank_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            multi_bank.append({
                'prompt': data['prompt'],
                'completion': data['completion']
            })
    print(f"   Loaded {len(multi_bank)} multi-bank samples")
    
    # Combine
    print("\n3. Combining datasets...")
    combined = existing + multi_bank
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
    
    # Backup
    if train_file.with_suffix('.jsonl.v7.bak').exists():
        pass  # Already backed up
    else:
        import shutil
        shutil.copy(train_file, train_file.with_suffix('.jsonl.v7.bak'))
        shutil.copy(valid_file, valid_file.with_suffix('.jsonl.v7.bak'))
    
    with open(train_file, 'w') as f:
        for sample in new_train:
            f.write(json.dumps(sample) + '\n')
    
    with open(valid_file, 'w') as f:
        for sample in new_valid:
            f.write(json.dumps(sample) + '\n')
    
    print(f"   Saved to {train_file}")
    
    # Count banks
    bank_keywords = ['HDFC', 'ICICI', 'SBI', 'AXIS', 'KOTAK']
    for kw in bank_keywords:
        count = sum(1 for s in new_train if kw in s.get('prompt', '').upper())
        print(f"   {kw}: ~{count} samples")
    
    print("\n✅ Data combined successfully!")
    print("\n5. Starting v8 training...")
    
    # Train v8 - more iterations for more data
    cmd = [
        "mlx_lm.lora",
        "--model", "models/base/phi3-finance-base",
        "--data", "data/training",
        "--train",
        "--iters", "800",  # More iterations for bigger dataset
        "--batch-size", "1",
        "--num-layers", "16",
        "--learning-rate", "1e-5",
        "--adapter-path", "models/adapters/finance-lora-v8",
        "--max-seq-length", "1024"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ v8 training complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")

if __name__ == "__main__":
    combine_and_retrain()
