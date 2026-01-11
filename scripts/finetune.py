#!/usr/bin/env python3
"""
LLM Fine-tuning Pipeline for Finance Entity Extraction
=======================================================

Fine-tunes a language model on the combined training data
for Indian banking transaction entity extraction.

Supports:
- MLX (Apple Silicon) via mlx-lm
- PyTorch/Transformers (GPU/CPU)

Usage:
    python finetune.py --model microsoft/Phi-3-mini-4k-instruct --epochs 3
"""

import json
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import random

# Check for MLX (Apple Silicon)
try:
    import mlx
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# Check for PyTorch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================================
# DATA PREPARATION
# ============================================================================

class DataPreparer:
    """Prepare training data for fine-tuning."""
    
    SYSTEM_PROMPT = """You are a finance entity extraction assistant for Indian banking. 
Extract structured information from banking SMS/email messages.

Output JSON with these fields (only include if found):
- amount: float (transaction amount)
- type: "debit" or "credit"
- account: string (last 4 digits)
- bank: string (bank name)
- date: string (transaction date)
- reference: string (UPI/NEFT reference)
- merchant: string (business name for P2M)
- beneficiary: string (person name for P2P)
- vpa: string (UPI ID)
- category: string (food, shopping, travel, etc.)
- is_p2m: boolean (true if merchant, false if person)

Be precise. Extract exactly what's in the message."""

    def __init__(self, data_path: Path, val_split: float = 0.1):
        self.data_path = data_path
        self.val_split = val_split
        self.train_data = []
        self.val_data = []
    
    def load_and_split(self) -> Tuple[List[Dict], List[Dict]]:
        """Load data and split into train/val."""
        print(f"Loading data from {self.data_path}...")
        
        all_data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    all_data.append(record)
                except json.JSONDecodeError:
                    continue
        
        print(f"  Loaded {len(all_data):,} records")
        
        # Shuffle
        random.shuffle(all_data)
        
        # Split
        split_idx = int(len(all_data) * (1 - self.val_split))
        self.train_data = all_data[:split_idx]
        self.val_data = all_data[split_idx:]
        
        print(f"  Train: {len(self.train_data):,}, Val: {len(self.val_data):,}")
        
        return self.train_data, self.val_data
    
    def format_for_chat(self, record: Dict) -> Dict:
        """Format record for chat-style fine-tuning."""
        input_text = record.get('input', record.get('text', ''))
        output_text = record.get('output', '{}')
        
        if isinstance(output_text, dict):
            output_text = json.dumps(output_text, ensure_ascii=False)
        
        return {
            'messages': [
                {'role': 'system', 'content': self.SYSTEM_PROMPT},
                {'role': 'user', 'content': input_text},
                {'role': 'assistant', 'content': output_text},
            ]
        }
    
    def format_for_completion(self, record: Dict) -> Dict:
        """Format record for completion-style fine-tuning."""
        input_text = record.get('input', record.get('text', ''))
        output_text = record.get('output', '{}')
        
        if isinstance(output_text, dict):
            output_text = json.dumps(output_text, ensure_ascii=False)
        
        prompt = f"""Extract financial entities from this message:

Message: {input_text}

JSON:"""
        
        return {
            'prompt': prompt,
            'completion': output_text,
        }
    
    def save_formatted(
        self,
        output_dir: Path,
        format_type: str = 'chat'
    ) -> Tuple[Path, Path]:
        """Save formatted train/val data."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = output_dir / 'train.jsonl'
        val_path = output_dir / 'valid.jsonl'
        
        formatter = (
            self.format_for_chat if format_type == 'chat'
            else self.format_for_completion
        )
        
        # Save train
        with open(train_path, 'w', encoding='utf-8') as f:
            for record in self.train_data:
                formatted = formatter(record)
                f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
        
        # Save val
        with open(val_path, 'w', encoding='utf-8') as f:
            for record in self.val_data:
                formatted = formatter(record)
                f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
        
        print(f"  Saved train: {train_path}")
        print(f"  Saved valid: {val_path}")
        
        return train_path, val_path


# ============================================================================
# MLX FINE-TUNING (Apple Silicon)
# ============================================================================

class MLXFineTuner:
    """Fine-tune using MLX-LM on Apple Silicon."""
    
    def __init__(
        self,
        model_name: str,
        output_dir: Path,
        lora_rank: int = 8,
        lora_layers: int = 16,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.lora_rank = lora_rank
        self.lora_layers = lora_layers
    
    def train(
        self,
        train_path: Path,
        val_path: Path,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        save_every: int = 100,
    ):
        """Run MLX-LM LoRA fine-tuning."""
        import subprocess
        
        cmd = [
            sys.executable, '-m', 'mlx_lm.lora',
            '--model', self.model_name,
            '--train',
            '--data', str(train_path.parent),
            '--lora-layers', str(self.lora_layers),
            '--lora-rank', str(self.lora_rank),
            '--batch-size', str(batch_size),
            '--iters', str(epochs * 1000),
            '--learning-rate', str(learning_rate),
            '--save-every', str(save_every),
            '--adapter-path', str(self.output_dir / 'adapters'),
        ]
        
        print(f"\n🚀 Starting MLX-LM LoRA training...")
        print(f"   Command: {' '.join(cmd)}")
        print()
        
        result = subprocess.run(cmd, capture_output=False)
        
        return result.returncode == 0
    
    def fuse(self):
        """Fuse LoRA adapters with base model."""
        import subprocess
        
        adapter_path = self.output_dir / 'adapters'
        fused_path = self.output_dir / 'fused'
        
        cmd = [
            sys.executable, '-m', 'mlx_lm.fuse',
            '--model', self.model_name,
            '--adapter-path', str(adapter_path),
            '--save-path', str(fused_path),
        ]
        
        print(f"\n🔗 Fusing LoRA adapters...")
        result = subprocess.run(cmd, capture_output=False)
        
        return result.returncode == 0


# ============================================================================
# PYTORCH/TRANSFORMERS FINE-TUNING
# ============================================================================

class TransformersFineTuner:
    """Fine-tune using PyTorch/Transformers."""
    
    def __init__(
        self,
        model_name: str,
        output_dir: Path,
        lora_rank: int = 8,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.lora_rank = lora_rank
    
    def train(
        self,
        train_path: Path,
        val_path: Path,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
    ):
        """Run Transformers fine-tuning with PEFT."""
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                Trainer,
                DataCollatorForSeq2Seq,
            )
            from peft import LoraConfig, get_peft_model
            from datasets import load_dataset
        except ImportError as e:
            print(f"❌ Missing dependencies: {e}")
            print("   Run: pip install transformers peft datasets")
            return False
        
        print(f"\n🚀 Loading model: {self.model_name}")
        
        # Load model & tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None,
        )
        
        # Add padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # LoRA config
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=32,
            target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM',
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Load dataset
        dataset = load_dataset(
            'json',
            data_files={
                'train': str(train_path),
                'validation': str(val_path),
            }
        )
        
        # Tokenize
        def tokenize(examples):
            # For chat format
            if 'messages' in examples:
                texts = []
                for msgs in examples['messages']:
                    text = ''
                    for msg in msgs:
                        text += f"<|{msg['role']}|>\n{msg['content']}\n"
                    texts.append(text)
            else:
                texts = [f"{p}\n{c}" for p, c in zip(examples['prompt'], examples['completion'])]
            
            return tokenizer(
                texts,
                truncation=True,
                max_length=512,
                padding='max_length',
            )
        
        tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset['train'].column_names)
        
        # Training args
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=100,
            save_steps=500,
            evaluation_strategy='steps',
            eval_steps=500,
            fp16=torch.cuda.is_available(),
            report_to='none',
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized['train'],
            eval_dataset=tokenized['validation'],
            data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
        )
        
        print(f"\n🚀 Starting training...")
        trainer.train()
        
        # Save
        model.save_pretrained(self.output_dir / 'adapters')
        tokenizer.save_pretrained(self.output_dir / 'adapters')
        
        print(f"\n✅ Saved to: {self.output_dir / 'adapters'}")
        return True


# ============================================================================
# EVALUATION
# ============================================================================

class Evaluator:
    """Evaluate fine-tuned model."""
    
    def __init__(self, model_path: Path, backend: str = 'mlx'):
        self.model_path = model_path
        self.backend = backend
    
    def evaluate(self, test_data: List[Dict], max_samples: int = 100) -> Dict:
        """Evaluate on test data."""
        if self.backend == 'mlx':
            return self._evaluate_mlx(test_data[:max_samples])
        else:
            return self._evaluate_torch(test_data[:max_samples])
    
    def _evaluate_mlx(self, test_data: List[Dict]) -> Dict:
        """Evaluate with MLX."""
        from mlx_lm import load, generate
        
        model, tokenizer = load(str(self.model_path))
        
        correct = 0
        total = 0
        field_matches = {'amount': 0, 'type': 0, 'merchant': 0}
        
        for record in test_data:
            input_text = record.get('input', record.get('text', ''))
            expected = record.get('output', '{}')
            if isinstance(expected, str):
                expected = json.loads(expected)
            
            prompt = f"Extract financial entities:\n\n{input_text}\n\nJSON:"
            
            output = generate(
                model, tokenizer, prompt,
                max_tokens=256,
                temp=0.0,
            )
            
            try:
                predicted = json.loads(output)
                
                # Check fields
                for field in field_matches:
                    if predicted.get(field) == expected.get(field):
                        field_matches[field] += 1
                
                # Full match
                if predicted == expected:
                    correct += 1
            except json.JSONDecodeError:
                pass
            
            total += 1
        
        return {
            'accuracy': correct / total if total > 0 else 0,
            'field_accuracy': {k: v/total for k, v in field_matches.items()},
            'total_samples': total,
        }
    
    def _evaluate_torch(self, test_data: List[Dict]) -> Dict:
        """Evaluate with PyTorch."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        # Load
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path.parent / 'base',
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        model = PeftModel.from_pretrained(base_model, str(self.model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        
        # Similar evaluation logic...
        return {'accuracy': 0, 'note': 'PyTorch evaluation not fully implemented'}


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM Fine-tuning Pipeline")
    parser.add_argument('--data', default='data/training/final_combined_training.jsonl',
                       help='Training data path')
    parser.add_argument('--model', default='microsoft/Phi-3-mini-4k-instruct',
                       help='Base model')
    parser.add_argument('--output', default='models/finetune',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--lora-rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--backend', choices=['mlx', 'torch', 'auto'], default='auto',
                       help='Training backend')
    parser.add_argument('--skip-train', action='store_true', help='Skip training, just prepare data')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate after training')
    
    args = parser.parse_args()
    
    # Determine backend
    if args.backend == 'auto':
        if HAS_MLX:
            backend = 'mlx'
            print("🍎 Using MLX (Apple Silicon)")
        elif HAS_TORCH:
            backend = 'torch'
            print("🔥 Using PyTorch/Transformers")
        else:
            print("❌ No backend available. Install mlx-lm or transformers+peft")
            return
    else:
        backend = args.backend
    
    data_path = Path(args.data)
    output_dir = Path(args.output)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f"run_{timestamp}"
    
    print("=" * 60)
    print("🚀 LLM FINE-TUNING PIPELINE")
    print("=" * 60)
    print(f"  Data: {data_path}")
    print(f"  Model: {args.model}")
    print(f"  Output: {run_dir}")
    print(f"  Backend: {backend}")
    print(f"  Epochs: {args.epochs}")
    
    # Step 1: Prepare data
    print("\n📋 Step 1: Preparing data...")
    preparer = DataPreparer(data_path)
    train_data, val_data = preparer.load_and_split()
    
    formatted_dir = run_dir / 'data'
    format_type = 'chat' if backend == 'torch' else 'completion'
    train_path, val_path = preparer.save_formatted(formatted_dir, format_type)
    
    if args.skip_train:
        print("\n⏭️ Skipping training (--skip-train)")
        return
    
    # Step 2: Train
    print("\n🎯 Step 2: Training...")
    
    if backend == 'mlx':
        trainer = MLXFineTuner(
            model_name=args.model,
            output_dir=run_dir,
            lora_rank=args.lora_rank,
        )
        success = trainer.train(
            train_path, val_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        
        if success:
            print("\n🔗 Step 3: Fusing adapters...")
            trainer.fuse()
    else:
        trainer = TransformersFineTuner(
            model_name=args.model,
            output_dir=run_dir,
            lora_rank=args.lora_rank,
        )
        success = trainer.train(
            train_path, val_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    
    # Step 3: Evaluate
    if args.evaluate and success:
        print("\n📊 Step 4: Evaluating...")
        evaluator = Evaluator(run_dir / 'fused' if backend == 'mlx' else run_dir / 'adapters', backend)
        results = evaluator.evaluate(val_data)
        
        print(f"\n📊 Results:")
        print(f"  Overall Accuracy: {results.get('accuracy', 0):.1%}")
        for field, acc in results.get('field_accuracy', {}).items():
            print(f"  {field}: {acc:.1%}")
        
        # Save results
        with open(run_dir / 'eval_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"  Output: {run_dir}")
    print(f"  Adapters: {run_dir / 'adapters'}")
    if backend == 'mlx':
        print(f"  Fused model: {run_dir / 'fused'}")


if __name__ == "__main__":
    main()
