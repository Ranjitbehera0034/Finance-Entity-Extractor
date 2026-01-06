"""
Domain Pre-training Script for MLX.

Performs continued pre-training on the financial domain corpus
to teach the model Indian banking/finance vocabulary before
task-specific fine-tuning.

Approach:
    1. Load base Phi-3 model
    2. Continue pre-training on financial corpus (1-2 epochs)
    3. Save domain-adapted model
    4. Use this as base for fine-tuning

Author: Ranjit Behera
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PretrainingConfig:
    """Configuration for domain pre-training."""
    
    # Paths
    base_model: str = "models/base/phi3-mini"
    corpus_file: str = "data/corpus/combined/corpus.jsonl"
    output_dir: str = "models/domain-pretrained/phi3-finance"
    
    # Training parameters
    batch_size: int = 1
    learning_rate: float = 5e-6  # Lower LR for pre-training
    num_epochs: int = 1
    max_seq_length: int = 2048
    save_every: int = 500
    
    # LoRA parameters for efficient pre-training
    use_lora: bool = True
    lora_rank: int = 16  # Higher rank for pre-training
    num_layers: int = 16  # More layers for pre-training
    
    def to_dict(self) -> Dict:
        return {
            "base_model": self.base_model,
            "corpus_file": self.corpus_file,
            "output_dir": self.output_dir,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "max_seq_length": self.max_seq_length,
            "save_every": self.save_every,
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "num_layers": self.num_layers
        }


class DomainPretrainer:
    """
    Handles domain pre-training on financial corpus.
    
    This uses MLX's continued pre-training capability to adapt
    the base model to Indian financial domain.
    """
    
    def __init__(self, config: PretrainingConfig = None):
        self.config = config or PretrainingConfig()
        self.project_root = Path.cwd()
    
    def prepare_corpus(self) -> bool:
        """Verify and prepare corpus for pre-training."""
        corpus_path = self.project_root / self.config.corpus_file
        
        if not corpus_path.exists():
            logger.error(f"Corpus file not found: {corpus_path}")
            logger.info("Run scripts/corpus_collection/collect_corpus.py first")
            return False
        
        # Count documents and estimate tokens
        with open(corpus_path) as f:
            lines = f.readlines()
        
        total_words = 0
        for line in lines:
            try:
                doc = json.loads(line)
                total_words += len(doc.get("text", "").split())
            except:
                pass
        
        estimated_tokens = int(total_words * 1.3)
        
        logger.info(f"Corpus: {len(lines):,} documents, ~{estimated_tokens:,} tokens")
        
        if estimated_tokens < 100_000:
            logger.warning("Corpus is very small. Consider adding more data.")
        
        return True
    
    def convert_to_mlx_format(self) -> Path:
        """Convert corpus to MLX training format."""
        corpus_path = self.project_root / self.config.corpus_file
        output_dir = self.project_root / "data/pretrain_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_file = output_dir / "train.jsonl"
        valid_file = output_dir / "valid.jsonl"
        
        # Read corpus
        with open(corpus_path) as f:
            documents = [json.loads(line) for line in f]
        
        # Shuffle and split
        import random
        random.shuffle(documents)
        
        split_idx = int(len(documents) * 0.95)
        train_docs = documents[:split_idx]
        valid_docs = documents[split_idx:]
        
        # Convert to MLX format (just "text" field for language modeling)
        with open(train_file, 'w') as f:
            for doc in train_docs:
                f.write(json.dumps({"text": doc.get("text", "")}) + '\n')
        
        with open(valid_file, 'w') as f:
            for doc in valid_docs:
                f.write(json.dumps({"text": doc.get("text", "")}) + '\n')
        
        logger.info(f"Created: {len(train_docs)} train, {len(valid_docs)} valid samples")
        
        return output_dir
    
    def calculate_iterations(self) -> int:
        """Calculate number of iterations for specified epochs."""
        corpus_path = self.project_root / self.config.corpus_file
        
        with open(corpus_path) as f:
            num_docs = sum(1 for _ in f)
        
        # Rough estimate: docs / batch_size * epochs
        iters = int(num_docs / self.config.batch_size * self.config.num_epochs)
        
        # Cap at reasonable number
        return min(iters, 5000)
    
    def get_pretrain_command(self, data_dir: Path) -> str:
        """Generate MLX pre-training command."""
        iters = self.calculate_iterations()
        
        cmd = f"""mlx_lm.lora \\
    --model {self.config.base_model} \\
    --data {data_dir} \\
    --train \\
    --batch-size {self.config.batch_size} \\
    --num-layers {self.config.num_layers} \\
    --learning-rate {self.config.learning_rate} \\
    --iters {iters} \\
    --save-every {self.config.save_every} \\
    --adapter-path {self.config.output_dir}"""
        
        return cmd
    
    def run_pretraining(self, dry_run: bool = False) -> bool:
        """Run the pre-training process."""
        logger.info("=" * 60)
        logger.info("🎓 DOMAIN PRE-TRAINING")
        logger.info("=" * 60)
        
        # Step 1: Verify corpus
        if not self.prepare_corpus():
            return False
        
        # Step 2: Prepare data
        logger.info("\n📁 Preparing training data...")
        data_dir = self.convert_to_mlx_format()
        
        # Step 3: Generate command
        cmd = self.get_pretrain_command(data_dir)
        
        logger.info(f"\n📋 Pre-training Configuration:")
        for key, value in self.config.to_dict().items():
            logger.info(f"   {key}: {value}")
        
        logger.info(f"\n🔧 Command:\n{cmd}")
        
        if dry_run:
            logger.info("\n[DRY RUN] Command not executed.")
            return True
        
        # Step 4: Run training
        logger.info("\n🚀 Starting pre-training...")
        logger.info("   This may take several hours depending on corpus size.")
        
        try:
            # Create output directory
            output_path = Path(self.config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save config
            with open(output_path / "pretrain_config.json", 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            
            # Run training
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=self.project_root,
                check=True
            )
            
            logger.info("\n✅ Pre-training completed!")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"\n❌ Pre-training failed: {e}")
            return False
    
    def verify_pretrained_model(self) -> bool:
        """Verify the pre-trained model works."""
        output_path = Path(self.config.output_dir)
        
        if not (output_path / "adapters.safetensors").exists():
            logger.error("Pre-trained adapter not found")
            return False
        
        logger.info("Testing pre-trained model...")
        
        try:
            from mlx_lm import load, generate
            
            model, tokenizer = load(
                self.config.base_model,
                adapter_path=str(output_path)
            )
            
            # Test with financial text
            prompt = "UPI transaction reference number 123456789012 indicates"
            response = generate(model, tokenizer, prompt=prompt, max_tokens=50)
            
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Response: {response}")
            
            return True
            
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            return False
    
    def print_instructions(self):
        """Print step-by-step instructions."""
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    DOMAIN PRE-TRAINING INSTRUCTIONS                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  STEP 1: Collect Corpus                                              ║
║  ──────────────────────                                              ║
║  a) Export Gmail to MBOX (Google Takeout)                           ║
║  b) Place bank statement PDFs in data/raw/pdfs/                     ║
║  c) Run: python scripts/corpus_collection/collect_corpus.py        ║
║                                                                      ║
║  STEP 2: Verify Corpus                                               ║
║  ─────────────────────                                               ║
║  Check data/corpus/combined/corpus.jsonl exists                     ║
║  Target: 1M+ tokens (ideally 10M+)                                  ║
║                                                                      ║
║  STEP 3: Run Pre-training                                            ║
║  ────────────────────────                                            ║
║  python scripts/domain_pretrain.py                                  ║
║                                                                      ║
║  STEP 4: Verify & Use                                                ║
║  ────────────────────                                                ║
║  Use models/domain-pretrained/phi3-finance as base for fine-tuning ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
        """)


def analyze_corpus(corpus_path: str):
    """Analyze corpus quality before pre-training."""
    print("\n📊 Corpus Analysis")
    print("=" * 60)
    
    path = Path(corpus_path)
    if not path.exists():
        print(f"❌ Corpus not found: {path}")
        return
    
    total_docs = 0
    total_words = 0
    sources = {}
    sample_texts = []
    
    with open(path) as f:
        for i, line in enumerate(f):
            try:
                doc = json.loads(line)
                text = doc.get("text", "")
                source = doc.get("source", "unknown")
                
                words = len(text.split())
                total_docs += 1
                total_words += words
                
                sources[source] = sources.get(source, 0) + 1
                
                if i < 3:
                    sample_texts.append(text[:200])
                    
            except:
                pass
    
    est_tokens = int(total_words * 1.3)
    
    print(f"Documents:  {total_docs:,}")
    print(f"Words:      {total_words:,}")
    print(f"Est Tokens: {est_tokens:,}")
    print(f"\nBy Source:")
    for source, count in sorted(sources.items()):
        print(f"  {source:15} {count:,}")
    
    print(f"\nSample Texts:")
    for i, text in enumerate(sample_texts, 1):
        print(f"  [{i}] {text}...")
    
    # Quality assessment
    print(f"\n{'=' * 60}")
    if est_tokens >= 10_000_000:
        print("✅ EXCELLENT: Corpus has 10M+ tokens - ideal for pre-training")
    elif est_tokens >= 1_000_000:
        print("✅ GOOD: Corpus has 1M+ tokens - sufficient for basic pre-training")
    elif est_tokens >= 100_000:
        print("⚠️  MARGINAL: Corpus has 100K+ tokens - may help but limited")
    else:
        print("❌ INSUFFICIENT: Corpus too small - add more data")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Domain Pre-training for Financial LLM")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    parser.add_argument("--analyze", action="store_true", help="Analyze corpus only")
    parser.add_argument("--corpus", default="data/corpus/combined/corpus.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_corpus(args.corpus)
        return
    
    config = PretrainingConfig(
        corpus_file=args.corpus,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    pretrainer = DomainPretrainer(config)
    pretrainer.print_instructions()
    
    if args.dry_run:
        pretrainer.prepare_corpus()
        data_dir = pretrainer.convert_to_mlx_format()
        cmd = pretrainer.get_pretrain_command(data_dir)
        print(f"\n🔧 Command (not executed):\n{cmd}")
    else:
        pretrainer.run_pretraining()


if __name__ == "__main__":
    main()
