#!/usr/bin/env python3
"""
FinEE Training Pipeline v1.0

Master orchestrator for training the Finance Entity Extractor.
Handles data generation, domain adaptation, fine-tuning, and export.
"""

import argparse
import json
import subprocess
import sys
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Pipeline Configuration
CONFIG = {
    "version": "1.0.0",
    "project_name": "finee",
    
    "models": {
        "base": "microsoft/Phi-3-mini-4k-instruct",
        "domain": "models/base/phi3-finance-base",
        "final": "models/finee-v1.0",
        "adapter": "models/adapters/finee-adapter-v1",
    },
    
    "data_generation": {
        "script": "scripts/generate_comprehensive_data.py",
        "output_dir": "data/training",
        "samples": 10000,
    },
    
    "domain_pretrain": {
        "enabled": False,  # Skip if using pre-trained base
        "script": "scripts/domain_pretrain.py",
        "iters": 2000,
    },
    
    "finetune": {
        "script": "scripts/retrain_v8.py", # Using latest retraining script logic
        "iters": 1000,
        "batch_size": 4, # Increased for M-series
        "learning_rate": 1e-5,
        "lora_layers": 16,
    },
    
    "evaluation": {
        "script": "scripts/test_multi_bank.py",
        "benchmark_dir": "data/benchmark",
    },
    
    "export": {
        "script": "scripts/upload_to_hf.py",
        "repo_id": "Ranjit0034/finance-entity-extractor",
    }
}

class Pipeline:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.start_time = time.time()
        self.ensure_directories()

    def ensure_directories(self):
        """Create necessary directories."""
        dirs = [
            "data/training",
            "data/benchmark",
            "models/base",
            "models/adapters",
            "logs"
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def run_step(self, name: str, cmd: List[str], cwd: str = ".") -> bool:
        """Run a single pipeline step."""
        logger.info(f"▶️  STARTING STEP: {name}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        if self.dry_run:
            logger.info("Dry run - Skipping execution")
            return True
        
        try:
            subprocess.run(cmd, cwd=cwd, check=True)
            logger.info(f"✅ COMPLETED STEP: {name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ FAILED STEP: {name}")
            logger.error(str(e))
            return False

    def check_dependencies(self):
        """Verify dependencies are installed."""
        logger.info("Verifying dependencies...")
        try:
            import mlx.core
            import finee
            logger.info(f"Found finee version: {finee.__version__}")
            return True
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error("Please run: pip install -e .[metal]")
            return False

    def generate_data(self):
        """Step 1: Generate synthetic training data."""
        script = CONFIG["data_generation"]["script"]
        return self.run_step(
            "Data Generation",
            [sys.executable, script]
        )

    def domain_pretrain(self):
        """Step 2: Domain Adaptation (Optional)."""
        if not CONFIG["domain_pretrain"]["enabled"]:
            logger.info("Skipping domain pre-training (disabled in config)")
            return True
            
        script = CONFIG["domain_pretrain"]["script"]
        return self.run_step(
            "Domain Pre-training",
            [sys.executable, script]
        )

    def finetune(self):
        """Step 3: Fine-tuning."""
        # We'll use mlx_lm directly or the wrapper script
        # Using direct mlx_lm command for transparency
        cmd = [
            "mlx_lm.lora",
            "--model", CONFIG["models"]["base"],
            "--train",
            "--data", CONFIG["data_generation"]["output_dir"],
            "--adapter-path", CONFIG["models"]["adapter"],
            "--iters", str(CONFIG["finetune"]["iters"]),
            "--batch-size", str(CONFIG["finetune"]["batch_size"]),
            "--learning-rate", str(CONFIG["finetune"]["learning_rate"]),
            "--lora-layers", str(CONFIG["finetune"]["lora_layers"]),
            "--seed", "42"
        ]
        return self.run_step("Fine-tuning", cmd)

    def fuse_model(self):
        """Step 4: Fuse adapters."""
        cmd = [
            "mlx_lm.fuse",
            "--model", CONFIG["models"]["base"],
            "--adapter-path", CONFIG["models"]["adapter"],
            "--save-path", CONFIG["models"]["final"]
        ]
        return self.run_step("Model Fusion", cmd)

    def evaluate(self):
        """Step 5: Evaluation."""
        script = CONFIG["evaluation"]["script"]
        return self.run_step(
            "Evaluation",
            [sys.executable, script]
        )

    def export(self):
        """Step 6: Export/Upload."""
        script = CONFIG["export"]["script"]
        return self.run_step(
            "HugginFace Export",
            [sys.executable, script]
        )

    def run_all(self):
        """Run full pipeline."""
        if not self.check_dependencies():
            return
        
        steps = [
            self.generate_data,
            self.domain_pretrain,
            self.finetune,
            self.fuse_model,
            self.evaluate,
            self.export
        ]
        
        for step in steps:
            if not step():
                logger.error("Pipeline aborted due to failure.")
                sys.exit(1)
        
        duration = time.time() - self.start_time
        logger.info(f"🎉 Pipeline completed successfully in {duration/60:.2f} minutes")

def main():
    parser = argparse.ArgumentParser(description="FinEE Training Pipeline")
    parser.add_argument("--step", choices=["data", "pretrain", "finetune", "fuse", "eval", "export", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()
    
    pipeline = Pipeline(dry_run=args.dry_run)
    
    if args.step == "all":
        pipeline.run_all()
    else:
        pipeline.check_dependencies()
        steps = {
            "data": pipeline.generate_data,
            "pretrain": pipeline.domain_pretrain,
            "finetune": pipeline.finetune,
            "fuse": pipeline.fuse_model,
            "eval": pipeline.evaluate,
            "export": pipeline.export
        }
        steps[args.step]()

if __name__ == "__main__":
    main()
