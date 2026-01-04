"""
LoRA Fine-tuning Module
Wrapper for MLX LoRA training with configuration management.
"""

import subprocess
import json
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    
    # Paths
    base_model: Path = field(default_factory=lambda: Path("models/base/phi3-mini"))
    data_dir: Path = field(default_factory=lambda: Path("data/training"))
    adapter_path: Path = field(default_factory=lambda: Path("models/adapters/finance-lora"))
    
    # Training parameters
    batch_size: int = 1
    lora_layers: int = 8
    iterations: int = 500
    learning_rate: float = 1e-5
    
    # LoRA parameters
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    def to_dict(self) -> Dict:
        return {
            "base_model": str(self.base_model),
            "data_dir": str(self.data_dir),
            "adapter_path": str(self.adapter_path),
            "batch_size": self.batch_size,
            "lora_layers": self.lora_layers,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
        }


class LoRATrainer:
    """
    Wrapper for MLX LoRA fine-tuning.
    
    Provides a Python interface to the mlx_lm.lora command.
    """
    
    def __init__(
        self,
        project_root: Path,
        config: Optional[LoRAConfig] = None
    ):
        """
        Initialize trainer.
        
        Args:
            project_root: Root directory of the project
            config: LoRA configuration (uses defaults if not provided)
        """
        self.project_root = Path(project_root)
        self.config = config or LoRAConfig()
        
        # Make paths absolute
        self.base_model_path = self.project_root / self.config.base_model
        self.data_dir_path = self.project_root / self.config.data_dir
        self.adapter_path = self.project_root / self.config.adapter_path
    
    def validate_setup(self) -> bool:
        """Validate that all required files exist."""
        errors = []
        
        # Check base model
        if not self.base_model_path.exists():
            errors.append(f"Base model not found: {self.base_model_path}")
        
        # Check training data
        train_file = self.data_dir_path / "train.jsonl"
        valid_file = self.data_dir_path / "valid.jsonl"
        
        if not train_file.exists():
            errors.append(f"Training data not found: {train_file}")
        if not valid_file.exists():
            errors.append(f"Validation data not found: {valid_file}")
        
        if errors:
            for error in errors:
                print(f"❌ {error}")
            return False
        
        print("✅ Setup validated successfully")
        return True
    
    def get_train_command(self) -> str:
        """Generate the mlx_lm.lora training command."""
        cmd = f"""mlx_lm.lora \\
    --model {self.base_model_path} \\
    --data {self.data_dir_path} \\
    --train \\
    --batch-size {self.config.batch_size} \\
    --lora-layers {self.config.lora_layers} \\
    --iters {self.config.iterations} \\
    --adapter-path {self.adapter_path}"""
        
        return cmd
    
    def get_fuse_command(self, output_path: Optional[Path] = None) -> str:
        """Generate the mlx_lm.fuse command to merge adapter with base model."""
        if output_path is None:
            output_path = self.project_root / "models/merged/finance-llm"
        
        cmd = f"""mlx_lm.fuse \\
    --model {self.base_model_path} \\
    --adapter-path {self.adapter_path} \\
    --save-path {output_path}"""
        
        return cmd
    
    def print_instructions(self):
        """Print training instructions for the user."""
        print("\n" + "=" * 60)
        print("🎓 FINE-TUNING INSTRUCTIONS")
        print("=" * 60)
        
        print("\n📋 Configuration:")
        print(f"   Base Model: {self.base_model_path}")
        print(f"   Training Data: {self.data_dir_path}")
        print(f"   Output Adapter: {self.adapter_path}")
        print(f"   Iterations: {self.config.iterations}")
        print(f"   LoRA Layers: {self.config.lora_layers}")
        
        print("\n⚠️  Fine-tuning takes 1-2 hours. Run in Terminal (not notebook):")
        print("\n" + "-" * 60)
        print(f"cd {self.project_root}")
        print("source venv/bin/activate")
        print()
        print(self.get_train_command())
        print("-" * 60)
        
        print("\n🔄 After training, merge the adapter:")
        print("-" * 60)
        print(self.get_fuse_command())
        print("-" * 60)
    
    def train(self, dry_run: bool = True) -> bool:
        """
        Run the training command.
        
        Args:
            dry_run: If True, only print command without running
            
        Returns:
            True if training succeeded or dry_run
        """
        if not self.validate_setup():
            return False
        
        cmd = self.get_train_command()
        
        if dry_run:
            print("\n🔍 DRY RUN - Command would be:")
            print(cmd)
            return True
        
        print("\n🚀 Starting training...")
        print(f"Command: {cmd}")
        
        try:
            # Create adapter directory
            self.adapter_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Run training
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=self.project_root,
                check=True
            )
            
            print("✅ Training completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def fuse(self, output_path: Optional[Path] = None, dry_run: bool = True) -> bool:
        """
        Fuse adapter with base model.
        
        Args:
            output_path: Path to save merged model
            dry_run: If True, only print command
            
        Returns:
            True if fuse succeeded or dry_run
        """
        cmd = self.get_fuse_command(output_path)
        
        if dry_run:
            print("\n🔍 DRY RUN - Command would be:")
            print(cmd)
            return True
        
        print("\n🔄 Fusing model...")
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=self.project_root,
                check=True
            )
            
            print("✅ Model fused successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Fuse failed: {e}")
            return False
    
    def save_config(self, path: Optional[Path] = None):
        """Save training configuration to JSON."""
        if path is None:
            path = self.adapter_path / "training_config.json"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        print(f"💾 Config saved to {path}")


if __name__ == "__main__":
    from pathlib import Path
    
    PROJECT = Path.home() / "llm-mail-trainer"
    
    trainer = LoRATrainer(project_root=PROJECT)
    trainer.validate_setup()
    trainer.print_instructions()
