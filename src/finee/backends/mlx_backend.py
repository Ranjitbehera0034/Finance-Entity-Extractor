"""
FinEE MLX Backend - Apple Silicon optimized backend.

Uses mlx-lm for fast inference on M1/M2/M3 chips.
"""

import logging
from typing import Optional

from .base import BaseBackend, BackendLoadError

logger = logging.getLogger(__name__)

# Check for MLX availability
try:
    import mlx.core as mx
    from mlx_lm import load, generate
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class MLXBackend(BaseBackend):
    """
    Apple Silicon (MLX) backend for fast local inference.
    
    Requirements:
    - Apple Silicon Mac (M1/M2/M3)
    - mlx-lm package installed
    """
    
    def __init__(self, model_id: str = "Ranjit0034/finance-entity-extractor",
                 adapter_path: str = "adapters"):
        """
        Initialize MLX backend.
        
        Args:
            model_id: Hugging Face model ID
            adapter_path: Path to LoRA adapters (relative to model)
        """
        super().__init__(model_id)
        self.adapter_path = adapter_path
    
    def is_available(self) -> bool:
        """Check if MLX is available on this system."""
        if not HAS_MLX:
            return False
        
        # Check if running on Apple Silicon
        try:
            import platform
            if platform.system() != 'Darwin':
                return False
            if platform.processor() not in ('arm', 'arm64'):
                return False
            return True
        except Exception:
            return False
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load model with MLX.
        
        Args:
            model_path: Optional local path (overrides model_id)
            
        Returns:
            True if successful
        """
        if not HAS_MLX:
            raise BackendLoadError("MLX not installed. Run: pip install mlx-lm")
        
        path = model_path or self.model_id
        
        try:
            logger.info(f"Loading model with MLX: {path}")
            
            # Load model with adapters
            self._model, self._tokenizer = load(
                path,
                adapter_path=self.adapter_path
            )
            
            self._loaded = True
            logger.info("MLX model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}")
            raise BackendLoadError(f"MLX model load failed: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 200,
                 temperature: float = 0.1, **kwargs) -> str:
        """
        Generate text using MLX.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        if not self._loaded:
            self.load_model()
        
        try:
            response = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False,
            )
            
            return response
            
        except Exception as e:
            logger.error(f"MLX generation failed: {e}")
            return ""
    
    def unload(self) -> None:
        """Free MLX model from memory."""
        super().unload()
        
        # Force garbage collection for MLX
        try:
            import gc
            gc.collect()
        except Exception:
            pass
