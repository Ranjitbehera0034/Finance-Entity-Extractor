"""
FinEE Transformers Backend - PyTorch/CUDA backend.

Uses Hugging Face Transformers for GPU inference on NVIDIA cards.
"""

import logging
from typing import Optional

from .base import BaseBackend, BackendLoadError

logger = logging.getLogger(__name__)

# Check for Transformers availability
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    torch = None


class TransformersBackend(BaseBackend):
    """
    PyTorch/Transformers backend for NVIDIA GPU inference.
    
    Requirements:
    - torch with CUDA support (or CPU)
    - transformers package
    - accelerate for device mapping
    """
    
    def __init__(self, model_id: str = "Ranjit0034/finance-entity-extractor",
                 device: Optional[str] = None,
                 torch_dtype: Optional[str] = "float16"):
        """
        Initialize Transformers backend.
        
        Args:
            model_id: Hugging Face model ID
            device: Device to use ('cuda', 'cpu', or None for auto)
            torch_dtype: PyTorch dtype ('float16', 'bfloat16', 'float32')
        """
        super().__init__(model_id)
        self.device = device
        self.torch_dtype_str = torch_dtype
    
    def is_available(self) -> bool:
        """Check if Transformers/PyTorch is available."""
        if not HAS_TRANSFORMERS:
            return False
        
        # Prefer CUDA, but also work on CPU
        return True
    
    def _get_torch_dtype(self):
        """Get PyTorch dtype from string."""
        if not HAS_TRANSFORMERS:
            return None
        
        dtype_map = {
            'float16': torch.float16,
            'fp16': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'float32': torch.float32,
            'fp32': torch.float32,
        }
        return dtype_map.get(self.torch_dtype_str, torch.float16)
    
    def _get_device(self) -> str:
        """Determine best device to use."""
        if self.device:
            return self.device
        
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load model with Transformers.
        
        Args:
            model_path: Optional local path (overrides model_id)
            
        Returns:
            True if successful
        """
        if not HAS_TRANSFORMERS:
            raise BackendLoadError("Transformers not installed. Run: pip install transformers torch")
        
        path = model_path or self.model_id
        
        try:
            logger.info(f"Loading model with Transformers: {path}")
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            
            # Load model with auto device mapping
            self._model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=self._get_torch_dtype(),
                device_map="auto",
                trust_remote_code=True,
            )
            
            self._loaded = True
            device = next(self._model.parameters()).device
            logger.info(f"Transformers model loaded on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Transformers model: {e}")
            raise BackendLoadError(f"Transformers model load failed: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 200,
                 temperature: float = 0.1, **kwargs) -> str:
        """
        Generate text using Transformers.
        
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
            # Tokenize
            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode (only new tokens)
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response
            
        except Exception as e:
            logger.error(f"Transformers generation failed: {e}")
            return ""
    
    def unload(self) -> None:
        """Free model from GPU memory."""
        super().unload()
        
        # Force CUDA memory cleanup
        if HAS_TRANSFORMERS and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    
    def get_info(self):
        """Get backend info including device."""
        info = super().get_info()
        if HAS_TRANSFORMERS:
            info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['gpu_name'] = torch.cuda.get_device_name(0)
        return info
