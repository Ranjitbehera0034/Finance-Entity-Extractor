"""
FinEE llama.cpp Backend - CPU fallback using llama-cpp-python.

Works on any platform (Linux, Windows, macOS) without GPU.
"""

import logging
from typing import Optional
from pathlib import Path

from .base import BaseBackend, BackendLoadError

logger = logging.getLogger(__name__)

# Check for llama-cpp-python availability
try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False


class LlamaCppBackend(BaseBackend):
    """
    llama.cpp backend for CPU inference.
    
    Works on any platform without GPU requirements.
    Uses GGUF format models.
    
    Requirements:
    - llama-cpp-python package
    - GGUF format model file
    """
    
    def __init__(self, model_id: str = "Ranjit0034/finance-entity-extractor",
                 n_ctx: int = 4096,
                 n_threads: Optional[int] = None):
        """
        Initialize llama.cpp backend.
        
        Args:
            model_id: Hugging Face model ID or local GGUF path
            n_ctx: Context length
            n_threads: Number of CPU threads (None = auto)
        """
        super().__init__(model_id)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._gguf_path: Optional[str] = None
    
    def is_available(self) -> bool:
        """Check if llama-cpp-python is available."""
        return HAS_LLAMA_CPP
    
    def _find_gguf_file(self, path: str) -> Optional[str]:
        """
        Find GGUF file in a directory or verify path.
        
        Args:
            path: Directory or file path
            
        Returns:
            Path to GGUF file or None
        """
        path_obj = Path(path)
        
        # If it's a file, check if it's GGUF
        if path_obj.is_file() and path_obj.suffix == '.gguf':
            return str(path_obj)
        
        # If it's a directory, look for GGUF files
        if path_obj.is_dir():
            gguf_files = list(path_obj.glob('*.gguf'))
            if gguf_files:
                # Prefer q4_k_m, then f16, then any
                for pattern in ['*q4_k_m*', '*f16*', '*']:
                    for f in gguf_files:
                        if pattern == '*' or pattern.replace('*', '') in f.name.lower():
                            return str(f)
        
        return None
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load GGUF model with llama.cpp.
        
        Args:
            model_path: Path to GGUF file or directory containing GGUF
            
        Returns:
            True if successful
        """
        if not HAS_LLAMA_CPP:
            raise BackendLoadError("llama-cpp-python not installed. Run: pip install llama-cpp-python")
        
        path = model_path or self.model_id
        
        # Find GGUF file
        gguf_path = self._find_gguf_file(path)
        
        if not gguf_path:
            # Try to download from HuggingFace
            try:
                from huggingface_hub import hf_hub_download
                gguf_path = hf_hub_download(
                    repo_id=self.model_id,
                    filename="finance-extractor-v8-f16.gguf"
                )
            except Exception as e:
                raise BackendLoadError(f"Could not find GGUF file: {path}. Error: {e}")
        
        try:
            logger.info(f"Loading GGUF model: {gguf_path}")
            
            self._model = Llama(
                model_path=gguf_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
            
            self._gguf_path = gguf_path
            self._loaded = True
            logger.info("llama.cpp model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load llama.cpp model: {e}")
            raise BackendLoadError(f"llama.cpp model load failed: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 200,
                 temperature: float = 0.1, **kwargs) -> str:
        """
        Generate text using llama.cpp.
        
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
            output = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["\n\n", "</s>", "<|end|>"],
                echo=False,
                **kwargs
            )
            
            return output["choices"][0]["text"]
            
        except Exception as e:
            logger.error(f"llama.cpp generation failed: {e}")
            return ""
    
    def get_info(self):
        """Get backend info including GGUF path."""
        info = super().get_info()
        info['gguf_path'] = self._gguf_path
        info['n_ctx'] = self.n_ctx
        info['n_threads'] = self.n_threads
        return info
