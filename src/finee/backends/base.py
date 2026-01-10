"""
FinEE Backends - Abstract interface for LLM backends.

All LLM backends must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseBackend(ABC):
    """
    Abstract base class for LLM backends.
    
    Any backend (MLX, Transformers, llama.cpp) must implement these methods.
    """
    
    def __init__(self, model_id: str = "Ranjit0034/finance-entity-extractor"):
        """
        Initialize backend.
        
        Args:
            model_id: Hugging Face model ID or local path
        """
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._loaded = False
    
    @property
    def name(self) -> str:
        """Return backend name."""
        return self.__class__.__name__
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this backend can be used on the current system.
        
        Returns:
            True if all dependencies are installed and hardware is compatible
        """
        raise NotImplementedError
    
    @abstractmethod
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the model into memory.
        
        Args:
            model_path: Optional local path (overrides model_id)
            
        Returns:
            True if model loaded successfully
        """
        raise NotImplementedError
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 200, 
                 temperature: float = 0.1, **kwargs) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        raise NotImplementedError
    
    def generate_batch(self, prompts: List[str], max_tokens: int = 200,
                       temperature: float = 0.1, **kwargs) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Default implementation calls generate() in a loop.
        Backends may override for batch optimization.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        return [self.generate(p, max_tokens, temperature, **kwargs) for p in prompts]
    
    def unload(self) -> None:
        """
        Free model from memory.
        
        Call this when done with the model to free GPU/system memory.
        """
        self._model = None
        self._tokenizer = None
        self._loaded = False
        logger.info(f"{self.name}: Model unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._loaded
    
    def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        return {
            'name': self.name,
            'model_id': self.model_id,
            'available': self.is_available(),
            'loaded': self.is_loaded,
        }
    
    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return f"{self.name}(model={self.model_id}, {status})"


class NoBackendError(Exception):
    """Raised when no LLM backend is available."""
    pass


class BackendLoadError(Exception):
    """Raised when backend fails to load model."""
    pass
