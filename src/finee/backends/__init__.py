"""
FinEE Backends Package.

Provides auto-detection and unified access to LLM backends.
"""

import logging
from typing import Optional, List, Type

from .base import BaseBackend, NoBackendError, BackendLoadError

logger = logging.getLogger(__name__)

# Backend registry (populated on import)
_BACKENDS: List[Type[BaseBackend]] = []
_active_backend: Optional[BaseBackend] = None


def _register_backends():
    """Register available backends."""
    global _BACKENDS
    _BACKENDS = []
    
    # Try MLX (Apple Silicon)
    try:
        from .mlx_backend import MLXBackend
        _BACKENDS.append(MLXBackend)
        logger.debug("MLX backend registered")
    except ImportError:
        pass
    
    # Try Transformers (CUDA/CPU)
    try:
        from .transformers_backend import TransformersBackend
        _BACKENDS.append(TransformersBackend)
        logger.debug("Transformers backend registered")
    except ImportError:
        pass
    
    # Try llama.cpp (CPU fallback)
    try:
        from .llamacpp_backend import LlamaCppBackend
        _BACKENDS.append(LlamaCppBackend)
        logger.debug("llama.cpp backend registered")
    except ImportError:
        pass


def get_available_backends() -> List[str]:
    """
    Get list of available backend names.
    
    Returns:
        List of backend class names that are available
    """
    if not _BACKENDS:
        _register_backends()
    
    available = []
    for backend_cls in _BACKENDS:
        try:
            backend = backend_cls()
            if backend.is_available():
                available.append(backend.name)
        except Exception:
            pass
    
    return available


def auto_select_backend(model_id: str = "Ranjit0034/finance-entity-extractor") -> Optional[BaseBackend]:
    """
    Automatically select the best available backend.
    
    Priority order:
    1. MLX (if on Apple Silicon with mlx installed)
    2. Transformers (if torch + CUDA available)
    3. llama.cpp (if llama-cpp-python installed)
    
    Args:
        model_id: Model ID to load
        
    Returns:
        Instantiated backend or None if none available
    """
    if not _BACKENDS:
        _register_backends()
    
    for backend_cls in _BACKENDS:
        try:
            backend = backend_cls(model_id=model_id)
            if backend.is_available():
                logger.info(f"Selected backend: {backend.name}")
                return backend
        except Exception as e:
            logger.debug(f"Backend {backend_cls.__name__} not available: {e}")
    
    return None


def get_backend(name: Optional[str] = None, 
                model_id: str = "Ranjit0034/finance-entity-extractor") -> Optional[BaseBackend]:
    """
    Get a specific backend by name or auto-select.
    
    Args:
        name: Backend name ('mlx', 'transformers', 'llamacpp') or None for auto
        model_id: Model ID to load
        
    Returns:
        Instantiated backend or None
    """
    global _active_backend
    
    if not _BACKENDS:
        _register_backends()
    
    if name is None:
        _active_backend = auto_select_backend(model_id)
        return _active_backend
    
    name_lower = name.lower().replace('-', '').replace('_', '')
    
    for backend_cls in _BACKENDS:
        backend_name = backend_cls.__name__.lower().replace('backend', '')
        if name_lower in backend_name or backend_name in name_lower:
            try:
                backend = backend_cls(model_id=model_id)
                if backend.is_available():
                    _active_backend = backend
                    return _active_backend
            except Exception as e:
                logger.warning(f"Failed to initialize {backend_cls.__name__}: {e}")
    
    return None


def get_active_backend() -> Optional[BaseBackend]:
    """Get the currently active backend."""
    global _active_backend
    return _active_backend


def set_active_backend(backend: BaseBackend) -> None:
    """Set the active backend."""
    global _active_backend
    _active_backend = backend


# Initialize backends on import
_register_backends()


__all__ = [
    'BaseBackend',
    'NoBackendError', 
    'BackendLoadError',
    'get_available_backends',
    'auto_select_backend',
    'get_backend',
    'get_active_backend',
    'set_active_backend',
]
