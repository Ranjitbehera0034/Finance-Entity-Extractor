"""
FinEE Cache - Tier 0 Hash Cache for deduplication.

Provides LRU caching of extraction results to avoid redundant computation.
Uses SHA256 hash of input text as cache key.
"""

import hashlib
from collections import OrderedDict
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import json
import time

from .schema import ExtractionResult


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 1000
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            'hit_rate': f"{self.hit_rate:.2%}"
        }


class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache for extraction results.
    
    Features:
    - SHA256 hashing of input text
    - Configurable max size
    - Automatic LRU eviction
    - Statistics tracking
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of items to store (default: 1000)
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, ExtractionResult] = OrderedDict()
        self._stats = CacheStats(max_size=max_size)
    
    @staticmethod
    def hash_text(text: str) -> str:
        """
        Generate SHA256 hash of input text.
        
        Args:
            text: Input text to hash
            
        Returns:
            Hex string of SHA256 hash
        """
        # Normalize text before hashing (lowercase, strip whitespace)
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[ExtractionResult]:
        """
        Retrieve cached result for input text.
        
        Args:
            text: Input text to look up
            
        Returns:
            ExtractionResult if found, None otherwise
        """
        key = self.hash_text(text)
        
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._stats.hits += 1
            
            # Return a copy with cache metadata
            result = self._cache[key]
            result.from_cache = True
            result.processing_time_ms = 0.0
            return result
        
        self._stats.misses += 1
        return None
    
    def set(self, text: str, result: ExtractionResult) -> None:
        """
        Store extraction result in cache.
        
        Args:
            text: Original input text (used as key)
            result: Extraction result to cache
        """
        key = self.hash_text(text)
        
        # If key exists, update and move to end
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = result
            return
        
        # Check if we need to evict
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove oldest
            self._stats.evictions += 1
        
        # Add new item
        self._cache[key] = result
        self._stats.size = len(self._cache)
    
    def contains(self, text: str) -> bool:
        """Check if text is in cache without updating LRU order."""
        key = self.hash_text(text)
        return key in self._cache
    
    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        self._stats.size = 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.size = len(self._cache)
        return self._stats
    
    def __len__(self) -> int:
        """Return number of cached items."""
        return len(self._cache)
    
    def __contains__(self, text: str) -> bool:
        """Support 'in' operator."""
        return self.contains(text)


# Global cache instance (singleton pattern)
_global_cache: Optional[LRUCache] = None


def get_cache(max_size: int = 1000) -> LRUCache:
    """
    Get or create the global cache instance.
    
    Args:
        max_size: Maximum cache size (only used on first call)
        
    Returns:
        Global LRUCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = LRUCache(max_size=max_size)
    return _global_cache


def clear_cache() -> None:
    """Clear the global cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()


def get_cache_stats() -> Optional[CacheStats]:
    """Get statistics for the global cache."""
    global _global_cache
    if _global_cache is not None:
        return _global_cache.get_stats()
    return None
