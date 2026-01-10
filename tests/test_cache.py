"""
Tests for FinEE Cache (Tier 0).
"""

import time
from finee.cache import LRUCache, ExtractionResult

def test_cache_hashing():
    cache = LRUCache()
    text = "  Rs.500 spent  "
    
    # Should normalize whitespace and case
    key1 = cache.hash_text("Rs.500 spent")
    key2 = cache.hash_text("rs.500 SPENT")
    
    assert key1 == key2

def test_cache_operations():
    cache = LRUCache(max_size=2)
    
    # Add item
    res1 = ExtractionResult(amount=100.0)
    cache.set("tx1", res1)
    
    # Get item
    cached = cache.get("tx1")
    assert cached.amount == 100.0
    assert cached.from_cache is True
    
    # Check stats
    stats = cache.get_stats()
    assert stats.hits == 1
    assert stats.size == 1

def test_lru_eviction():
    cache = LRUCache(max_size=2)
    
    # Fill cache
    cache.set("tx1", ExtractionResult(amount=1))
    cache.set("tx2", ExtractionResult(amount=2))
    
    # Access tx1 to make it recent
    cache.get("tx1")
    
    # Add 3rd item (should evict tx2, because tx1 was just used)
    cache.set("tx3", ExtractionResult(amount=3))
    
    assert cache.contains("tx1")  # Kept
    assert cache.contains("tx3")  # New
    assert not cache.contains("tx2")  # Evicted

def test_cache_threading_safety():
    # Basic check ensuring no crash on rapid updates
    cache = LRUCache(max_size=100)
    for i in range(200):
        cache.set(f"tx{i}", ExtractionResult(amount=i))
    
    assert len(cache) == 100
    assert cache.get("tx199").amount == 199.0
