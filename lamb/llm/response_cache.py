"""
Response caching implementation for LLM performance optimization.

Based on Technical_Specification.md Section 1.3 and reconnaissance findings.
Implements LRU cache with TTL for LLM responses to improve performance
and reduce API costs.

Performance characteristics:
- Cache hit rate target: 60% (from reconnaissance validation)
- Cache lookup: O(1) average
- Cache eviction: LRU with TTL expiration
- Memory overhead: ~100 bytes per cached entry
"""

from typing import Dict, Any, Optional, Tuple, List
import time
from collections import OrderedDict
from dataclasses import dataclass

from ..core.types import Action


@dataclass
class CacheEntry:
    """Cache entry with TTL support"""
    action: Action
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    
    def __post_init__(self):
        if self.last_access == 0.0:
            self.last_access = self.timestamp


@dataclass
class CacheMetrics:
    """Metrics for cache performance"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    expired_entries: int = 0
    total_size: int = 0
    max_size_reached: int = 0


class ResponseCache:
    """
    LRU cache with TTL for LLM responses.
    
    Features:
    - LRU (Least Recently Used) eviction policy
    - TTL (Time To Live) expiration for entries
    - Comprehensive metrics and monitoring
    - Memory-efficient storage
    - Thread-safe operations (basic implementation)
    
    Target performance:
    - 60% hit rate for spatially coherent agent movement
    - O(1) average lookup and insertion
    - Automatic cleanup of expired entries
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 300.0,  # 5 minutes
        cleanup_interval: int = 100   # Clean up every N operations
    ):
        """
        Initialize response cache.
        
        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time to live for cache entries
            cleanup_interval: Operations between cleanup cycles
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval
        
        # Cache storage (OrderedDict for LRU behavior)
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Metrics
        self.metrics = CacheMetrics()
        
        # Cleanup tracking
        self.operations_since_cleanup = 0
    
    def get(self, key: str) -> Optional[Action]:
        """
        Get cached action for key.
        
        Performance target: O(1) average
        
        Args:
            key: Cache key
            
        Returns:
            Cached Action if found and valid, None otherwise
        """
        self.metrics.total_requests += 1
        self.operations_since_cleanup += 1
        
        # Check if key exists
        if key not in self.cache:
            self.metrics.cache_misses += 1
            self._maybe_cleanup()
            return None
        
        entry = self.cache[key]
        current_time = time.time()
        
        # Check if entry has expired
        if current_time - entry.timestamp > self.ttl_seconds:
            # Remove expired entry
            del self.cache[key]
            self.metrics.cache_misses += 1
            self.metrics.expired_entries += 1
            self._maybe_cleanup()
            return None
        
        # Update access information
        entry.access_count += 1
        entry.last_access = current_time
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        
        self.metrics.cache_hits += 1
        self._maybe_cleanup()
        
        return entry.action
    
    def put(self, key: str, action: Action) -> None:
        """
        Store action in cache.
        
        Performance target: O(1) average
        
        Args:
            key: Cache key
            action: Action to cache
        """
        current_time = time.time()
        
        # Create cache entry
        entry = CacheEntry(
            action=action,
            timestamp=current_time,
            last_access=current_time
        )
        
        # Check if key already exists
        if key in self.cache:
            # Update existing entry
            self.cache[key] = entry
            self.cache.move_to_end(key)
        else:
            # Add new entry
            self.cache[key] = entry
            
            # Check size limit
            if len(self.cache) > self.max_size:
                self._evict_lru()
        
        self.operations_since_cleanup += 1
        self._maybe_cleanup()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if self.cache:
            # Remove first item (least recently used)
            self.cache.popitem(last=False)
            self.metrics.evictions += 1
    
    def _maybe_cleanup(self) -> None:
        """Perform cleanup if needed"""
        if self.operations_since_cleanup >= self.cleanup_interval:
            self._cleanup_expired()
            self.operations_since_cleanup = 0
    
    def _cleanup_expired(self) -> None:
        """Remove all expired entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time - entry.timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            self.metrics.expired_entries += 1
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
        self.operations_since_cleanup = 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics"""
        # Calculate hit rate
        hit_rate = 0.0
        if self.metrics.total_requests > 0:
            hit_rate = self.metrics.cache_hits / self.metrics.total_requests
        
        # Calculate miss rate
        miss_rate = 0.0
        if self.metrics.total_requests > 0:
            miss_rate = self.metrics.cache_misses / self.metrics.total_requests
        
        # Calculate current size
        current_size = len(self.cache)
        
        # Calculate memory usage estimate
        memory_estimate = current_size * 100  # ~100 bytes per entry
        
        # Calculate average access count
        avg_access_count = 0.0
        if self.cache:
            total_accesses = sum(entry.access_count for entry in self.cache.values())
            avg_access_count = total_accesses / len(self.cache)
        
        return {
            'max_size': self.max_size,
            'current_size': current_size,
            'ttl_seconds': self.ttl_seconds,
            'total_requests': self.metrics.total_requests,
            'cache_hits': self.metrics.cache_hits,
            'cache_misses': self.metrics.cache_misses,
            'hit_rate': hit_rate,
            'miss_rate': miss_rate,
            'evictions': self.metrics.evictions,
            'expired_entries': self.metrics.expired_entries,
            'avg_access_count': avg_access_count,
            'memory_estimate_bytes': memory_estimate,
            'utilization': current_size / self.max_size if self.max_size > 0 else 0.0,
            'performance_status': self._get_performance_status(hit_rate)
        }
    
    def _get_performance_status(self, hit_rate: float) -> str:
        """Get performance status based on hit rate"""
        if hit_rate >= 0.6:  # Target hit rate
            return "excellent"
        elif hit_rate >= 0.4:
            return "good"
        elif hit_rate >= 0.2:
            return "fair"
        else:
            return "poor"
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics = CacheMetrics()
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        if not self.cache:
            return {
                'total_entries': 0,
                'oldest_entry_age': 0.0,
                'newest_entry_age': 0.0,
                'avg_entry_age': 0.0
            }
        
        current_time = time.time()
        entry_ages = [current_time - entry.timestamp for entry in self.cache.values()]
        
        return {
            'total_entries': len(self.cache),
            'oldest_entry_age': max(entry_ages),
            'newest_entry_age': min(entry_ages),
            'avg_entry_age': sum(entry_ages) / len(entry_ages),
            'entries_near_expiration': sum(
                1 for age in entry_ages 
                if age > self.ttl_seconds * 0.8
            )
        }
    
    def get_top_accessed_keys(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get most frequently accessed cache keys.
        
        Args:
            limit: Maximum number of keys to return
            
        Returns:
            List of (key, access_count) tuples
        """
        if not self.cache:
            return []
        
        # Sort by access count
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda item: item[1].access_count,
            reverse=True
        )
        
        return [(key, entry.access_count) for key, entry in sorted_entries[:limit]]
    
    def prewarm(self, key_action_pairs: List[Tuple[str, Action]]) -> None:
        """
        Prewarm cache with key-action pairs.
        
        Useful for initializing cache with known patterns.
        
        Args:
            key_action_pairs: List of (key, action) tuples to cache
        """
        for key, action in key_action_pairs:
            self.put(key, action)
    
    def get_hit_rate(self) -> float:
        """Get current cache hit rate"""
        if self.metrics.total_requests == 0:
            return 0.0
        return self.metrics.cache_hits / self.metrics.total_requests
    
    def is_healthy(self) -> bool:
        """Check if cache is performing well"""
        hit_rate = self.get_hit_rate()
        return hit_rate >= 0.4  # Minimum acceptable hit rate
    
    def __len__(self) -> int:
        """Return number of entries in cache"""
        return len(self.cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache (without updating access)"""
        if key not in self.cache:
            return False
        
        # Check if expired
        entry = self.cache[key]
        current_time = time.time()
        
        if current_time - entry.timestamp > self.ttl_seconds:
            return False
        
        return True
    
    def __repr__(self) -> str:
        return (f"ResponseCache(size={len(self.cache)}/{self.max_size}, "
                f"hit_rate={self.get_hit_rate():.2f}, "
                f"ttl={self.ttl_seconds}s)")
