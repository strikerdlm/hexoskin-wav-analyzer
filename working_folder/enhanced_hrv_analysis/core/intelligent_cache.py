"""
Intelligent Caching System for HRV Analysis

This module provides comprehensive caching capabilities for HRV analysis results,
featuring smart cache invalidation, persistent storage, and memory management.

Author: Dr. Diego Malpica - Aerospace Medicine Specialist
Organization: DIMAE / FAC / Colombia
Project: Valquiria Crew Space Simulation HRV Analysis System
"""

import hashlib
import json
import pickle
import sqlite3
import threading
import time
import lzma
import gzip
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import psutil
import os

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Container for cache entry metadata."""
    key: str
    data_hash: str
    created_at: datetime
    last_accessed: datetime
    access_count: int
    data_size: int
    ttl_hours: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    compression_ratio: float = 1.0
    access_frequency: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_hours is None:
            return False
        expiry_time = self.created_at + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry_time

    def update_frequency(self, window_hours: float = 24.0):
        """Update access frequency based on recent access patterns."""
        hours_since_created = (datetime.now() - self.created_at).total_seconds() / 3600
        if hours_since_created > 0:
            self.access_frequency = self.access_count / min(hours_since_created, window_hours)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create instance from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)

@dataclass
class CacheAnalytics:
    """Advanced cache analytics and performance metrics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    memory_usage_history: deque = None
    access_patterns: Dict[str, int] = None
    compression_savings_mb: float = 0.0
    prefetch_requests: int = 0
    prefetch_hits: int = 0
    average_load_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.memory_usage_history is None:
            self.memory_usage_history = deque(maxlen=100)  # Keep last 100 measurements
        if self.access_patterns is None:
            self.access_patterns = defaultdict(int)
    
    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def prefetch_efficiency(self) -> float:
        """Calculate prefetch efficiency."""
        return self.prefetch_hits / self.prefetch_requests if self.prefetch_requests > 0 else 0.0

class HRVResultsCache:
    """
    Advanced intelligent caching system for HRV analysis results.
    
    Enhanced Features:
    - Content-based cache keys with data fingerprinting
    - Smart compression with multiple algorithms
    - Predictive prefetching based on access patterns
    - Advanced analytics and performance monitoring
    - Memory-adaptive management with real-time monitoring
    - Cache warming strategies for common patterns
    """
    
    def __init__(self, 
                 cache_dir: Union[str, Path] = "hrv_cache",
                 max_memory_mb: int = 500,
                 max_entries: int = 1000,
                 default_ttl_hours: float = 24.0,
                 compression_level: int = 6,
                 enable_compression: bool = True,
                 enable_prefetch: bool = True,
                 analytics_window_hours: float = 24.0):
        """
        Initialize the advanced intelligent cache.
        
        Args:
            cache_dir: Directory for persistent cache storage
            max_memory_mb: Maximum memory usage in MB
            max_entries: Maximum number of cached entries
            default_ttl_hours: Default time-to-live in hours
            compression_level: Compression level (0-9, higher = better compression)
            enable_compression: Enable intelligent compression
            enable_prefetch: Enable predictive prefetching
            analytics_window_hours: Window for analytics calculations
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_entries = max_entries
        self.default_ttl_hours = default_ttl_hours
        self.compression_level = compression_level
        self.enable_compression = enable_compression
        self.enable_prefetch = enable_prefetch
        self.analytics_window_hours = analytics_window_hours
        
        # In-memory cache
        self._memory_cache: Dict[str, Any] = {}
        self._cache_metadata: Dict[str, CacheEntry] = {}
        self._current_memory_usage = 0
        
        # Advanced features
        self._analytics = CacheAnalytics()
        self._access_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._prefetch_candidates: Set[str] = set()
        self._compression_stats = {'lzma': 0, 'gzip': 0, 'pickle': 0}
        
        # Thread safety
        self._cache_lock = threading.RLock()
        self._analytics_lock = threading.Lock()
        
        # Performance monitoring thread
        self._monitoring_active = False
        self._monitor_thread = None
        
        # Persistent storage
        self.db_path = self.cache_dir / "cache_metadata.db"
        self._init_database()
        
        # Load existing cache metadata
        self._load_metadata()
        
        # Start performance monitoring
        self._start_performance_monitoring()
        
        logger.info(f"Advanced HRV Cache initialized - Max: {max_memory_mb}MB, {max_entries} entries, "
                   f"Compression: {'ON' if enable_compression else 'OFF'}, "
                   f"Prefetch: {'ON' if enable_prefetch else 'OFF'}")
        
    def _init_database(self):
        """Initialize SQLite database for cache metadata."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_metadata (
                        key TEXT PRIMARY KEY,
                        data_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        last_accessed TEXT NOT NULL,
                        access_count INTEGER NOT NULL,
                        data_size INTEGER NOT NULL,
                        ttl_hours REAL,
                        metadata TEXT
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_last_accessed 
                    ON cache_metadata(last_accessed)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_data_hash 
                    ON cache_metadata(data_hash)
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing cache database: {e}")
            
    def _load_metadata(self):
        """Load cache metadata from database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT * FROM cache_metadata")
                for row in cursor.fetchall():
                    key, data_hash, created_at, last_accessed, access_count, data_size, ttl_hours, metadata_json = row
                    
                    metadata = None
                    if metadata_json:
                        try:
                            metadata = json.loads(metadata_json)
                        except json.JSONDecodeError:
                            pass
                    
                    entry = CacheEntry(
                        key=key,
                        data_hash=data_hash,
                        created_at=datetime.fromisoformat(created_at),
                        last_accessed=datetime.fromisoformat(last_accessed),
                        access_count=access_count,
                        data_size=data_size,
                        ttl_hours=ttl_hours,
                        metadata=metadata
                    )
                    
                    # Check if entry is expired or file is missing
                    cache_file = self.cache_dir / f"{key}.cache"
                    if entry.is_expired() or not cache_file.exists():
                        self._remove_entry(key, from_db=True)
                    else:
                        self._cache_metadata[key] = entry
                        
            logger.info(f"Loaded {len(self._cache_metadata)} cache entries from database")
            
        except Exception as e:
            logger.error(f"Error loading cache metadata: {e}")
            
    def _generate_cache_key(self, 
                          subject_id: str, 
                          session_id: str,
                          data_fingerprint: str,
                          analysis_config: Dict[str, Any]) -> str:
        """
        Generate a unique cache key based on inputs.
        
        Args:
            subject_id: Subject identifier
            session_id: Session identifier  
            data_fingerprint: Hash of input data
            analysis_config: Analysis configuration parameters
            
        Returns:
            Unique cache key string
        """
        # Create deterministic hash of all parameters
        key_data = {
            'subject_id': subject_id,
            'session_id': session_id,
            'data_fingerprint': data_fingerprint,
            'analysis_config': analysis_config
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"{subject_id}_{session_id}_{key_hash}"
    
    def _compute_data_fingerprint(self, data: pd.DataFrame) -> str:
        """
        Compute fingerprint of input data for change detection.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Data fingerprint hash
        """
        try:
            # Create hash based on data shape, columns, and sample of values
            fingerprint_data = {
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
            }
            
            # Add sample of data values for content verification
            if not data.empty:
                # Sample first, middle, and last rows
                sample_indices = [0]
                if len(data) > 2:
                    sample_indices.extend([len(data) // 2, len(data) - 1])
                
                sample_data = data.iloc[sample_indices]
                fingerprint_data['sample_hash'] = hashlib.md5(
                    pd.util.hash_pandas_object(sample_data).values.tobytes()
                ).hexdigest()
            
            fingerprint_string = json.dumps(fingerprint_data, sort_keys=True)
            return hashlib.sha256(fingerprint_string.encode()).hexdigest()
            
        except Exception as e:
            logger.warning(f"Error computing data fingerprint: {e}")
            # Fallback to simple hash
            return hashlib.sha256(str(data.shape).encode()).hexdigest()
    
    def _start_performance_monitoring(self):
        """Start background performance monitoring."""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
            self._monitor_thread.start()
            logger.debug("Cache performance monitoring started")

    def _monitor_performance(self):
        """Background thread for performance monitoring."""
        while self._monitoring_active:
            try:
                with self._analytics_lock:
                    # Track memory usage
                    memory_usage_mb = self._current_memory_usage / (1024 * 1024)
                    self._analytics.memory_usage_history.append({
                        'timestamp': datetime.now(),
                        'memory_mb': memory_usage_mb,
                        'system_memory_percent': psutil.virtual_memory().percent
                    })
                    
                    # Update access frequencies for all entries
                    for entry in self._cache_metadata.values():
                        entry.update_frequency(self.analytics_window_hours)
                    
                    # Identify prefetch candidates based on patterns
                    if self.enable_prefetch:
                        self._identify_prefetch_candidates()
                
                # Sleep for monitoring interval
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.warning(f"Error in performance monitoring: {e}")
                time.sleep(10)

    def _identify_prefetch_candidates(self):
        """Identify potential prefetch candidates based on access patterns."""
        try:
            current_time = datetime.now()
            candidates = set()
            
            # Analyze access patterns for each key
            for key, access_times in self._access_patterns.items():
                if len(access_times) >= 3:  # Need at least 3 accesses for pattern
                    # Calculate average time between accesses
                    intervals = []
                    for i in range(1, len(access_times)):
                        interval = (access_times[i] - access_times[i-1]).total_seconds()
                        intervals.append(interval)
                    
                    if intervals:
                        avg_interval = sum(intervals) / len(intervals)
                        last_access = access_times[-1]
                        
                        # If we're approaching the next expected access time
                        time_since_last = (current_time - last_access).total_seconds()
                        if time_since_last > (avg_interval * 0.7):  # 70% of expected interval
                            candidates.add(key)
            
            self._prefetch_candidates = candidates
            logger.debug(f"Identified {len(candidates)} prefetch candidates")
            
        except Exception as e:
            logger.warning(f"Error identifying prefetch candidates: {e}")

    def _intelligent_compression(self, data: Any) -> Tuple[bytes, str, float]:
        """
        Apply intelligent compression based on data characteristics.
        
        Args:
            data: Data to compress
            
        Returns:
            Tuple of (compressed_data, compression_method, compression_ratio)
        """
        if not self.enable_compression:
            pickled_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            return pickled_data, 'pickle', 1.0
        
        # First pickle the data
        pickled_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        original_size = len(pickled_data)
        
        # Try different compression methods
        compression_results = {}
        
        # Try LZMA (best compression)
        try:
            lzma_compressed = lzma.compress(pickled_data, preset=self.compression_level)
            compression_results['lzma'] = (lzma_compressed, len(lzma_compressed))
        except Exception as e:
            logger.debug(f"LZMA compression failed: {e}")
        
        # Try GZIP (good balance)
        try:
            gzip_compressed = gzip.compress(pickled_data, compresslevel=self.compression_level)
            compression_results['gzip'] = (gzip_compressed, len(gzip_compressed))
        except Exception as e:
            logger.debug(f"GZIP compression failed: {e}")
        
        # Choose best compression method
        if compression_results:
            best_method = min(compression_results.keys(), key=lambda k: compression_results[k][1])
            compressed_data, compressed_size = compression_results[best_method]
            compression_ratio = original_size / compressed_size
            
            # Update compression stats
            self._compression_stats[best_method] += 1
            
            # Only use compression if it provides significant benefit
            if compression_ratio > 1.1:  # At least 10% savings
                with self._analytics_lock:
                    self._analytics.compression_savings_mb += (original_size - compressed_size) / (1024**2)
                return compressed_data, best_method, compression_ratio
        
        # Fallback to uncompressed pickle
        self._compression_stats['pickle'] += 1
        return pickled_data, 'pickle', 1.0

    def _decompress_data(self, compressed_data: bytes, compression_method: str) -> Any:
        """
        Decompress cached data.
        
        Args:
            compressed_data: Compressed data bytes
            compression_method: Compression method used
            
        Returns:
            Decompressed original data
        """
        try:
            if compression_method == 'lzma':
                decompressed = lzma.decompress(compressed_data)
            elif compression_method == 'gzip':
                decompressed = gzip.decompress(compressed_data)
            elif compression_method == 'pickle':
                decompressed = compressed_data
            else:
                logger.warning(f"Unknown compression method: {compression_method}")
                decompressed = compressed_data
            
            return pickle.loads(decompressed)
            
        except Exception as e:
            logger.error(f"Error decompressing data: {e}")
            raise

    def get(self, 
            subject_id: str,
            session_id: str, 
            data: pd.DataFrame,
            analysis_config: Dict[str, Any]) -> Optional[Any]:
        """
        Retrieve cached HRV analysis result with advanced analytics.
        
        Args:
            subject_id: Subject identifier
            session_id: Session identifier
            data: Input data for fingerprint generation
            analysis_config: Analysis configuration
            
        Returns:
            Cached result if available and valid, None otherwise
        """
        with self._cache_lock:
            start_time = time.time()
            
            try:
                # Update analytics
                with self._analytics_lock:
                    self._analytics.total_requests += 1
                
                # Generate cache key
                data_fingerprint = self._compute_data_fingerprint(data)
                cache_key = self._generate_cache_key(
                    subject_id, session_id, data_fingerprint, analysis_config
                )
                
                # Track access pattern
                self._access_patterns[cache_key].append(datetime.now())
                
                # Check if key exists and is valid
                if cache_key not in self._cache_metadata:
                    with self._analytics_lock:
                        self._analytics.cache_misses += 1
                    return None
                
                entry = self._cache_metadata[cache_key]
                
                # Check expiration
                if entry.is_expired():
                    self._remove_entry(cache_key)
                    with self._analytics_lock:
                        self._analytics.cache_misses += 1
                    return None
                
                # Check data fingerprint (data changed)
                if entry.data_hash != data_fingerprint:
                    self._remove_entry(cache_key)
                    with self._analytics_lock:
                        self._analytics.cache_misses += 1
                    return None
                
                # Try to load from memory cache first
                if cache_key in self._memory_cache:
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    self._update_metadata_db(entry)
                    
                    with self._analytics_lock:
                        self._analytics.cache_hits += 1
                        load_time_ms = (time.time() - start_time) * 1000
                        self._analytics.average_load_time_ms = (
                            (self._analytics.average_load_time_ms * (self._analytics.cache_hits - 1) + load_time_ms) 
                            / self._analytics.cache_hits
                        )
                    
                    logger.debug(f"Cache HIT (memory): {cache_key}")
                    return self._memory_cache[cache_key]
                
                # Load from persistent storage
                cache_file = self.cache_dir / f"{cache_key}.cache"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            compressed_data = f.read()
                        
                        # Get compression method from metadata
                        compression_method = entry.metadata.get('compression_method', 'pickle') if entry.metadata else 'pickle'
                        result = self._decompress_data(compressed_data, compression_method)
                        
                        # Add to memory cache if there's space
                        if self._current_memory_usage + entry.data_size <= self.max_memory_bytes:
                            self._memory_cache[cache_key] = result
                            self._current_memory_usage += entry.data_size
                        
                        entry.last_accessed = datetime.now()
                        entry.access_count += 1
                        self._update_metadata_db(entry)
                        
                        with self._analytics_lock:
                            self._analytics.cache_hits += 1
                            load_time_ms = (time.time() - start_time) * 1000
                            self._analytics.average_load_time_ms = (
                                (self._analytics.average_load_time_ms * (self._analytics.cache_hits - 1) + load_time_ms) 
                                / self._analytics.cache_hits
                            )
                        
                        logger.debug(f"Cache HIT (disk): {cache_key}, compression: {compression_method}")
                        return result
                        
                    except Exception as e:
                        logger.warning(f"Error loading cached data: {e}")
                        self._remove_entry(cache_key)
                        with self._analytics_lock:
                            self._analytics.cache_misses += 1
                        return None
                else:
                    # File missing, remove metadata
                    self._remove_entry(cache_key)
                    with self._analytics_lock:
                        self._analytics.cache_misses += 1
                    return None
                    
            except Exception as e:
                logger.error(f"Error retrieving from cache: {e}")
                with self._analytics_lock:
                    self._analytics.cache_misses += 1
                return None
    
    def put(self,
            subject_id: str,
            session_id: str,
            data: pd.DataFrame,
            analysis_config: Dict[str, Any],
            result: Any,
            ttl_hours: Optional[float] = None,
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store HRV analysis result in cache with intelligent compression.
        
        Args:
            subject_id: Subject identifier
            session_id: Session identifier
            data: Input data for fingerprint generation
            analysis_config: Analysis configuration
            result: Analysis result to cache
            ttl_hours: Time-to-live in hours
            metadata: Additional metadata
            
        Returns:
            True if successfully cached, False otherwise
        """
        with self._cache_lock:
            try:
                # Generate cache key
                data_fingerprint = self._compute_data_fingerprint(data)
                cache_key = self._generate_cache_key(
                    subject_id, session_id, data_fingerprint, analysis_config
                )
                
                # Apply intelligent compression
                compressed_data, compression_method, compression_ratio = self._intelligent_compression(result)
                data_size = len(compressed_data)
                
                # Check if we need to make space
                self._ensure_capacity(data_size)
                
                # Prepare metadata
                entry_metadata = metadata or {}
                entry_metadata.update({
                    'compression_method': compression_method,
                    'compression_ratio': compression_ratio,
                    'original_data_shape': getattr(data, 'shape', None),
                    'analysis_config_hash': hashlib.md5(
                        json.dumps(analysis_config, sort_keys=True).encode()
                    ).hexdigest()
                })
                
                # Create cache entry
                entry = CacheEntry(
                    key=cache_key,
                    data_hash=data_fingerprint,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    data_size=data_size,
                    ttl_hours=ttl_hours or self.default_ttl_hours,
                    metadata=entry_metadata,
                    compression_ratio=compression_ratio
                )
                
                # Save to persistent storage
                cache_file = self.cache_dir / f"{cache_key}.cache"
                try:
                    with open(cache_file, 'wb') as f:
                        f.write(compressed_data)
                except Exception as e:
                    logger.error(f"Error saving to cache file: {e}")
                    return False
                
                # Add to memory cache if space available
                if self._current_memory_usage + data_size <= self.max_memory_bytes:
                    self._memory_cache[cache_key] = result
                    self._current_memory_usage += data_size
                
                # Update metadata
                self._cache_metadata[cache_key] = entry
                self._update_metadata_db(entry)
                
                logger.info(f"Cached result for {cache_key} (size: {data_size:,} bytes, "
                           f"compression: {compression_method}, ratio: {compression_ratio:.2f}x)")
                return True
                
            except Exception as e:
                logger.error(f"Error storing in cache: {e}")
                return False
    
    def _ensure_capacity(self, required_size: int):
        """Ensure cache has capacity for new entry."""
        # Remove expired entries first
        self._cleanup_expired()
        
        # Check entry count limit
        while len(self._cache_metadata) >= self.max_entries:
            self._evict_lru_entry()
        
        # Check memory limit
        while (self._current_memory_usage + required_size > self.max_memory_bytes and 
               self._current_memory_usage > 0):
            self._evict_lru_memory_entry()
    
    def _cleanup_expired(self):
        """Remove expired cache entries."""
        expired_keys = [
            key for key, entry in self._cache_metadata.items() 
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
        
        if expired_keys:
            logger.info(f"Removed {len(expired_keys)} expired cache entries")
    
    def _evict_lru_entry(self):
        """Evict least recently used entry."""
        if not self._cache_metadata:
            return
        
        # Find LRU entry
        lru_key = min(
            self._cache_metadata.keys(),
            key=lambda k: self._cache_metadata[k].last_accessed
        )
        
        self._remove_entry(lru_key)
        logger.debug(f"Evicted LRU entry: {lru_key}")
    
    def _evict_lru_memory_entry(self):
        """Evict least recently used entry from memory cache."""
        if not self._memory_cache:
            return
        
        # Find LRU entry in memory
        memory_keys = set(self._memory_cache.keys())
        lru_key = min(
            memory_keys,
            key=lambda k: self._cache_metadata[k].last_accessed
        )
        
        # Remove from memory only
        if lru_key in self._memory_cache:
            entry = self._cache_metadata[lru_key]
            self._current_memory_usage -= entry.data_size
            del self._memory_cache[lru_key]
            logger.debug(f"Evicted from memory: {lru_key}")
    
    def _remove_entry(self, key: str, from_db: bool = False):
        """Remove cache entry completely."""
        try:
            # Remove from memory
            if key in self._memory_cache:
                if key in self._cache_metadata:
                    self._current_memory_usage -= self._cache_metadata[key].data_size
                del self._memory_cache[key]
            
            # Remove metadata
            if key in self._cache_metadata:
                del self._cache_metadata[key]
            
            # Remove from database
            if not from_db:
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute("DELETE FROM cache_metadata WHERE key = ?", (key,))
                    conn.commit()
            
            # Remove cache file
            cache_file = self.cache_dir / f"{key}.cache"
            if cache_file.exists():
                cache_file.unlink()
                
        except Exception as e:
            logger.error(f"Error removing cache entry {key}: {e}")
    
    def _update_metadata_db(self, entry: CacheEntry):
        """Update metadata in database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                metadata_json = None
                if entry.metadata:
                    metadata_json = json.dumps(entry.metadata)
                
                conn.execute("""
                    INSERT OR REPLACE INTO cache_metadata 
                    (key, data_hash, created_at, last_accessed, access_count, 
                     data_size, ttl_hours, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.key,
                    entry.data_hash,
                    entry.created_at.isoformat(),
                    entry.last_accessed.isoformat(),
                    entry.access_count,
                    entry.data_size,
                    entry.ttl_hours,
                    metadata_json
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating metadata in database: {e}")
    
    def clear_cache(self):
        """Clear all cached data."""
        with self._cache_lock:
            try:
                # Clear memory cache
                self._memory_cache.clear()
                self._cache_metadata.clear()
                self._current_memory_usage = 0
                
                # Clear database
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute("DELETE FROM cache_metadata")
                    conn.commit()
                
                # Remove cache files
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
                
                logger.info("Cache cleared completely")
                
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics with advanced analytics."""
        with self._cache_lock:
            with self._analytics_lock:
                return {
                    'total_entries': len(self._cache_metadata),
                    'memory_entries': len(self._memory_cache),
                    'memory_usage_mb': self._current_memory_usage / (1024 * 1024),
                    'memory_limit_mb': self.max_memory_bytes / (1024 * 1024),
                    'entry_limit': self.max_entries,
                    'cache_dir': str(self.cache_dir),
                    'hit_ratio': self._analytics.hit_ratio,
                    'total_requests': self._analytics.total_requests,
                    'cache_hits': self._analytics.cache_hits,
                    'cache_misses': self._analytics.cache_misses,
                    'evictions': self._analytics.evictions,
                    'compression_savings_mb': self._analytics.compression_savings_mb,
                    'average_load_time_ms': self._analytics.average_load_time_ms,
                    'compression_stats': dict(self._compression_stats),
                    'prefetch_candidates': len(self._prefetch_candidates),
                    'prefetch_efficiency': self._analytics.prefetch_efficiency,
                    'access_patterns_tracked': len(self._access_patterns),
                    'memory_usage_trend': list(self._analytics.memory_usage_history)[-10:] if self._analytics.memory_usage_history else []
                }

    def warm_cache_for_subjects(self, subjects: List[str], common_configs: List[Dict[str, Any]]):
        """
        Warm cache with commonly accessed subject/configuration combinations.
        
        Args:
            subjects: List of subject IDs to warm cache for
            common_configs: List of common analysis configurations
        """
        logger.info(f"Starting cache warming for {len(subjects)} subjects with {len(common_configs)} configurations")
        
        # This would typically be called with actual data and analysis functions
        # For now, we'll just mark these as prefetch candidates
        for subject in subjects:
            for config in common_configs:
                # Create a simplified cache key for tracking
                key_data = {'subject_id': subject, 'analysis_config': config}
                key_string = json.dumps(key_data, sort_keys=True)
                key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
                cache_key = f"{subject}_warming_{key_hash}"
                
                self._prefetch_candidates.add(cache_key)
        
        logger.info(f"Cache warming completed - {len(self._prefetch_candidates)} candidates identified")

    def get_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        stats = self.get_cache_stats()
        
        report = f"""
=== HRV Cache Performance Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 CACHE EFFICIENCY:
• Hit Ratio: {stats['hit_ratio']:.1%}
• Total Requests: {stats['total_requests']:,}
• Cache Hits: {stats['cache_hits']:,}
• Cache Misses: {stats['cache_misses']:,}
• Average Load Time: {stats['average_load_time_ms']:.1f}ms

💾 MEMORY USAGE:
• Current Usage: {stats['memory_usage_mb']:.1f}MB / {stats['memory_limit_mb']:.1f}MB
• Total Entries: {stats['total_entries']:,}
• Memory Entries: {stats['memory_entries']:,}
• Evictions: {stats['evictions']:,}

🗜️ COMPRESSION:
• Total Savings: {stats['compression_savings_mb']:.1f}MB
• LZMA: {stats['compression_stats'].get('lzma', 0)} files
• GZIP: {stats['compression_stats'].get('gzip', 0)} files
• Uncompressed: {stats['compression_stats'].get('pickle', 0)} files

🚀 PREDICTIVE FEATURES:
• Prefetch Candidates: {stats['prefetch_candidates']}
• Prefetch Efficiency: {stats['prefetch_efficiency']:.1%}
• Access Patterns Tracked: {stats['access_patterns_tracked']}

📈 PERFORMANCE TRENDS:
• Recent Memory Usage: {[f"{m['memory_mb']:.1f}MB" for m in stats['memory_usage_trend'][-5:]]}
"""
        return report

    def stop_monitoring(self):
        """Stop background performance monitoring."""
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        logger.info("Cache performance monitoring stopped")

    def __del__(self):
        """Cleanup when cache is destroyed."""
        self.stop_monitoring() 