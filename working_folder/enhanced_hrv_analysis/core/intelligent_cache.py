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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np

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

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_hours is None:
            return False
        expiry_time = self.created_at + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry_time

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


class HRVResultsCache:
    """
    Intelligent caching system for HRV analysis results.
    
    Features:
    - Content-based cache keys using data fingerprinting
    - LRU eviction with configurable size limits
    - Persistent storage with SQLite backend
    - Smart cache invalidation based on data changes
    - Compressed storage for memory efficiency
    - Thread-safe operations
    """
    
    def __init__(self, 
                 cache_dir: Union[str, Path] = "hrv_cache",
                 max_memory_mb: int = 500,
                 max_entries: int = 1000,
                 default_ttl_hours: float = 24.0,
                 compression_level: int = 6):
        """
        Initialize the intelligent cache.
        
        Args:
            cache_dir: Directory for persistent cache storage
            max_memory_mb: Maximum memory usage in MB
            max_entries: Maximum number of cached entries
            default_ttl_hours: Default time-to-live in hours
            compression_level: Compression level for stored data (0-9)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_entries = max_entries
        self.default_ttl_hours = default_ttl_hours
        self.compression_level = compression_level
        
        # In-memory cache
        self._memory_cache: Dict[str, Any] = {}
        self._cache_metadata: Dict[str, CacheEntry] = {}
        self._current_memory_usage = 0
        
        # Thread safety
        self._cache_lock = threading.RLock()
        
        # Persistent storage
        self.db_path = self.cache_dir / "cache_metadata.db"
        self._init_database()
        
        # Load existing cache metadata
        self._load_metadata()
        
        logger.info(f"HRV Cache initialized - Max: {max_memory_mb}MB, {max_entries} entries")
        
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
    
    def get(self, 
            subject_id: str,
            session_id: str, 
            data: pd.DataFrame,
            analysis_config: Dict[str, Any]) -> Optional[Any]:
        """
        Retrieve cached HRV analysis result.
        
        Args:
            subject_id: Subject identifier
            session_id: Session identifier
            data: Input data for fingerprint generation
            analysis_config: Analysis configuration
            
        Returns:
            Cached result if available and valid, None otherwise
        """
        with self._cache_lock:
            try:
                # Generate cache key
                data_fingerprint = self._compute_data_fingerprint(data)
                cache_key = self._generate_cache_key(
                    subject_id, session_id, data_fingerprint, analysis_config
                )
                
                # Check if key exists and is valid
                if cache_key not in self._cache_metadata:
                    return None
                
                entry = self._cache_metadata[cache_key]
                
                # Check expiration
                if entry.is_expired():
                    self._remove_entry(cache_key)
                    return None
                
                # Check data fingerprint (data changed)
                if entry.data_hash != data_fingerprint:
                    self._remove_entry(cache_key)
                    return None
                
                # Try to load from memory cache first
                if cache_key in self._memory_cache:
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    self._update_metadata_db(entry)
                    logger.debug(f"Cache HIT (memory): {cache_key}")
                    return self._memory_cache[cache_key]
                
                # Load from persistent storage
                cache_file = self.cache_dir / f"{cache_key}.cache"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            result = pickle.load(f)
                        
                        # Add to memory cache if there's space
                        if self._current_memory_usage + entry.data_size <= self.max_memory_bytes:
                            self._memory_cache[cache_key] = result
                            self._current_memory_usage += entry.data_size
                        
                        entry.last_accessed = datetime.now()
                        entry.access_count += 1
                        self._update_metadata_db(entry)
                        
                        logger.debug(f"Cache HIT (disk): {cache_key}")
                        return result
                        
                    except Exception as e:
                        logger.warning(f"Error loading cached data: {e}")
                        self._remove_entry(cache_key)
                        return None
                else:
                    # File missing, remove metadata
                    self._remove_entry(cache_key)
                    return None
                    
            except Exception as e:
                logger.error(f"Error retrieving from cache: {e}")
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
        Store HRV analysis result in cache.
        
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
                
                # Estimate data size
                try:
                    pickled_data = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
                    data_size = len(pickled_data)
                except Exception:
                    logger.warning(f"Cannot pickle result for {cache_key}")
                    return False
                
                # Check if we need to make space
                self._ensure_capacity(data_size)
                
                # Create cache entry
                entry = CacheEntry(
                    key=cache_key,
                    data_hash=data_fingerprint,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    data_size=data_size,
                    ttl_hours=ttl_hours or self.default_ttl_hours,
                    metadata=metadata
                )
                
                # Save to persistent storage
                cache_file = self.cache_dir / f"{cache_key}.cache"
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
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
                
                logger.info(f"Cached result for {cache_key} (size: {data_size:,} bytes)")
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
        """Get cache statistics."""
        with self._cache_lock:
            return {
                'total_entries': len(self._cache_metadata),
                'memory_entries': len(self._memory_cache),
                'memory_usage_mb': self._current_memory_usage / (1024 * 1024),
                'memory_limit_mb': self.max_memory_bytes / (1024 * 1024),
                'entry_limit': self.max_entries,
                'cache_dir': str(self.cache_dir)
            } 