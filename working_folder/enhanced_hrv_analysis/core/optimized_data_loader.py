"""
Optimized Database Loader for Large HRV Datasets

This module provides high-performance data loading capabilities specifically optimized
for handling large datasets (1.5M+ records) with minimal memory footprint and 
efficient query processing.

Author: Dr. Diego Malpica - Aerospace Medicine Specialist
Organization: DIMAE / FAC / Colombia
Project: Valquiria Crew Space Simulation HRV Analysis System
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
import threading
import time
from contextlib import contextmanager
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

@dataclass
class DatasetInfo:
    """Information about the loaded dataset."""
    total_records: int
    subjects: List[str]
    sols: List[int]
    date_range: Tuple[str, str]
    memory_usage_mb: float
    load_time_seconds: float

@dataclass
class ChunkInfo:
    """Information about a data chunk."""
    chunk_id: int
    subject: str
    sol: Optional[int]
    record_count: int
    date_range: Tuple[str, str]
    memory_mb: float

class OptimizedDataLoader:
    """
    High-performance data loader optimized for large HRV datasets.
    
    Features:
    - Chunked loading to minimize memory usage
    - Intelligent query optimization with indexes
    - Subject/SOL-based partitioning
    - Memory-efficient data processing
    - Parallel loading for multiple subjects
    - Progress tracking and cancellation support
    """
    
    def __init__(self, 
                 chunk_size: int = 50000,
                 max_memory_mb: float = 1000.0,
                 enable_parallel: bool = True,
                 max_workers: int = 4):
        """
        Initialize the optimized data loader.
        
        Args:
            chunk_size: Number of records to load per chunk
            max_memory_mb: Maximum memory usage in MB
            enable_parallel: Enable parallel chunk processing
            max_workers: Maximum parallel workers
        """
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        
        # Performance tracking
        self.load_stats = {
            'chunks_loaded': 0,
            'total_records': 0,
            'load_time': 0.0,
            'memory_peak_mb': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._cancel_flag = threading.Event()
        
        logger.info(f"OptimizedDataLoader initialized: chunk_size={chunk_size:,}, max_memory={max_memory_mb}MB")
        
    @contextmanager
    def _get_db_connection(self, db_path: str, read_only: bool = True):
        """
        Context manager for database connections with optimization.
        
        Args:
            db_path: Path to database file
            read_only: Whether to open in read-only mode
        """
        conn = None
        try:
            # Open connection with optimizations
            if read_only:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            else:
                conn = sqlite3.connect(db_path)
            
            # Performance optimizations for reading
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL") 
            conn.execute("PRAGMA cache_size=10000")  # 10MB cache
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
            
            yield conn
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _create_indexes_if_needed(self, db_path: str, table_name: str = "merged_data"):
        """
        Create performance indexes if they don't exist.
        
        Args:
            db_path: Path to database
            table_name: Name of the main data table
        """
        try:
            with self._get_db_connection(db_path, read_only=False) as conn:
                cursor = conn.cursor()
                
                # Check existing indexes
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
                existing_indexes = [row[0] for row in cursor.fetchall()]
                
                # Create indexes for common query patterns
                indexes_to_create = [
                    (f"idx_{table_name}_subject", f"CREATE INDEX IF NOT EXISTS idx_{table_name}_subject ON {table_name}(subject)"),
                    (f"idx_{table_name}_sol", f"CREATE INDEX IF NOT EXISTS idx_{table_name}_sol ON {table_name}(Sol)"),
                    (f"idx_{table_name}_subject_sol", f"CREATE INDEX IF NOT EXISTS idx_{table_name}_subject_sol ON {table_name}(subject, Sol)"),
                    (f"idx_{table_name}_time", f"CREATE INDEX IF NOT EXISTS idx_{table_name}_time ON {table_name}(\"time [s/1000]\")"),
                ]
                
                created_count = 0
                for index_name, create_sql in indexes_to_create:
                    if index_name not in existing_indexes:
                        logger.info(f"Creating index: {index_name}")
                        cursor.execute(create_sql)
                        created_count += 1
                
                if created_count > 0:
                    conn.commit()
                    logger.info(f"Created {created_count} performance indexes")
                else:
                    logger.info("All performance indexes already exist")
                    
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    def get_dataset_info(self, db_path: str, table_name: str = "merged_data") -> Optional[DatasetInfo]:
        """
        Get comprehensive information about the dataset without loading all data.
        
        Args:
            db_path: Path to database file
            table_name: Name of the data table
            
        Returns:
            DatasetInfo object with dataset statistics
        """
        try:
            start_time = time.time()
            
            with self._get_db_connection(db_path) as conn:
                # Get total record count
                count_query = f"SELECT COUNT(*) FROM {table_name}"
                total_records = conn.execute(count_query).fetchone()[0]
                
                # Get unique subjects
                subjects_query = f"SELECT DISTINCT subject FROM {table_name} ORDER BY subject"
                subjects = [row[0] for row in conn.execute(subjects_query).fetchall()]
                
                # Get unique SOLs
                sols_query = f"SELECT DISTINCT Sol FROM {table_name} WHERE Sol IS NOT NULL ORDER BY Sol"
                sols = [row[0] for row in conn.execute(sols_query).fetchall()]
                
                # Get date range if time column exists
                try:
                    date_query = f'SELECT MIN("time [s/1000]"), MAX("time [s/1000]") FROM {table_name} WHERE "time [s/1000]" IS NOT NULL'
                    date_result = conn.execute(date_query).fetchone()
                    date_range = (str(date_result[0]), str(date_result[1])) if date_result[0] else ("N/A", "N/A")
                except:
                    date_range = ("N/A", "N/A")
                
                # Estimate memory usage (rough approximation)
                memory_usage_mb = (total_records * 50) / (1024 * 1024)  # Assume ~50 bytes per record
                
                load_time = time.time() - start_time
                
                logger.info(f"Dataset info retrieved: {total_records:,} records, {len(subjects)} subjects, {len(sols)} SOLs")
                
                return DatasetInfo(
                    total_records=total_records,
                    subjects=subjects,
                    sols=sols,
                    date_range=date_range,
                    memory_usage_mb=memory_usage_mb,
                    load_time_seconds=load_time
                )
                
        except Exception as e:
            logger.error(f"Error getting dataset info: {e}")
            return None
    
    def load_subjects_chunked(self, 
                             db_path: str, 
                             subjects: Optional[List[str]] = None,
                             sols: Optional[List[int]] = None,
                             table_name: str = "merged_data",
                             progress_callback: Optional[callable] = None) -> Iterator[Tuple[str, pd.DataFrame]]:
        """
        Load data in chunks by subject/SOL combination for memory efficiency.
        
        Args:
            db_path: Path to database file
            subjects: List of subjects to load (None for all)
            sols: List of SOLs to load (None for all)
            table_name: Name of the data table
            progress_callback: Function to call with progress updates
            
        Yields:
            Tuple of (subject_key, dataframe_chunk) for each chunk
        """
        try:
            # Create indexes for performance
            self._create_indexes_if_needed(db_path, table_name)
            
            start_time = time.time()
            total_chunks = 0
            
            with self._get_db_connection(db_path) as conn:
                # Build WHERE clause
                where_conditions = []
                if subjects:
                    subjects_str = "', '".join(subjects)
                    where_conditions.append(f"subject IN ('{subjects_str}')")
                if sols:
                    sols_str = ", ".join(map(str, sols))
                    where_conditions.append(f"Sol IN ({sols_str})")
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                
                # Get subject/SOL combinations
                combo_query = f"""
                    SELECT subject, Sol, COUNT(*) as record_count
                    FROM {table_name} 
                    WHERE {where_clause}
                    GROUP BY subject, Sol 
                    ORDER BY subject, Sol
                """
                
                combinations = conn.execute(combo_query).fetchall()
                total_combinations = len(combinations)
                
                logger.info(f"Loading {total_combinations} subject/SOL combinations in chunks of {self.chunk_size:,}")
                
                for i, (subject, sol, record_count) in enumerate(combinations):
                    if self._cancel_flag.is_set():
                        logger.info("Loading cancelled by user")
                        break
                        
                    # Update progress
                    if progress_callback:
                        progress = (i / total_combinations) * 100
                        progress_callback(f"Loading {subject}_Sol{sol}", progress)
                    
                    # Load this subject/SOL combination in chunks
                    subject_key = f"{subject}_Sol{sol}" if sol is not None else subject
                    
                    # Calculate chunks needed for this combination
                    chunks_needed = (record_count + self.chunk_size - 1) // self.chunk_size
                    
                    for chunk_idx in range(chunks_needed):
                        offset = chunk_idx * self.chunk_size
                        
                        # Optimized query with LIMIT/OFFSET
                        chunk_query = f"""
                            SELECT * FROM {table_name}
                            WHERE subject = ? AND Sol = ?
                            ORDER BY "time [s/1000]"
                            LIMIT ? OFFSET ?
                        """
                        
                        try:
                            chunk_df = pd.read_sql_query(
                                chunk_query, 
                                conn, 
                                params=(subject, sol, self.chunk_size, offset)
                            )
                            
                            if not chunk_df.empty:
                                # Memory optimization: convert data types
                                chunk_df = self._optimize_datatypes(chunk_df)
                                
                                # Track memory usage
                                chunk_memory = chunk_df.memory_usage(deep=True).sum() / (1024**2)
                                self.load_stats['memory_peak_mb'] = max(
                                    self.load_stats['memory_peak_mb'], 
                                    chunk_memory
                                )
                                
                                # Update stats
                                self.load_stats['chunks_loaded'] += 1
                                self.load_stats['total_records'] += len(chunk_df)
                                total_chunks += 1
                                
                                yield subject_key, chunk_df
                                
                                # Force garbage collection if memory usage is high
                                if chunk_memory > self.max_memory_mb / 4:
                                    gc.collect()
                                    
                        except Exception as e:
                            logger.error(f"Error loading chunk for {subject_key}: {e}")
                            continue
                
                # Final stats
                self.load_stats['load_time'] = time.time() - start_time
                logger.info(f"Completed chunked loading: {total_chunks} chunks, {self.load_stats['total_records']:,} total records")
                
        except Exception as e:
            logger.error(f"Error in chunked loading: {e}")
            raise
    
    def load_subject_data(self,
                         db_path: str,
                         subject: str,
                         sols: Optional[List[int]] = None,
                         table_name: str = "merged_data") -> Optional[pd.DataFrame]:
        """
        Load all data for a specific subject efficiently.
        
        Args:
            db_path: Path to database file
            subject: Subject identifier
            sols: List of SOLs to load (None for all)
            table_name: Name of the data table
            
        Returns:
            Combined DataFrame for the subject
        """
        try:
            chunks = []
            total_records = 0
            
            for subject_key, chunk_df in self.load_subjects_chunked(
                db_path, 
                subjects=[subject], 
                sols=sols, 
                table_name=table_name
            ):
                chunks.append(chunk_df)
                total_records += len(chunk_df)
                
                # Memory safety check
                current_memory = sum(chunk.memory_usage(deep=True).sum() for chunk in chunks) / (1024**2)
                if current_memory > self.max_memory_mb:
                    logger.warning(f"Memory limit exceeded for {subject}, combining chunks early")
                    break
            
            if chunks:
                combined_df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Loaded {total_records:,} records for {subject}")
                return combined_df
            else:
                logger.warning(f"No data found for subject {subject}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading subject {subject}: {e}")
            return None
    
    def _optimize_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame data types for memory efficiency.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with optimized data types
        """
        try:
            optimized = df.copy()
            
            # Optimize numeric columns
            for col in optimized.select_dtypes(include=[np.number]).columns:
                col_data = optimized[col]
                
                # Skip if all NaN
                if col_data.isna().all():
                    continue
                
                # Integer optimization
                if col_data.dtype in ['int64', 'int32']:
                    min_val, max_val = col_data.min(), col_data.max()
                    if min_val >= 0:
                        if max_val <= 255:
                            optimized[col] = col_data.astype('uint8')
                        elif max_val <= 65535:
                            optimized[col] = col_data.astype('uint16')
                        elif max_val <= 4294967295:
                            optimized[col] = col_data.astype('uint32')
                    else:
                        if min_val >= -128 and max_val <= 127:
                            optimized[col] = col_data.astype('int8')
                        elif min_val >= -32768 and max_val <= 32767:
                            optimized[col] = col_data.astype('int16')
                
                # Float optimization
                elif col_data.dtype == 'float64':
                    if col_data.min() >= np.finfo(np.float32).min and col_data.max() <= np.finfo(np.float32).max:
                        optimized[col] = col_data.astype('float32')
            
            # String optimization
            for col in optimized.select_dtypes(include=['object']).columns:
                if optimized[col].dtype == object:
                    # Try to convert to category if many repeated values
                    unique_ratio = optimized[col].nunique() / len(optimized[col])
                    if unique_ratio < 0.5:  # Less than 50% unique values
                        optimized[col] = optimized[col].astype('category')
            
            # Log memory savings
            original_memory = df.memory_usage(deep=True).sum() / (1024**2)
            optimized_memory = optimized.memory_usage(deep=True).sum() / (1024**2)
            memory_savings = ((original_memory - optimized_memory) / original_memory) * 100
            
            if memory_savings > 5:  # Only log significant savings
                logger.debug(f"Memory optimization: {memory_savings:.1f}% reduction ({original_memory:.1f}MB → {optimized_memory:.1f}MB)")
            
            return optimized
            
        except Exception as e:
            logger.warning(f"Error optimizing data types: {e}")
            return df
    
    def cancel_loading(self):
        """Cancel ongoing loading operations."""
        self._cancel_flag.set()
        logger.info("Loading cancellation requested")
    
    def reset_cancel_flag(self):
        """Reset cancellation flag for new operations."""
        self._cancel_flag.clear()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from recent operations."""
        return {
            'chunks_loaded': self.load_stats['chunks_loaded'],
            'total_records_loaded': self.load_stats['total_records'],
            'load_time_seconds': self.load_stats['load_time'],
            'records_per_second': self.load_stats['total_records'] / self.load_stats['load_time'] if self.load_stats['load_time'] > 0 else 0,
            'memory_peak_mb': self.load_stats['memory_peak_mb'],
            'chunk_size': self.chunk_size,
            'max_memory_mb': self.max_memory_mb
        } 