"""
Advanced Asynchronous Processing Module for HRV Analysis

This module provides safe parallel processing capabilities without blocking the GUI,
featuring proper thread management, timeout handling, and progress tracking.

Author: Dr. Diego Malpica - Aerospace Medicine Specialist
Organization: DIMAE / FAC / Colombia
Project: Valquiria Crew Space Simulation HRV Analysis System
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from queue import Queue, Empty
from typing import Dict, Any, Optional, Callable, List, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import functools

logger = logging.getLogger(__name__)

class ProcessingState(Enum):
    """Processing task states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class ProcessingTask:
    """Container for processing task information."""
    task_id: str
    func: Callable
    args: tuple
    kwargs: dict
    state: ProcessingState = ProcessingState.PENDING
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    timeout_seconds: float = 300.0  # 5 minute default timeout

    @property
    def duration(self) -> Optional[float]:
        """Calculate task duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

class SafeAsyncProcessor:
    """
    Safe asynchronous processor with GUI non-blocking parallel execution.
    
    Features:
    - Thread-safe parallel processing with configurable workers
    - Timeout management for individual tasks
    - Progress tracking and status callbacks
    - Memory-efficient task queue management
    - Graceful shutdown and cleanup
    """
    
    def __init__(self, 
                 max_workers: int = 2, 
                 default_timeout: float = 300.0,
                 progress_callback: Optional[Callable[[str, float], None]] = None,
                 status_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the async processor.
        
        Args:
            max_workers: Maximum number of parallel worker threads
            default_timeout: Default timeout for tasks in seconds
            progress_callback: Function to call with progress updates (message, percentage)
            status_callback: Function to call with status updates
        """
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        
        # Threading components
        self._executor: Optional[ThreadPoolExecutor] = None
        self._task_queue: Queue = Queue()
        self._active_tasks: Dict[str, ProcessingTask] = {}
        self._completed_tasks: Dict[str, ProcessingTask] = {}
        
        # State management
        self._is_running = False
        self._shutdown_event = threading.Event()
        self._processing_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._total_tasks = 0
        self._completed_count = 0
        self._failed_count = 0
        
        # Cleanup settings
        self._max_completed_tasks = 50  # Maximum completed tasks to keep in memory
        self._cleanup_interval = 300  # Cleanup every 5 minutes
        self._last_cleanup = time.time()
        
        logger.info(f"SafeAsyncProcessor initialized with {max_workers} workers")
        
    def start(self):
        """Start the async processor."""
        if self._is_running:
            logger.warning("Processor is already running")
            return
            
        self._is_running = True
        self._shutdown_event.clear()
        
        # Initialize thread pool
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="HRV_Worker"
        )
        
        # Start processing thread
        self._processing_thread = threading.Thread(
            target=self._process_tasks,
            name="HRV_Processor",
            daemon=True
        )
        self._processing_thread.start()
        
        if self.status_callback:
            self.status_callback("Async processor started")
        
        logger.info("Async processor started successfully")
        
    def shutdown(self, wait_for_completion: bool = True, timeout: float = 30.0):
        """
        Gracefully shutdown the processor.
        
        Args:
            wait_for_completion: Whether to wait for current tasks to complete
            timeout: Maximum time to wait for shutdown
        """
        if not self._is_running:
            return
            
        logger.info("Shutting down async processor...")
        
        self._is_running = False
        self._shutdown_event.set()
        
        # Cancel pending tasks
        pending_tasks = []
        try:
            while True:
                task = self._task_queue.get_nowait()
                task.state = ProcessingState.CANCELLED
                pending_tasks.append(task)
        except Empty:
            pass
        
        if pending_tasks:
            logger.info(f"Cancelled {len(pending_tasks)} pending tasks")
        
        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=wait_for_completion)
        
        # Wait for processing thread
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=timeout)
            
        if self.status_callback:
            self.status_callback("Async processor stopped")
            
        logger.info("Async processor shutdown complete")
        
    def submit_task(self, 
                   task_id: str,
                   func: Callable,
                   *args,
                   timeout: Optional[float] = None,
                   **kwargs) -> bool:
        """
        Submit a task for asynchronous processing.
        
        Args:
            task_id: Unique identifier for the task
            func: Function to execute
            *args: Positional arguments for the function
            timeout: Task-specific timeout in seconds
            **kwargs: Keyword arguments for the function
            
        Returns:
            True if task was submitted successfully, False otherwise
        """
        if not self._is_running:
            logger.error("Processor is not running. Call start() first.")
            return False
            
        if task_id in self._active_tasks:
            logger.warning(f"Task {task_id} is already active")
            return False
            
        # Create task
        task = ProcessingTask(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            timeout_seconds=timeout or self.default_timeout
        )
        
        # Queue task
        self._task_queue.put(task)
        self._total_tasks += 1
        
        logger.info(f"Task {task_id} queued for processing")
        return True
        
    def get_task_status(self, task_id: str) -> Optional[ProcessingState]:
        """Get the current status of a task with enhanced tracking."""
        # Check active tasks first
        if task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            # Double-check task hasn't been abandoned
            if task.start_time and (time.time() - task.start_time) > task.timeout_seconds + 30:
                # Task has been running too long, mark as timeout
                logger.warning(f"Task {task_id} exceeded timeout, marking as timeout")
                task.state = ProcessingState.TIMEOUT
                task.error = TimeoutError(f"Task exceeded timeout of {task.timeout_seconds}s")
                task.end_time = time.time()
                self._completed_tasks[task_id] = self._active_tasks.pop(task_id)
                return ProcessingState.TIMEOUT
            return task.state
        
        # Check completed tasks
        elif task_id in self._completed_tasks:
            return self._completed_tasks[task_id].state
        
        # Check if task is still in queue (pending)
        try:
            # This is a bit expensive but necessary for proper tracking
            queue_copy = []
            found_in_queue = False
            
            # Temporarily drain queue to check for task
            try:
                while True:
                    task = self._task_queue.get_nowait()
                    queue_copy.append(task)
                    if task.task_id == task_id:
                        found_in_queue = True
            except Empty:
                pass
            
            # Put all tasks back in queue
            for task in queue_copy:
                self._task_queue.put(task)
            
            if found_in_queue:
                return ProcessingState.PENDING
                
        except Exception as e:
            logger.warning(f"Error checking queue for task {task_id}: {e}")
        
        # Task truly not found
        logger.warning(f"Task {task_id} not found in any collection")
        return None
        
    def get_task_result(self, task_id: str) -> Any:
        """Get the result of a completed task."""
        if task_id in self._completed_tasks:
            task = self._completed_tasks[task_id]
            if task.state == ProcessingState.COMPLETED:
                return task.result
            elif task.state == ProcessingState.FAILED:
                raise task.error
        return None
        
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current processing progress information."""
        active_count = len(self._active_tasks)
        pending_count = self._task_queue.qsize()
        
        progress_percentage = 0.0
        if self._total_tasks > 0:
            progress_percentage = (self._completed_count / self._total_tasks) * 100
        
        return {
            'total_tasks': self._total_tasks,
            'completed': self._completed_count,
            'failed': self._failed_count,
            'active': active_count,
            'pending': pending_count,
            'progress_percentage': progress_percentage,
            'is_running': self._is_running
        }
        
    def _process_tasks(self):
        """Main processing loop running in separate thread."""
        logger.info("Task processing thread started")
        
        while self._is_running:
            try:
                # Periodic cleanup of old completed tasks
                if time.time() - self._last_cleanup > self._cleanup_interval:
                    self._cleanup_completed_tasks()
                    
                # Get task from queue with timeout
                try:
                    task = self._task_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                if not self._is_running:
                    break
                
                # Move task to active
                self._active_tasks[task.task_id] = task
                task.state = ProcessingState.RUNNING
                task.start_time = time.time()
                
                if self.status_callback:
                    self.status_callback(f"Processing {task.task_id}...")
                
                # Submit to executor with timeout handling
                future = self._executor.submit(
                    self._execute_with_timeout,
                    task
                )
                
                # Wait for completion or timeout
                try:
                    result = future.result(timeout=task.timeout_seconds)
                    task.result = result
                    task.state = ProcessingState.COMPLETED
                    self._completed_count += 1
                    
                    logger.info(f"Task {task.task_id} completed successfully")
                    
                except TimeoutError:
                    task.state = ProcessingState.TIMEOUT
                    task.error = TimeoutError(f"Task {task.task_id} timed out after {task.timeout_seconds}s")
                    self._failed_count += 1
                    
                    logger.warning(f"Task {task.task_id} timed out")
                    
                except Exception as e:
                    task.state = ProcessingState.FAILED
                    task.error = e
                    self._failed_count += 1
                    
                    logger.error(f"Task {task.task_id} failed: {e}")
                
                # Finalize task
                task.end_time = time.time()
                
                # Move to completed and cleanup
                self._completed_tasks[task.task_id] = self._active_tasks.pop(task.task_id)
                
                # Update progress
                if self.progress_callback:
                    progress_info = self.get_progress_info()
                    self.progress_callback(
                        f"Completed {progress_info['completed']}/{progress_info['total_tasks']} tasks",
                        progress_info['progress_percentage']
                    )
                
            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                
        logger.info("Task processing thread stopped")
        
    def _execute_with_timeout(self, task: ProcessingTask) -> Any:
        """
        Execute a task function with proper error handling.
        
        Args:
            task: Task to execute
            
        Returns:
            Task result
        """
        try:
            return task.func(*task.args, **task.kwargs)
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            raise


    def _cleanup_completed_tasks(self):
        """
        Clean up old completed tasks to prevent memory accumulation.
        
        MEMORY MANAGEMENT:
        - Removes oldest completed tasks when limit is exceeded
        - Preserves recent task results for retrieval
        - Logs cleanup statistics
        """
        try:
            current_count = len(self._completed_tasks)
            
            if current_count <= self._max_completed_tasks:
                self._last_cleanup = time.time()
                return
                
            # Sort by completion time and keep only recent tasks
            sorted_tasks = sorted(
                self._completed_tasks.items(), 
                key=lambda x: x[1].end_time or 0, 
                reverse=True
            )
            
            # Keep only the most recent tasks
            tasks_to_keep = dict(sorted_tasks[:self._max_completed_tasks])
            removed_count = current_count - len(tasks_to_keep)
            
            self._completed_tasks = tasks_to_keep
            self._last_cleanup = time.time()
            
            logger.info(f"Cleaned up {removed_count} old completed tasks, {len(tasks_to_keep)} remaining")
            
        except Exception as e:
            logger.warning(f"Error during task cleanup: {e}")
            self._last_cleanup = time.time()  # Prevent continuous cleanup attempts


def async_timeout(timeout_seconds: float = 300.0):
    """
    Decorator for adding timeout protection to functions.
    
    Args:
        timeout_seconds: Timeout in seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def target_func():
                return func(*args, **kwargs)
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(target_func)
                try:
                    return future.result(timeout=timeout_seconds)
                except TimeoutError:
                    logger.warning(f"Function {func.__name__} timed out after {timeout_seconds}s")
                    raise TimeoutError(f"Function timed out after {timeout_seconds}s")
        return wrapper
    return decorator


class ProgressTracker:
    """Thread-safe progress tracker for multi-step operations."""
    
    def __init__(self, total_steps: int, callback: Optional[Callable[[str, float], None]] = None):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps in the operation
            callback: Function to call with progress updates
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.callback = callback
        self._lock = threading.Lock()
        
    def update(self, step_name: str, increment: int = 1):
        """
        Update progress.
        
        Args:
            step_name: Name of the current step
            increment: Number of steps to increment
        """
        with self._lock:
            self.current_step += increment
            if self.current_step > self.total_steps:
                self.current_step = self.total_steps
                
            percentage = (self.current_step / self.total_steps) * 100
            
            if self.callback:
                self.callback(step_name, percentage)
                
    def complete(self, message: str = "Complete"):
        """Mark progress as complete."""
        with self._lock:
            self.current_step = self.total_steps
            if self.callback:
                self.callback(message, 100.0) 