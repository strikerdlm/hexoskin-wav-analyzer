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
import json
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from queue import Queue, Empty
from typing import Dict, Any, Optional, Callable, List, Tuple
import logging
from dataclasses import dataclass, asdict
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
    - Timeout management with graceful recovery
    - Progress tracking and status callbacks
    - Memory-efficient task queue management
    - Graceful shutdown and cleanup
    - Result persistence to prevent work loss
    - Background processing controls
    - User notification system
    """
    
    def __init__(self, 
                 max_workers: int = 2, 
                 default_timeout: float = 300.0,
                 progress_callback: Optional[Callable[[str, float], None]] = None,
                 status_callback: Optional[Callable[[str], None]] = None,
                 result_persistence_dir: Optional[str] = None,
                 allow_background_processing: bool = False):
        """
        Initialize the async processor.
        
        Args:
            max_workers: Maximum number of parallel worker threads
            default_timeout: Default timeout for tasks in seconds
            progress_callback: Function to call with progress updates (message, percentage)
            status_callback: Function to call with status updates
            result_persistence_dir: Directory to save results for recovery
            allow_background_processing: Whether to continue processing when GUI is closed
        """
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        self.allow_background_processing = allow_background_processing
        
        # Result persistence
        self.result_persistence_dir = Path(result_persistence_dir) if result_persistence_dir else Path("hrv_results")
        self.result_persistence_dir.mkdir(exist_ok=True)
        
        # Threading components
        self._executor: Optional[ThreadPoolExecutor] = None
        self._task_queue: Queue = Queue()
        self._active_tasks: Dict[str, ProcessingTask] = {}
        self._completed_tasks: Dict[str, ProcessingTask] = {}
        self._persisted_results: Dict[str, str] = {}  # task_id -> file_path
        
        # State management
        self._is_running = False
        self._shutdown_event = threading.Event()
        self._processing_thread: Optional[threading.Thread] = None
        self._gui_connected = True  # Track if GUI is still connected
        
        # Statistics
        self._total_tasks = 0
        self._completed_count = 0
        self._failed_count = 0
        
        # Cleanup settings
        self._max_completed_tasks = 50  # Maximum completed tasks to keep in memory
        self._cleanup_interval = 300  # Cleanup every 5 minutes
        self._last_cleanup = time.time()
        
        # Recovery settings
        self._save_intermediate_results = True
        self._recovery_check_interval = 60  # Check for stalled tasks every minute
        self._max_retry_attempts = 2
        
        logger.info(f"SafeAsyncProcessor initialized with {max_workers} workers")
        logger.info(f"Result persistence: {'Enabled' if result_persistence_dir else 'Disabled'}")
        logger.info(f"Background processing: {'Allowed' if allow_background_processing else 'Disabled'}")
        
        # Load any existing persisted results
        self._load_persisted_results()
        
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
        
    def set_gui_connection_status(self, connected: bool):
        """
        Set GUI connection status to control background processing.
        
        Args:
            connected: True if GUI is connected and active
        """
        was_connected = self._gui_connected
        self._gui_connected = connected
        
        if connected and not was_connected:
            logger.info("GUI reconnected - resuming normal operation")
            if self.status_callback:
                self.status_callback("GUI reconnected")
                
        elif not connected and was_connected:
            logger.info("GUI disconnected")
            if not self.allow_background_processing:
                logger.info("Background processing disabled - pausing new tasks")
                if self.status_callback:
                    self.status_callback("GUI disconnected - background processing paused")
            else:
                logger.info("Background processing enabled - continuing tasks")
                self._notify_background_processing()
    
    def _safe_gui_callback(self, callback_func, *args, **kwargs):
        """
        Safely call a GUI callback function, handling thread disconnection.
        
        Args:
            callback_func: The callback function to call
            *args: Positional arguments for the callback
            **kwargs: Keyword arguments for the callback
            
        Returns:
            True if callback succeeded, False if GUI is disconnected
        """
        if not self._gui_connected or not callback_func:
            return False
            
        try:
            callback_func(*args, **kwargs)
            return True
        except Exception as e:
            if "main thread is not in main loop" in str(e) or "invalid command name" in str(e):
                logger.warning("GUI callback failed - main thread unavailable, disabling GUI callbacks")
                self._gui_connected = False
                return False
            else:
                logger.warning(f"GUI callback error: {e}")
                return False
    
    def _notify_background_processing(self):
        """Notify user about background processing."""
        if not self._gui_connected:
            logger.info("GUI not connected - skipping background processing notification")
            return
            
        try:
            import tkinter.messagebox as messagebox
            import threading
            
            def show_notification():
                try:
                    result = messagebox.askyesno(
                        "Background Processing",
                        "HRV Analysis is continuing in the background.\n\n" +
                        "• Analysis results will be automatically saved\n" +
                        "• You can close this application safely\n" +
                        "• Results will be available when you restart\n" +
                        "• Processing may take several more minutes\n\n" +
                        "Would you like to keep the application open to monitor progress?"
                    )
                    
                    if not result:
                        logger.info("User chose to close application during background processing")
                        self._save_current_state()
                        
                except Exception as e:
                    logger.warning(f"Could not show background processing notification: {e}")
                    if "main thread is not in main loop" in str(e):
                        self._gui_connected = False
            
            # Show notification in separate thread to avoid GUI blocking
            notification_thread = threading.Thread(target=show_notification, daemon=True)
            notification_thread.start()
            
        except ImportError:
            logger.info("GUI not available - background processing notification skipped")
    
    def _save_current_state(self):
        """Save current processing state for recovery."""
        try:
            state_file = self.result_persistence_dir / "processor_state.json"
            state = {
                'active_tasks': [task_id for task_id in self._active_tasks.keys()],
                'completed_tasks': list(self._persisted_results.keys()),
                'total_tasks': self._total_tasks,
                'completed_count': self._completed_count,
                'failed_count': self._failed_count,
                'timestamp': time.time()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Processor state saved to {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to save processor state: {e}")
    
    def _load_persisted_results(self):
        """Load previously persisted results."""
        try:
            # Load processor state
            state_file = self.result_persistence_dir / "processor_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                logger.info(f"Found previous session with {len(state.get('completed_tasks', []))} completed tasks")
            
            # Load individual result files
            for result_file in self.result_persistence_dir.glob("result_*.pkl"):
                task_id = result_file.stem.replace("result_", "")
                self._persisted_results[task_id] = str(result_file)
                logger.debug(f"Found persisted result for task: {task_id}")
                
            if self._persisted_results:
                logger.info(f"Loaded {len(self._persisted_results)} persisted results from previous sessions")
                
        except Exception as e:
            logger.warning(f"Error loading persisted results: {e}")
    
    def _persist_result(self, task_id: str, result: Any):
        """Persist task result to prevent loss."""
        try:
            if self._save_intermediate_results:
                result_file = self.result_persistence_dir / f"result_{task_id}.pkl"
                with open(result_file, 'wb') as f:
                    pickle.dump(result, f)
                self._persisted_results[task_id] = str(result_file)
                logger.debug(f"Persisted result for task {task_id}")
                
        except Exception as e:
            logger.warning(f"Failed to persist result for task {task_id}: {e}")
    
    def _load_persisted_result(self, task_id: str) -> Any:
        """Load a persisted result."""
        try:
            if task_id in self._persisted_results:
                result_file = self._persisted_results[task_id]
                with open(result_file, 'rb') as f:
                    result = pickle.load(f)
                logger.info(f"Loaded persisted result for task {task_id}")
                return result
        except Exception as e:
            logger.warning(f"Failed to load persisted result for task {task_id}: {e}")
        return None
    
    def get_persisted_tasks(self) -> List[str]:
        """Get list of tasks with persisted results."""
        return list(self._persisted_results.keys())
    
    def cleanup_persisted_results(self, max_age_hours: float = 24):
        """Clean up old persisted results."""
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            cleaned_count = 0
            
            for result_file in self.result_persistence_dir.glob("result_*.pkl"):
                if result_file.stat().st_mtime < cutoff_time:
                    task_id = result_file.stem.replace("result_", "")
                    result_file.unlink()
                    if task_id in self._persisted_results:
                        del self._persisted_results[task_id]
                    cleaned_count += 1
                    
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old persisted results")
                
        except Exception as e:
            logger.warning(f"Error cleaning up persisted results: {e}")
        
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
        """Enhanced processing loop with timeout recovery and result persistence."""
        logger.info("Task processing thread started with enhanced recovery features")
        
        while self._is_running:
            try:
                # Periodic cleanup of old completed tasks
                if time.time() - self._last_cleanup > self._cleanup_interval:
                    self._cleanup_completed_tasks()
                    self.cleanup_persisted_results()
                
                # Check if GUI is disconnected and background processing is disabled
                if not self._gui_connected and not self.allow_background_processing:
                    # Pause processing but don't exit
                    time.sleep(5)
                    continue
                    
                # Get task from queue with timeout
                try:
                    task = self._task_queue.get(timeout=1.0)
                except Empty:
                    # Check for stalled tasks during idle time
                    self._check_stalled_tasks()
                    continue
                
                if not self._is_running:
                    break
                
                # Check if we already have a persisted result for this task
                if task.task_id in self._persisted_results:
                    logger.info(f"Loading persisted result for task {task.task_id}")
                    persisted_result = self._load_persisted_result(task.task_id)
                    if persisted_result is not None:
                        task.result = persisted_result
                        task.state = ProcessingState.COMPLETED
                        task.start_time = time.time()
                        task.end_time = time.time()
                        self._completed_tasks[task.task_id] = task
                        self._completed_count += 1
                        
                        if self.status_callback:
                            self.status_callback(f"Loaded cached result for {task.task_id}")
                        continue
                
                # Move task to active
                self._active_tasks[task.task_id] = task
                task.state = ProcessingState.RUNNING
                task.start_time = time.time()
                
                self._safe_gui_callback(self.status_callback, f"Processing {task.task_id}...")
                
                # Submit to executor with enhanced timeout handling
                future = self._executor.submit(
                    self._execute_with_timeout,
                    task
                )
                
                # Wait for completion with extended timeout recovery
                try:
                    result = future.result(timeout=task.timeout_seconds)
                    task.result = result
                    task.state = ProcessingState.COMPLETED
                    self._completed_count += 1
                    
                    # Persist successful result
                    self._persist_result(task.task_id, result)
                    
                    logger.info(f"Task {task.task_id} completed and result persisted")
                    
                    # Notify user if GUI is disconnected
                    if not self._gui_connected:
                        self._notify_task_completion(task.task_id, success=True)
                    
                except TimeoutError:
                    # Enhanced timeout handling with retry logic
                    retry_count = getattr(task, 'retry_count', 0)
                    if retry_count < self._max_retry_attempts:
                        task.retry_count = retry_count + 1
                        logger.warning(f"Task {task.task_id} timed out, retrying ({retry_count + 1}/{self._max_retry_attempts})")
                        
                        # Increase timeout for retry
                        task.timeout_seconds = task.timeout_seconds * 1.5
                        
                        # Put task back in queue for retry
                        self._task_queue.put(task)
                        
                        # Remove from active tasks
                        if task.task_id in self._active_tasks:
                            del self._active_tasks[task.task_id]
                        continue
                    else:
                        # Final timeout - save partial results if any
                        task.state = ProcessingState.TIMEOUT
                        task.error = TimeoutError(f"Task {task.task_id} timed out after {task.timeout_seconds}s (max retries exceeded)")
                        self._failed_count += 1
                        
                        logger.error(f"Task {task.task_id} finally timed out after {retry_count + 1} attempts")
                        
                        # Try to save any partial results
                        self._try_save_partial_results(task)
                        
                        # Notify user if GUI is disconnected
                        if not self._gui_connected:
                            self._notify_task_completion(task.task_id, success=False, error="Timeout after retries")
                    
                except Exception as e:
                    task.state = ProcessingState.FAILED
                    task.error = e
                    self._failed_count += 1
                    
                    logger.error(f"Task {task.task_id} failed: {e}")
                    
                    # Try to save any partial results
                    self._try_save_partial_results(task)
                    
                    # Notify user if GUI is disconnected
                    if not self._gui_connected:
                        self._notify_task_completion(task.task_id, success=False, error=str(e))
                
                # Finalize task
                task.end_time = time.time()
                
                # Move to completed and cleanup
                self._completed_tasks[task.task_id] = self._active_tasks.pop(task.task_id, task)
                
                # Update progress with enhanced status
                if self.progress_callback and self._gui_connected:
                    try:
                        progress_info = self.get_progress_info()
                        
                        # Enhanced progress message
                        if progress_info['total_tasks'] > 0:
                            completed_pct = (progress_info['completed'] / progress_info['total_tasks']) * 100
                            status_msg = f"Completed {progress_info['completed']}/{progress_info['total_tasks']} tasks"
                            
                            if not self._gui_connected and self.allow_background_processing:
                                status_msg += " (Background)"
                                
                            self.progress_callback(status_msg, completed_pct)
                    except Exception as e:
                        # If GUI callback fails, likely main thread is gone
                        if "main thread is not in main loop" in str(e):
                            logger.warning("GUI callback failed - main thread unavailable, disabling GUI callbacks")
                            self._gui_connected = False
                        else:
                            logger.warning(f"Progress callback error: {e}")
                    
            except Exception as e:
                logger.error(f"Error in enhanced task processing loop: {e}")
                
        logger.info("Enhanced task processing thread stopped")
        
        # Save final state when shutting down
        if self._active_tasks or self._completed_tasks:
            self._save_current_state()
    
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

    def _check_stalled_tasks(self):
        """Check for and handle stalled tasks."""
        try:
            current_time = time.time()
            stalled_tasks = []
            
            for task_id, task in self._active_tasks.items():
                if task.start_time and (current_time - task.start_time) > (task.timeout_seconds + 60):
                    # Task has been running way too long
                    stalled_tasks.append(task_id)
            
            for task_id in stalled_tasks:
                task = self._active_tasks.get(task_id)
                if task:
                    logger.warning(f"Found stalled task {task_id}, moving to failed state")
                    task.state = ProcessingState.FAILED
                    task.error = Exception(f"Task stalled for {current_time - task.start_time:.0f} seconds")
                    task.end_time = current_time
                    
                    # Try to save partial results
                    self._try_save_partial_results(task)
                    
                    # Move to completed
                    self._completed_tasks[task_id] = self._active_tasks.pop(task_id)
                    self._failed_count += 1
                    
        except Exception as e:
            logger.warning(f"Error checking stalled tasks: {e}")
    
    def _try_save_partial_results(self, task: ProcessingTask):
        """Try to save any partial results from a failed/timed out task."""
        try:
            # Check if task has any partial results we can save
            if hasattr(task, 'partial_results') and task.partial_results:
                partial_file = self.result_persistence_dir / f"partial_{task.task_id}.json"
                with open(partial_file, 'w') as f:
                    json.dump(task.partial_results, f, indent=2)
                logger.info(f"Saved partial results for task {task.task_id}")
                
            # Also save task metadata for debugging
            metadata = {
                'task_id': task.task_id,
                'start_time': task.start_time,
                'end_time': task.end_time,
                'timeout_seconds': task.timeout_seconds,
                'error': str(task.error) if task.error else None,
                'state': task.state.value,
                'retry_count': getattr(task, 'retry_count', 0)
            }
            
            metadata_file = self.result_persistence_dir / f"metadata_{task.task_id}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save partial results for task {task.task_id}: {e}")
    
    def _notify_task_completion(self, task_id: str, success: bool, error: str = None):
        """Notify user about task completion when GUI is disconnected."""
        try:
            # Create notification file
            notification_file = self.result_persistence_dir / f"notification_{task_id}.json"
            notification = {
                'task_id': task_id,
                'success': success,
                'error': error,
                'timestamp': time.time(),
                'message': f"Task {task_id} {'completed successfully' if success else 'failed'}" + 
                          (f": {error}" if error else "")
            }
            
            with open(notification_file, 'w') as f:
                json.dump(notification, f, indent=2)
            
            logger.info(f"Created notification for task {task_id}: {'Success' if success else 'Failed'}")
            
            # Try system notification if available
            try:
                import subprocess
                import platform
                
                message = notification['message']
                title = "HRV Analysis - Background Processing"
                
                if platform.system() == "Windows":
                    # Windows toast notification
                    subprocess.run([
                        "powershell", "-Command",
                        f"[reflection.assembly]::loadwithpartialname('System.Windows.Forms'); " +
                        f"[System.Windows.Forms.MessageBox]::Show('{message}', '{title}', 'OK', 'Information')"
                    ], check=False, timeout=5)
                elif platform.system() == "Darwin":
                    # macOS notification
                    subprocess.run([
                        "osascript", "-e",
                        f'display notification "{message}" with title "{title}"'
                    ], check=False, timeout=5)
                elif platform.system() == "Linux":
                    # Linux notification
                    subprocess.run([
                        "notify-send", title, message
                    ], check=False, timeout=5)
                    
            except Exception as notify_error:
                logger.debug(f"System notification failed (non-critical): {notify_error}")
                
        except Exception as e:
            logger.warning(f"Failed to create notification for task {task_id}: {e}")

    def get_pending_notifications(self) -> List[Dict[str, Any]]:
        """Get any pending notifications from background processing."""
        notifications = []
        try:
            for notification_file in self.result_persistence_dir.glob("notification_*.json"):
                with open(notification_file) as f:
                    notification = json.load(f)
                notifications.append(notification)
                
                # Clean up notification file after reading
                notification_file.unlink()
                
        except Exception as e:
            logger.warning(f"Error reading pending notifications: {e}")
            
        return notifications


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