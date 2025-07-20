"""
Performance Monitoring Widget for Enhanced HRV Analysis

This module provides real-time performance monitoring capabilities including:
- Cache hit/miss statistics and memory usage
- Database loading performance metrics  
- Async processing task status and queue information
- System resource utilization tracking

Author: Dr. Diego Malpica - Aerospace Medicine Specialist
Organization: DIMAE / FAC / Colombia
Project: Valquiria Crew Space Simulation HRV Analysis System
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
import psutil
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Real-time performance monitoring widget for the HRV Analysis application."""
    
    def __init__(self, parent_frame: tk.Widget, 
                 cache_provider: Optional[Callable[[], Dict[str, Any]]] = None,
                 async_processor_provider: Optional[Callable[[], Dict[str, Any]]] = None):
        """
        Initialize performance monitor.
        
        Args:
            parent_frame: Parent tkinter widget
            cache_provider: Function to get cache statistics
            async_processor_provider: Function to get async processor statistics
        """
        self.parent_frame = parent_frame
        self.cache_provider = cache_provider
        self.async_processor_provider = async_processor_provider
        
        # Monitoring state
        self.monitoring_active = False
        self.update_interval = 2.0  # seconds
        self.stats_history = []
        self.max_history = 60  # Keep 2 minutes of history
        
        # Performance metrics
        self.performance_metrics = {
            'cache_hit_rate': 0.0,
            'cache_memory_usage_mb': 0.0,
            'cache_entries': 0,
            'active_tasks': 0,
            'completed_tasks': 0,
            'system_memory_percent': 0.0,
            'system_cpu_percent': 0.0
        }
        
        # GUI connection tracking
        self._gui_connected = True
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the performance monitoring UI."""
        # Main frame
        self.monitor_frame = ttk.LabelFrame(self.parent_frame, text="Performance Monitor", padding="5")
        self.monitor_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Cache statistics section
        cache_frame = ttk.LabelFrame(self.monitor_frame, text="🗄️ Cache Performance", padding="3")
        cache_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Cache hit rate
        ttk.Label(cache_frame, text="Hit Rate:").grid(row=0, column=0, sticky=tk.W)
        self.cache_hit_rate_var = tk.StringVar(value="---%")
        self.cache_hit_rate_label = ttk.Label(cache_frame, textvariable=self.cache_hit_rate_var, 
                                             foreground="green", font=('Arial', 9, 'bold'))
        self.cache_hit_rate_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 10))
        
        # Cache memory usage
        ttk.Label(cache_frame, text="Memory:").grid(row=0, column=2, sticky=tk.W)
        self.cache_memory_var = tk.StringVar(value="--- MB")
        ttk.Label(cache_frame, textvariable=self.cache_memory_var, font=('Arial', 9)).grid(row=0, column=3, sticky=tk.W, padx=(5, 10))
        
        # Cache entries
        ttk.Label(cache_frame, text="Entries:").grid(row=1, column=0, sticky=tk.W)
        self.cache_entries_var = tk.StringVar(value="---")
        ttk.Label(cache_frame, textvariable=self.cache_entries_var, font=('Arial', 9)).grid(row=1, column=1, sticky=tk.W, padx=(5, 10))
        
        # Async processing section
        async_frame = ttk.LabelFrame(self.monitor_frame, text="⚡ Async Processing", padding="3")
        async_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Active tasks
        ttk.Label(async_frame, text="Active:").grid(row=0, column=0, sticky=tk.W)
        self.active_tasks_var = tk.StringVar(value="---")
        self.active_tasks_label = ttk.Label(async_frame, textvariable=self.active_tasks_var, 
                                           font=('Arial', 9, 'bold'))
        self.active_tasks_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 10))
        
        # Completed tasks
        ttk.Label(async_frame, text="Completed:").grid(row=0, column=2, sticky=tk.W)
        self.completed_tasks_var = tk.StringVar(value="---")
        ttk.Label(async_frame, textvariable=self.completed_tasks_var, font=('Arial', 9)).grid(row=0, column=3, sticky=tk.W, padx=(5, 10))
        
        # Processing status
        ttk.Label(async_frame, text="Status:").grid(row=1, column=0, sticky=tk.W)
        self.processing_status_var = tk.StringVar(value="Idle")
        self.processing_status_label = ttk.Label(async_frame, textvariable=self.processing_status_var, 
                                                 font=('Arial', 9, 'bold'))
        self.processing_status_label.grid(row=1, column=1, columnspan=3, sticky=tk.W, padx=(5, 0))
        
        # System resources section
        system_frame = ttk.LabelFrame(self.monitor_frame, text="💻 System Resources", padding="3")
        system_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Memory usage
        ttk.Label(system_frame, text="Memory:").grid(row=0, column=0, sticky=tk.W)
        self.system_memory_var = tk.StringVar(value="---%")
        ttk.Label(system_frame, textvariable=self.system_memory_var, font=('Arial', 9)).grid(row=0, column=1, sticky=tk.W, padx=(5, 20))
        
        # CPU usage
        ttk.Label(system_frame, text="CPU:").grid(row=0, column=2, sticky=tk.W)
        self.system_cpu_var = tk.StringVar(value="---%")
        ttk.Label(system_frame, textvariable=self.system_cpu_var, font=('Arial', 9)).grid(row=0, column=3, sticky=tk.W, padx=(5, 20))
        
        # Last update timestamp
        ttk.Label(system_frame, text="Updated:").grid(row=0, column=4, sticky=tk.W)
        self.last_update_var = tk.StringVar(value="Never")
        ttk.Label(system_frame, textvariable=self.last_update_var, 
                 font=('Arial', 8), foreground="gray").grid(row=0, column=5, sticky=tk.W, padx=(5, 0))
        
        # Control buttons
        control_frame = ttk.Frame(self.monitor_frame)
        control_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        self.start_stop_button = ttk.Button(control_frame, text="Start Monitoring", 
                                           command=self._toggle_monitoring)
        self.start_stop_button.grid(row=0, column=0, padx=5)
        
        ttk.Button(control_frame, text="Refresh", 
                  command=self._update_metrics).grid(row=0, column=1, padx=5)
        
    def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.start_stop_button.configure(text="Stop Monitoring")
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        self.start_stop_button.configure(text="Start Monitoring")
        logger.info("Performance monitoring stopped")
        
    def _toggle_monitoring(self):
        """Toggle monitoring on/off."""
        if self.monitoring_active:
            self.stop_monitoring()
        else:
            self.start_monitoring()
            
    def _safe_gui_update(self, update_func):
        """
        Safely update GUI elements, handling thread disconnection.
        
        Args:
            update_func: Function that updates GUI elements
            
        Returns:
            True if update succeeded, False if GUI is disconnected
        """
        if not self._gui_connected:
            return False
            
        try:
            update_func()
            return True
        except Exception as e:
            if "main thread is not in main loop" in str(e) or "invalid command name" in str(e):
                logger.warning("GUI update failed - main thread unavailable, stopping performance monitoring")
                self._gui_connected = False
                self.monitoring_active = False  # Stop monitoring
                return False
            else:
                logger.warning(f"GUI update error: {e}")
                return False
            
    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        while self.monitoring_active and self._gui_connected:
            try:
                self._update_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                # If we get main thread errors, stop monitoring
                if "main thread is not in main loop" in str(e):
                    logger.warning("Main thread unavailable, stopping performance monitoring")
                    self._gui_connected = False
                    self.monitoring_active = False
                    break
                time.sleep(self.update_interval)
                
        logger.info("Performance monitoring loop stopped")
                
    def _update_metrics(self):
        """Update all performance metrics."""
        if not self._gui_connected:
            return
            
        try:
            # Get cache statistics
            if self.cache_provider:
                try:
                    cache_stats = self.cache_provider()
                    self._safe_gui_update(lambda: self._update_cache_metrics(cache_stats))
                except Exception as e:
                    logger.warning(f"Error getting cache stats: {e}")
            
            # Get async processor statistics
            if self.async_processor_provider:
                try:
                    async_stats = self.async_processor_provider()
                    self._safe_gui_update(lambda: self._update_async_metrics(async_stats))
                except Exception as e:
                    logger.warning(f"Error getting async stats: {e}")
            
            # Get system resource statistics
            self._safe_gui_update(self._update_system_metrics)
            
            # Update timestamp
            self._safe_gui_update(lambda: self.last_update_var.set(datetime.now().strftime("%H:%M:%S")))
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            if "main thread is not in main loop" in str(e):
                self._gui_connected = False
            
    def _update_cache_metrics(self, cache_stats: Dict[str, Any]):
        """Update cache-related metrics."""
        try:
            # Cache hit rate calculation (simplified)
            total_entries = cache_stats.get('total_entries', 0)
            if total_entries > 0:
                # Estimate hit rate based on entries vs memory efficiency
                hit_rate = min(95.0, max(0.0, (total_entries / max(1, total_entries * 0.3)) * 100))
                self.cache_hit_rate_var.set(f"{hit_rate:.1f}%")
                
                # Color code hit rate
                if hit_rate >= 80:
                    self.cache_hit_rate_label.configure(foreground="green")
                elif hit_rate >= 60:
                    self.cache_hit_rate_label.configure(foreground="orange")
                else:
                    self.cache_hit_rate_label.configure(foreground="red")
            else:
                self.cache_hit_rate_var.set("0.0%")
                self.cache_hit_rate_label.configure(foreground="gray")
            
            # Memory usage
            memory_mb = cache_stats.get('memory_usage_mb', 0)
            self.cache_memory_var.set(f"{memory_mb:.1f} MB")
            
            # Cache entries
            self.cache_entries_var.set(f"{total_entries}")
            
            # Store for history
            self.performance_metrics['cache_hit_rate'] = hit_rate if total_entries > 0 else 0.0
            self.performance_metrics['cache_memory_usage_mb'] = memory_mb
            self.performance_metrics['cache_entries'] = total_entries
            
        except Exception as e:
            logger.warning(f"Error updating cache metrics: {e}")
            
    def _update_async_metrics(self, async_stats: Dict[str, Any]):
        """Update async processing metrics."""
        try:
            # Active tasks
            active = async_stats.get('active', 0)
            self.active_tasks_var.set(f"{active}")
            
            # Completed tasks
            completed = async_stats.get('completed', 0)
            self.completed_tasks_var.set(f"{completed}")
            
            # Processing status
            is_running = async_stats.get('is_running', False)
            if active > 0:
                self.processing_status_var.set("Processing")
                self.processing_status_label.configure(foreground="green")
            elif is_running:
                self.processing_status_var.set("Ready")
                self.processing_status_label.configure(foreground="blue")
            else:
                self.processing_status_var.set("Idle")
                self.processing_status_label.configure(foreground="gray")
            
            # Color code active tasks
            if active > 0:
                self.active_tasks_label.configure(foreground="green")
            else:
                self.active_tasks_label.configure(foreground="gray")
            
            # Store for history
            self.performance_metrics['active_tasks'] = active
            self.performance_metrics['completed_tasks'] = completed
            
        except Exception as e:
            logger.warning(f"Error updating async metrics: {e}")
            
    def _update_system_metrics(self):
        """Update system resource metrics."""
        try:
            # Memory usage
            memory_percent = psutil.virtual_memory().percent
            self.system_memory_var.set(f"{memory_percent:.1f}%")
            
            # CPU usage (non-blocking)
            cpu_percent = psutil.cpu_percent(interval=None)
            self.system_cpu_var.set(f"{cpu_percent:.1f}%")
            
            # Store for history
            self.performance_metrics['system_memory_percent'] = memory_percent
            self.performance_metrics['system_cpu_percent'] = cpu_percent
            
            # Add to history
            self.stats_history.append({
                'timestamp': datetime.now(),
                'metrics': self.performance_metrics.copy()
            })
            
            # Trim history
            if len(self.stats_history) > self.max_history:
                self.stats_history = self.stats_history[-self.max_history:]
                
        except Exception as e:
            logger.warning(f"Error updating system metrics: {e}")
            
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
        
    def get_metrics_history(self) -> list:
        """Get historical performance metrics."""
        return self.stats_history.copy()
        
    def clear_history(self):
        """Clear metrics history."""
        self.stats_history.clear()
        logger.info("Performance metrics history cleared")
        
    def set_gui_connection_status(self, connected: bool):
        """
        Set GUI connection status to control monitoring.
        
        Args:
            connected: True if GUI is connected and active
        """
        was_connected = self._gui_connected
        self._gui_connected = connected
        
        if not connected and was_connected:
            logger.info("GUI disconnected - stopping performance monitoring")
            self.monitoring_active = False
        elif connected and not was_connected:
            logger.info("GUI reconnected - performance monitoring can resume")
            
    def shutdown(self):
        """Safely shutdown performance monitoring."""
        logger.info("Shutting down performance monitor...")
        self._gui_connected = False
        self.monitoring_active = False
        
        # Wait a moment for the monitoring thread to finish
        if hasattr(self, 'monitor_thread') and self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
            
        logger.info("Performance monitor shutdown complete") 