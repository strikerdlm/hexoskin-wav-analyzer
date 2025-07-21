"""
Settings Configuration Panel for Enhanced HRV Analysis

This module provides a comprehensive settings interface for configuring:
- Intelligent caching parameters (memory limits, TTL, etc.)
- Asynchronous processing settings (workers, timeouts)
- Database optimization options (chunk sizes, indexing)
- Performance monitoring preferences
- Advanced analysis configurations

Author: Dr. Diego Malpica - Aerospace Medicine Specialist
Organization: DIMAE / FAC / Colombia
Project: Valquiria Crew Space Simulation HRV Analysis System
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)

class SettingsPanel:
    """Advanced settings configuration panel for HRV Analysis application."""
    
    def __init__(self, parent_window: tk.Tk, 
                 settings_file: str = "hrv_analysis_settings.json",
                 on_settings_changed: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize settings panel.
        
        Args:
            parent_window: Parent tkinter window
            settings_file: Path to settings file for persistence
            on_settings_changed: Callback when settings are modified
        """
        self.parent_window = parent_window
        self.settings_file = Path(settings_file)
        self.on_settings_changed = on_settings_changed
        
        # Default settings
        self.default_settings = {
            # Cache settings
            "cache_max_memory_mb": 500,
            "cache_max_entries": 1000,
            "cache_ttl_hours": 24.0,
            "cache_enabled": True,
            
            # Async processing settings
            "async_max_workers": 2,
            "async_timeout_seconds": 300.0,
            "async_enabled": True,
            "async_allow_background_processing": False,  # New setting
            
            # Database optimization settings
            "db_chunk_size": 50000,
            "db_max_memory_mb": 1000.0,
            "db_parallel_loading": True,
            "db_create_indexes": True,
            
            # Performance monitoring
            "monitor_update_interval": 2.0,
            "monitor_max_history": 60,
            "monitor_auto_start": False,
            
            # Analysis settings
            "analysis_memory_limit_enabled": True,
            "analysis_max_bootstrap_samples": 50,
            "analysis_fast_mode_default": False,
            
            # UI settings
            "ui_show_performance_monitor": True,
            "ui_show_cache_status": True,
            "ui_auto_save_settings": True
        }
        
        # Current settings (loaded from file or defaults)
        self.current_settings = self._load_settings()
        
        # UI components
        self.settings_window = None
        self.setting_vars = {}
        
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from file or use defaults."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                
                # Merge with defaults to handle new settings
                settings = self.default_settings.copy()
                settings.update(loaded_settings)
                
                logger.info(f"Settings loaded from {self.settings_file}")
                return settings
            else:
                logger.info("Using default settings")
                return self.default_settings.copy()
                
        except Exception as e:
            logger.error(f"Error loading settings: {e}, using defaults")
            return self.default_settings.copy()
    
    def _save_settings(self):
        """Save current settings to file."""
        try:
            # Update current settings from UI
            self._update_settings_from_ui()
            
            with open(self.settings_file, 'w') as f:
                json.dump(self.current_settings, f, indent=2)
            
            logger.info(f"Settings saved to {self.settings_file}")
            
            # Notify callback
            if self.on_settings_changed:
                self.on_settings_changed(self.current_settings.copy())
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {e}")
            return False
    
    def show_settings_dialog(self):
        """Show the settings configuration dialog."""
        if self.settings_window and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return
        
        # Create settings window
        self.settings_window = tk.Toplevel(self.parent_window)
        self.settings_window.title("Enhanced HRV Analysis - Advanced Settings")
        self.settings_window.geometry("600x700")
        self.settings_window.resizable(True, True)
        
        # Make it modal
        self.settings_window.transient(self.parent_window)
        self.settings_window.grab_set()
        
        self._setup_settings_ui()
        
        # Center the window
        self.settings_window.update_idletasks()
        x = (self.settings_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (self.settings_window.winfo_screenheight() // 2) - (700 // 2)
        self.settings_window.geometry(f"600x700+{x}+{y}")
    
    def _setup_settings_ui(self):
        """Setup the settings UI components."""
        # Main scrollable frame
        main_frame = ttk.Frame(self.settings_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for different setting categories
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Cache settings tab
        self._setup_cache_settings_tab(notebook)
        
        # Async processing settings tab
        self._setup_async_settings_tab(notebook)
        
        # Database optimization settings tab
        self._setup_database_settings_tab(notebook)
        
        # Performance monitoring settings tab
        self._setup_monitoring_settings_tab(notebook)
        
        # Analysis settings tab
        self._setup_analysis_settings_tab(notebook)
        
        # UI settings tab
        self._setup_ui_settings_tab(notebook)
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=5)
        
        # Control buttons
        ttk.Button(buttons_frame, text="Save", 
                  command=self._save_and_close).pack(side=tk.RIGHT, padx=5)
        ttk.Button(buttons_frame, text="Cancel", 
                  command=self._cancel_changes).pack(side=tk.RIGHT, padx=5)
        ttk.Button(buttons_frame, text="Apply", 
                  command=self._apply_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(buttons_frame, text="Reset to Defaults", 
                  command=self._reset_to_defaults).pack(side=tk.LEFT, padx=5)
    
    def _setup_cache_settings_tab(self, notebook: ttk.Notebook):
        """Setup cache settings tab."""
        cache_frame = ttk.Frame(notebook)
        notebook.add(cache_frame, text="🗄️ Intelligent Cache")
        
        # Cache enabled - Fix: specify master window to prevent "no default root window" error
        self.setting_vars['cache_enabled'] = tk.BooleanVar(master=self.settings_window, 
                                                          value=self.current_settings['cache_enabled'])
        ttk.Checkbutton(cache_frame, text="Enable intelligent caching", 
                       variable=self.setting_vars['cache_enabled']).pack(anchor=tk.W, pady=5)
        
        # Memory limit
        memory_frame = ttk.LabelFrame(cache_frame, text="Memory Settings", padding="5")
        memory_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(memory_frame, text="Maximum Cache Memory (MB):").pack(anchor=tk.W)
        self.setting_vars['cache_max_memory_mb'] = tk.IntVar(master=self.settings_window,
                                                            value=self.current_settings['cache_max_memory_mb'])
        memory_scale = ttk.Scale(memory_frame, from_=100, to=2000, 
                                variable=self.setting_vars['cache_max_memory_mb'],
                                orient=tk.HORIZONTAL, length=300)
        memory_scale.pack(fill=tk.X, pady=2)
        memory_label = ttk.Label(memory_frame, text="")
        memory_label.pack(anchor=tk.W)
        
        def update_memory_label(*args):
            memory_label.config(text=f"Current: {self.setting_vars['cache_max_memory_mb'].get()} MB")
        
        self.setting_vars['cache_max_memory_mb'].trace('w', update_memory_label)
        update_memory_label()
        
        # Max entries
        entries_frame = ttk.LabelFrame(cache_frame, text="Entry Limits", padding="5")
        entries_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(entries_frame, text="Maximum Cache Entries:").pack(anchor=tk.W)
        self.setting_vars['cache_max_entries'] = tk.IntVar(master=self.settings_window,
                                                          value=self.current_settings['cache_max_entries'])
        entries_scale = ttk.Scale(entries_frame, from_=100, to=5000,
                                 variable=self.setting_vars['cache_max_entries'],
                                 orient=tk.HORIZONTAL, length=300)
        entries_scale.pack(fill=tk.X, pady=2)
        entries_label = ttk.Label(entries_frame, text="")
        entries_label.pack(anchor=tk.W)
        
        def update_entries_label(*args):
            entries_label.config(text=f"Current: {self.setting_vars['cache_max_entries'].get()} entries")
        
        self.setting_vars['cache_max_entries'].trace('w', update_entries_label)
        update_entries_label()
        
        # TTL
        ttl_frame = ttk.LabelFrame(cache_frame, text="Time-to-Live Settings", padding="5")
        ttl_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(ttl_frame, text="Cache TTL (hours):").pack(anchor=tk.W)
        self.setting_vars['cache_ttl_hours'] = tk.DoubleVar(master=self.settings_window,
                                                           value=self.current_settings['cache_ttl_hours'])
        ttk.Entry(ttl_frame, textvariable=self.setting_vars['cache_ttl_hours'], width=15).pack(anchor=tk.W, pady=2)
    
    def _setup_async_settings_tab(self, notebook: ttk.Notebook):
        """Setup async processing settings tab."""
        async_frame = ttk.Frame(notebook)
        notebook.add(async_frame, text="⚡ Async Processing")
        
        # Async enabled
        self.setting_vars['async_enabled'] = tk.BooleanVar(value=self.current_settings['async_enabled'])
        ttk.Checkbutton(async_frame, text="Enable asynchronous processing", 
                       variable=self.setting_vars['async_enabled']).pack(anchor=tk.W, pady=5)
        
        # Worker threads
        workers_frame = ttk.LabelFrame(async_frame, text="Thread Pool Settings", padding="5")
        workers_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(workers_frame, text="Maximum Worker Threads:").pack(anchor=tk.W)
        self.setting_vars['async_max_workers'] = tk.IntVar(value=self.current_settings['async_max_workers'])
        workers_scale = ttk.Scale(workers_frame, from_=1, to=8,
                                 variable=self.setting_vars['async_max_workers'],
                                 orient=tk.HORIZONTAL, length=300)
        workers_scale.pack(fill=tk.X, pady=2)
        
        # Timeout
        timeout_frame = ttk.LabelFrame(async_frame, text="Timeout Settings", padding="5")
        timeout_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(timeout_frame, text="Task Timeout (seconds):").pack(anchor=tk.W)
        self.setting_vars['async_timeout_seconds'] = tk.DoubleVar(value=self.current_settings['async_timeout_seconds'])
        ttk.Entry(timeout_frame, textvariable=self.setting_vars['async_timeout_seconds'], width=15).pack(anchor=tk.W, pady=2)

        # Background processing
        self.setting_vars['async_allow_background_processing'] = tk.BooleanVar(value=self.current_settings['async_allow_background_processing'])
        ttk.Checkbutton(async_frame, text="Allow background processing (experimental)", 
                       variable=self.setting_vars['async_allow_background_processing']).pack(anchor=tk.W, pady=5)
    
    def _setup_database_settings_tab(self, notebook: ttk.Notebook):
        """Setup database optimization settings tab."""
        db_frame = ttk.Frame(notebook)
        notebook.add(db_frame, text="🗄️ Database")
        
        # Parallel loading
        self.setting_vars['db_parallel_loading'] = tk.BooleanVar(value=self.current_settings['db_parallel_loading'])
        ttk.Checkbutton(db_frame, text="Enable parallel database loading", 
                       variable=self.setting_vars['db_parallel_loading']).pack(anchor=tk.W, pady=5)
        
        # Auto-create indexes
        self.setting_vars['db_create_indexes'] = tk.BooleanVar(value=self.current_settings['db_create_indexes'])
        ttk.Checkbutton(db_frame, text="Automatically create performance indexes", 
                       variable=self.setting_vars['db_create_indexes']).pack(anchor=tk.W, pady=5)
        
        # Chunk size
        chunk_frame = ttk.LabelFrame(db_frame, text="Chunked Loading", padding="5")
        chunk_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(chunk_frame, text="Records per Chunk:").pack(anchor=tk.W)
        self.setting_vars['db_chunk_size'] = tk.IntVar(value=self.current_settings['db_chunk_size'])
        chunk_scale = ttk.Scale(chunk_frame, from_=10000, to=200000,
                               variable=self.setting_vars['db_chunk_size'],
                               orient=tk.HORIZONTAL, length=300)
        chunk_scale.pack(fill=tk.X, pady=2)
        
        # Database memory limit
        db_memory_frame = ttk.LabelFrame(db_frame, text="Memory Management", padding="5")
        db_memory_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(db_memory_frame, text="Database Memory Limit (MB):").pack(anchor=tk.W)
        self.setting_vars['db_max_memory_mb'] = tk.DoubleVar(value=self.current_settings['db_max_memory_mb'])
        ttk.Entry(db_memory_frame, textvariable=self.setting_vars['db_max_memory_mb'], width=15).pack(anchor=tk.W, pady=2)
    
    def _setup_monitoring_settings_tab(self, notebook: ttk.Notebook):
        """Setup performance monitoring settings tab."""
        monitor_frame = ttk.Frame(notebook)
        notebook.add(monitor_frame, text="📊 Monitoring")
        
        # Auto-start monitoring
        self.setting_vars['monitor_auto_start'] = tk.BooleanVar(value=self.current_settings['monitor_auto_start'])
        ttk.Checkbutton(monitor_frame, text="Auto-start performance monitoring", 
                       variable=self.setting_vars['monitor_auto_start']).pack(anchor=tk.W, pady=5)
        
        # Update interval
        interval_frame = ttk.LabelFrame(monitor_frame, text="Update Settings", padding="5")
        interval_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(interval_frame, text="Update Interval (seconds):").pack(anchor=tk.W)
        self.setting_vars['monitor_update_interval'] = tk.DoubleVar(value=self.current_settings['monitor_update_interval'])
        ttk.Entry(interval_frame, textvariable=self.setting_vars['monitor_update_interval'], width=15).pack(anchor=tk.W, pady=2)
        
        # History length
        ttk.Label(interval_frame, text="History Length (samples):").pack(anchor=tk.W, pady=(10,0))
        self.setting_vars['monitor_max_history'] = tk.IntVar(value=self.current_settings['monitor_max_history'])
        ttk.Entry(interval_frame, textvariable=self.setting_vars['monitor_max_history'], width=15).pack(anchor=tk.W, pady=2)
    
    def _setup_analysis_settings_tab(self, notebook: ttk.Notebook):
        """Setup analysis settings tab."""
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="🔬 Analysis")
        
        # Memory limiting
        self.setting_vars['analysis_memory_limit_enabled'] = tk.BooleanVar(value=self.current_settings['analysis_memory_limit_enabled'])
        ttk.Checkbutton(analysis_frame, text="Enable intelligent memory management", 
                       variable=self.setting_vars['analysis_memory_limit_enabled']).pack(anchor=tk.W, pady=5)
        
        # Bootstrap samples
        bootstrap_frame = ttk.LabelFrame(analysis_frame, text="Statistical Settings", padding="5")
        bootstrap_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(bootstrap_frame, text="Maximum Bootstrap Samples:").pack(anchor=tk.W)
        self.setting_vars['analysis_max_bootstrap_samples'] = tk.IntVar(value=self.current_settings['analysis_max_bootstrap_samples'])
        bootstrap_scale = ttk.Scale(bootstrap_frame, from_=25, to=250,
                                   variable=self.setting_vars['analysis_max_bootstrap_samples'],
                                   orient=tk.HORIZONTAL, length=300)
        bootstrap_scale.pack(fill=tk.X, pady=2)
        
        # Fast mode default
        self.setting_vars['analysis_fast_mode_default'] = tk.BooleanVar(value=self.current_settings['analysis_fast_mode_default'])
        ttk.Checkbutton(analysis_frame, text="Enable fast mode by default", 
                       variable=self.setting_vars['analysis_fast_mode_default']).pack(anchor=tk.W, pady=5)
    
    def _setup_ui_settings_tab(self, notebook: ttk.Notebook):
        """Setup UI settings tab."""
        ui_frame = ttk.Frame(notebook)
        notebook.add(ui_frame, text="🖥️ Interface")
        
        # Performance monitor display
        self.setting_vars['ui_show_performance_monitor'] = tk.BooleanVar(value=self.current_settings['ui_show_performance_monitor'])
        ttk.Checkbutton(ui_frame, text="Show performance monitor panel", 
                       variable=self.setting_vars['ui_show_performance_monitor']).pack(anchor=tk.W, pady=5)
        
        # Cache status display
        self.setting_vars['ui_show_cache_status'] = tk.BooleanVar(value=self.current_settings['ui_show_cache_status'])
        ttk.Checkbutton(ui_frame, text="Show cache status indicators", 
                       variable=self.setting_vars['ui_show_cache_status']).pack(anchor=tk.W, pady=5)
        
        # Auto-save settings
        self.setting_vars['ui_auto_save_settings'] = tk.BooleanVar(value=self.current_settings['ui_auto_save_settings'])
        ttk.Checkbutton(ui_frame, text="Auto-save settings on changes", 
                       variable=self.setting_vars['ui_auto_save_settings']).pack(anchor=tk.W, pady=5)
    
    def _update_settings_from_ui(self):
        """Update current settings from UI components."""
        for key, var in self.setting_vars.items():
            if hasattr(var, 'get'):
                self.current_settings[key] = var.get()
    
    def _apply_settings(self):
        """Apply settings without closing dialog."""
        if self._save_settings():
            messagebox.showinfo("Settings", "Settings applied successfully!")
    
    def _save_and_close(self):
        """Save settings and close dialog."""
        if self._save_settings():
            self.settings_window.destroy()
    
    def _cancel_changes(self):
        """Cancel changes and close dialog."""
        self.settings_window.destroy()
    
    def _reset_to_defaults(self):
        """Reset all settings to defaults."""
        result = messagebox.askyesno("Reset Settings", 
                                   "Are you sure you want to reset all settings to defaults?\n" +
                                   "This cannot be undone.")
        if result:
            self.current_settings = self.default_settings.copy()
            
            # Update UI
            for key, var in self.setting_vars.items():
                if key in self.current_settings:
                    var.set(self.current_settings[key])
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current settings."""
        return self.current_settings.copy()
    
    def update_setting(self, key: str, value: Any):
        """Update a specific setting."""
        if key in self.current_settings:
            self.current_settings[key] = value
            
            # Auto-save if enabled
            if self.current_settings.get('ui_auto_save_settings', True):
                self._save_settings() 