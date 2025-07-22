"""
Main Tkinter Application for Enhanced HRV Analysis

Author: Dr. Diego Malpica - Aerospace Medicine Specialist
Organization: DIMAE / FAC / Colombia
Project: Valquiria Crew Space Simulation HRV Analysis System

This module provides a comprehensive GUI application for HRV analysis featuring:
- Modern themed interface with customizable styling
- Data loading and management with quality assessment
- Interactive analysis configuration
- Real-time visualization and results display
- Export capabilities for results and visualizations
- Progress tracking and status updates
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.scrolledtext as scrolledtext
from pathlib import Path
import threading
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import gc
import psutil
import time

# Import enhanced HRV analysis components
# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from core.data_loader import DataLoader
    from core.signal_processing import SignalProcessor
    from core.hrv_processor import HRVProcessor, HRVDomain
    from core.intelligent_cache import HRVResultsCache
    from core.async_processor import SafeAsyncProcessor, ProgressTracker
    from visualization.interactive_plots import InteractivePlotter
    from stats.advanced_statistics import AdvancedStats
    from ml_analysis.clustering import HRVClustering
    from ml_analysis.forecasting import HRVForecasting

    from gui.settings_panel import SettingsPanel
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import paths...")
    # Fallback to absolute imports
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from enhanced_hrv_analysis.core.data_loader import DataLoader
    from enhanced_hrv_analysis.core.signal_processing import SignalProcessor
    from enhanced_hrv_analysis.core.hrv_processor import HRVProcessor, HRVDomain
    from enhanced_hrv_analysis.core.intelligent_cache import HRVResultsCache
    from enhanced_hrv_analysis.core.async_processor import SafeAsyncProcessor, ProgressTracker
    from enhanced_hrv_analysis.visualization.interactive_plots import InteractivePlotter
    from enhanced_hrv_analysis.stats.advanced_statistics import AdvancedStats
    from enhanced_hrv_analysis.ml_analysis.clustering import HRVClustering
    from enhanced_hrv_analysis.ml_analysis.forecasting import HRVForecasting

    from enhanced_hrv_analysis.gui.settings_panel import SettingsPanel

# Import HRV explanations module
try:
    from gui.hrv_metrics_explanations import show_hrv_explanations
    from gui.citation_display import show_hrv_citations
except ImportError:
    try:
        from enhanced_hrv_analysis.gui.hrv_metrics_explanations import show_hrv_explanations
        from enhanced_hrv_analysis.gui.citation_display import show_hrv_citations
    except ImportError:
        show_hrv_explanations = None
        show_hrv_citations = None

# Import mission phases boxplot generator
try:
    from visualization.mission_phases_boxplots import MissionPhasesBoxplotGenerator
except ImportError:
    try:
        from enhanced_hrv_analysis.visualization.mission_phases_boxplots import MissionPhasesBoxplotGenerator
    except ImportError:
        MissionPhasesBoxplotGenerator = None

logger = logging.getLogger(__name__)

class HRVAnalysisApp:
    """Main application class for enhanced HRV analysis GUI."""
    
    def __init__(self, root: tk.Tk, progress_callback=None):
        """
        Initialize the HRV Analysis application.
        
        Args:
            root: Main Tkinter root window
            progress_callback: Optional callback function for initialization progress updates
        """
        self.root = root
        self.root.title("Enhanced HRV Analysis System - Valquiria Dataset")
        self.root.geometry("1200x800")
        
        # Store progress callback for initialization updates
        self.progress_callback = progress_callback
        
        # Track GUI connection for background processing
        self._gui_active = True
        self._shutdown_in_progress = False
        
        # Setup window event handlers for background processing control
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_closing)
        self.root.bind("<FocusIn>", self._on_window_focus)
        self.root.bind("<FocusOut>", self._on_window_unfocus)
        
        self._update_init_progress(5, "Initializing data loaders...")
        
        # Initialize components
        self.data_loader = DataLoader()
        self.signal_processor = SignalProcessor()
        # PERFORMANCE FIX: Disable parallel processing and reduce bootstrap samples to prevent hanging
        self.hrv_processor = HRVProcessor(parallel_processing=False, n_jobs=1, confidence_level=0.95)
        
        self._update_init_progress(10, "Setting up cache system...")
        
        # Initialize intelligent caching system
        cache_dir = Path(__file__).parent.parent / "hrv_cache"
        self.results_cache = HRVResultsCache(
            cache_dir=cache_dir,
            max_memory_mb=500,  # 500MB cache limit
            max_entries=1000,   # Maximum 1000 cached entries
            default_ttl_hours=24.0  # 24-hour cache expiry
        )
        
        self._update_init_progress(15, "Initializing async processor...")
        
        # TEMPORARILY DISABLED: Async processor to eliminate threading issues
        self.async_processor = None
        logger.info("Async processing disabled to prevent threading issues")
        
        self._update_init_progress(20, "Loading settings...")
        
        # Initialize settings panel
        self.settings_panel = SettingsPanel(
            parent_window=root,
            settings_file=str(Path(__file__).parent.parent / "hrv_analysis_settings.json"),
            on_settings_changed=self._on_settings_changed
        )
        
        # Load and apply settings
        self._apply_settings(self.settings_panel.get_settings())
        
        self._update_init_progress(30, "Initializing analysis components...")
        
        self.interactive_plotter = InteractivePlotter()
        self.advanced_stats = AdvancedStats(n_bootstrap=50)  # Reduced from 1000
        self.hrv_clustering = HRVClustering()
        self.hrv_forecasting = HRVForecasting()
        
        # Initialize mission phases boxplot generator
        if MissionPhasesBoxplotGenerator:
            self.mission_phases_generator = MissionPhasesBoxplotGenerator()
        else:
            self.mission_phases_generator = None
            logger.warning("Mission phases boxplot generator not available")
        
        # Application state
        self.loaded_data = None
        self.processed_data = None
        self.analysis_results = {}
        self.current_subject = None
        self.analysis_running = False
        self.analysis_timeout = 300  # 5 minute timeout for analysis
        self.current_analysis_tasks = []  # Track async analysis tasks
        
        # Performance monitoring
        self.performance_monitor = None
        
        # Performance optimization flags
        self.fast_mode = False  # Analyze all subjects by default
        self.max_bootstrap_samples = 50  # Limit bootstrap samples
        
        # Set default data paths - Updated to point to root /Data folder
        self.data_directory = Path(__file__).parent.parent.parent.parent.parent / "Data"  # Points to root /Data folder
        self.default_db_path = self.data_directory / "merged_data.db"
        self.default_csv_path = self.data_directory
        
        self._update_init_progress(40, "Setting up user interface...")
        
        # Setup GUI
        self._setup_theme()
        self._setup_layout()
        self._setup_menu()
        self._setup_status_bar()
        
        # Center window
        self._center_window()
        
        self._update_init_progress(60, "Loading Valquiria dataset...")
        
        # Auto-load all subject data on startup
        self._auto_load_valquiria_data()
        
        self._update_init_progress(100, "Initialization complete!")
        
        logger.info("HRV Analysis application initialized with intelligent caching enabled")
        
    def _setup_theme(self):
        """Setup modern theme and styling."""
        try:
            # Try to use modern themes
            style = ttk.Style()
            
            # Available themes
            available_themes = style.theme_names()
            
            # Preferred themes in order
            preferred_themes = ['clam', 'alt', 'vista', 'xpnative']
            
            selected_theme = 'default'
            for theme in preferred_themes:
                if theme in available_themes:
                    selected_theme = theme
                    break
                    
            style.theme_use(selected_theme)
            
            # Custom styling
            style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
            style.configure('Subtitle.TLabel', font=('Helvetica', 11, 'italic'), foreground='#4A5568')
            style.configure('Caption.TLabel', font=('Helvetica', 10), foreground='#6A737D')
            style.configure('Heading.TLabel', font=('Helvetica', 12, 'bold'))
            style.configure('Status.TLabel', font=('Helvetica', 10))
            style.configure('Warning.TLabel', font=('Helvetica', 9), foreground='#D69E2E', background='#FFFBF0')
            
            # Button styles
            style.configure('Primary.TButton', font=('Helvetica', 10, 'bold'))
            style.configure('Secondary.TButton', font=('Helvetica', 9))
            
            logger.info(f"Applied theme: {selected_theme}")
            
        except Exception as e:
            logger.warning(f"Error setting up theme: {e}")
            
    def _setup_layout(self):
        """Setup the main application layout."""
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Main container frame
        main_container = ttk.Frame(self.root)
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=2)
        main_container.rowconfigure(0, weight=1)
        
        # Left panel for controls
        self.left_panel = ttk.Frame(main_container, padding="5")
        self.left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Right panel for results  
        self.right_panel = ttk.Frame(main_container, padding="5")
        self.right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Setup individual sections
        self._setup_data_section()
        self._setup_analysis_section() 
        self._setup_control_section()
        self._setup_export_section()
        
        # Add performance monitor if enabled in settings
        if self.settings_panel.get_settings().get('ui_show_performance_monitor', True):
            self._setup_performance_monitor()
        
        self._setup_right_panel()
        
    def _setup_data_section(self):
        """Setup data information display (data is auto-loaded)."""
        # Title
        title_label = ttk.Label(self.left_panel, text="Enhanced HRV Analysis", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 15), sticky=tk.W)
        
        # Data Loading Frame
        data_frame = ttk.LabelFrame(self.left_panel, text="Valquiria Dataset", padding="5")
        data_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        data_frame.columnconfigure(0, weight=1)
        
        # Status label for data loading
        self.data_status_label = ttk.Label(data_frame, text="Loading Valquiria dataset...", 
                                          style='Heading.TLabel')
        self.data_status_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Data info display
        self.data_info_text = tk.Text(data_frame, height=6, width=50, wrap=tk.WORD, state=tk.DISABLED)
        self.data_info_text.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Subject selection
        ttk.Label(data_frame, text="Analyze Subject:", style='Heading.TLabel').grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
        self.subject_var = tk.StringVar()
        self.subject_combo = ttk.Combobox(data_frame, textvariable=self.subject_var, 
                                         state='readonly', font=('Helvetica', 10))
        self.subject_combo.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        self.subject_combo.bind('<<ComboboxSelected>>', self._on_subject_selected)
        
        # Refresh data button (hidden initially, can be shown if needed)
        self.refresh_button = ttk.Button(data_frame, text="Refresh Data", 
                                        command=self._auto_load_valquiria_data,
                                        style='Secondary.TButton')
        # Don't grid it initially - can be added later if needed
        
    def _setup_analysis_section(self):
        """Setup analysis configuration controls."""
        # Analysis Configuration Frame
        config_frame = ttk.LabelFrame(self.left_panel, text="Analysis Configuration", padding="5")
        config_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # HRV Domains selection
        ttk.Label(config_frame, text="HRV Domains:", style='Heading.TLabel').grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        self.domain_vars = {}
        domains = [('Time Domain', HRVDomain.TIME, True),
                  ('Frequency Domain', HRVDomain.FREQUENCY, True), 
                  ('⚠️ Nonlinear (SLOW)', HRVDomain.NONLINEAR, False),
                  ('Parasympathetic', HRVDomain.PARASYMPATHETIC, True),
                  ('Sympathetic', HRVDomain.SYMPATHETIC, True)]
        
        for i, (label, domain, default_enabled) in enumerate(domains):
            var = tk.BooleanVar(value=default_enabled)  # Nonlinear disabled by default
            self.domain_vars[domain] = var
            
            # Add special styling for nonlinear analysis
            if domain == HRVDomain.NONLINEAR:
                checkbox = ttk.Checkbutton(config_frame, text=label, variable=var)
                checkbox.grid(row=i+1, column=0, sticky=tk.W, padx=10)
                
                # Add warning label
                warning_label = ttk.Label(config_frame, 
                                        text="⚠️ WARNING: Very slow! May take 5-10 minutes per subject",
                                        font=('Arial', 8), 
                                        foreground='red')
                warning_label.grid(row=i+1, column=1, sticky=tk.W, padx=(10, 0))
            else:
                ttk.Checkbutton(config_frame, text=label, variable=var).grid(
                    row=i+1, column=0, sticky=tk.W, padx=10)
        
        # Add processing time warning
        warning_frame = ttk.Frame(config_frame)
        warning_frame.grid(row=len(domains)+1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        warning_text = "⚠️ NOTE: Nonlinear analysis (DFA, entropy) may take several minutes.\n" \
                      "Monitor progress via async status and progress bar below."
        warning_label = ttk.Label(warning_frame, text=warning_text, 
                                 style='Warning.TLabel', justify=tk.LEFT)
        warning_label.grid(row=0, column=0, sticky=tk.W)
        
        # Advanced Options
        ttk.Label(config_frame, text="Advanced Options:", 
                 style='Heading.TLabel').grid(row=len(domains)+2, column=0, 
                                            columnspan=2, sticky=tk.W, pady=(10, 5))
        
        # PERFORMANCE FIX: Disable bootstrap by default to prevent hanging
        self.bootstrap_ci_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(config_frame, text="Bootstrap Confidence Intervals (Slow)", 
                       variable=self.bootstrap_ci_var).grid(
            row=len(domains)+3, column=0, sticky=tk.W, padx=10)
        
        self.fast_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(config_frame, text="Limit Data Size (prevents hanging)", 
                       variable=self.fast_mode_var).grid(
            row=len(domains)+4, column=0, sticky=tk.W, padx=10)
        
        self.clustering_var = tk.BooleanVar(value=False)  
        ttk.Checkbutton(config_frame, text="Perform Clustering Analysis (Slow)",
                       variable=self.clustering_var).grid(
            row=len(domains)+5, column=0, sticky=tk.W, padx=10)
        
        self.forecasting_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(config_frame, text="Time Series Forecasting (Slow)",
                       variable=self.forecasting_var).grid(
            row=len(domains)+6, column=0, sticky=tk.W, padx=10)
            
    def _setup_control_section(self):
        """Setup processing controls with enhanced async status visibility."""
        # Processing Controls Frame
        control_frame = ttk.LabelFrame(self.left_panel, text="Processing Controls", padding="5")
        control_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Process buttons frame
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Analysis button (synchronous only)
        self.simple_process_button = ttk.Button(button_frame, text="Run HRV Analysis", 
                                               command=self._run_simple_analysis,
                                               style='Primary.TButton')
        self.simple_process_button.grid(row=0, column=0, sticky=tk.W)
        
        # Advanced Analysis button (DISABLED due to threading issues)
        self.process_button = ttk.Button(button_frame, text="Advanced Analysis (Disabled)", 
                                        command=self._show_advanced_disabled_message,
                                        style='Secondary.TButton',
                                        state='disabled')
        self.process_button.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Analysis help label
        help_label = ttk.Label(control_frame, 
                              text="💡 HRV Analysis: Runs while GUI is open - Real-time progress updates\n" +
                                   "⚠️  Analysis stops if you close the application (no background processing)\n" +
                                   "🧠 Nonlinear analysis disabled by default (enable if needed, but very slow)",
                              font=('Arial', 8),
                              foreground='gray')
        help_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(20, 0))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var,
                                          mode='determinate', length=200)
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(0, 0), pady=5)
        control_frame.columnconfigure(1, weight=1)
        
        # Async Status Display (NEW)
        status_frame = ttk.LabelFrame(control_frame, text="Processing Status", padding="3")
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 5))
        
        # Status text label
        self.async_status_var = tk.StringVar(value="Ready")
        self.async_status_label = ttk.Label(status_frame, textvariable=self.async_status_var, 
                                           style='Status.TLabel')
        self.async_status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Progress details
        self.progress_details_var = tk.StringVar(value="Ready to start analysis...")
        self.progress_details_label = ttk.Label(status_frame, textvariable=self.progress_details_var,
                                              style='Caption.TLabel')
        self.progress_details_label.grid(row=1, column=0, sticky=tk.W)
        
        # Overall progress display
        overall_frame = ttk.Frame(status_frame)
        overall_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Label(overall_frame, text="Overall Progress:", font=('Arial', 8, 'bold')).grid(row=0, column=0, sticky=tk.W)
        self.overall_progress_var = tk.StringVar(value="0/0 subjects completed")
        self.overall_progress_label = ttk.Label(overall_frame, textvariable=self.overall_progress_var,
                                               font=('Arial', 8), foreground='blue')
        self.overall_progress_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Clear button
        ttk.Button(control_frame, text="Clear Results", 
                  command=self._clear_results,
                  style='Secondary.TButton').grid(row=2, column=0, sticky=tk.W, pady=5)
                  
    def _setup_export_section(self):
        """Setup export controls.""" 
        # Export Frame
        export_frame = ttk.LabelFrame(self.left_panel, text="Export", padding="5")
        export_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(export_frame, text="Export Results", 
                  command=self._export_results,
                  style='Secondary.TButton').grid(row=0, column=0, sticky=tk.W, pady=2)
        
        ttk.Button(export_frame, text="Export Plots",
                  command=self._export_plots, 
                  style='Secondary.TButton').grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Button(export_frame, text="Generate Report",
                  command=self._generate_report,
                  style='Secondary.TButton').grid(row=1, column=0, sticky=tk.W, pady=2)
                  
    def _setup_right_panel(self):
        """Setup the right results panel.""" 
        # Create notebook for tabbed interface
        self.results_notebook = ttk.Notebook(self.right_panel)
        self.results_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.right_panel.columnconfigure(0, weight=1)
        self.right_panel.rowconfigure(0, weight=1)
        
        # Results tab
        self.results_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.results_frame, text="Analysis Results")
        
        # Plots tab
        self.plots_frame = ttk.Frame(self.results_notebook)  
        self.results_notebook.add(self.plots_frame, text="Visualizations")
        
        # Statistics tab
        self.stats_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.stats_frame, text="Statistics")
        
        # Setup individual tabs
        self._setup_results_tab()
        self._setup_plots_tab()
        self._setup_stats_tab()
        
    def _setup_results_tab(self):
        """Setup the results display tab."""
        # Results text area with scrollbar
        self.results_text = scrolledtext.ScrolledText(self.results_frame, 
                                                     wrap=tk.WORD, 
                                                     width=80, height=30)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(0, weight=1)
        
    def _setup_plots_tab(self):
        """Setup the modern, interactive plots tab."""
        # Main frame with a professional background color
        main_frame = ttk.Frame(self.plots_frame, style='Plots.TFrame')
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.plots_frame.columnconfigure(0, weight=1)
        self.plots_frame.rowconfigure(0, weight=1)

        # Configure styles
        style = ttk.Style()
        style.configure('Plots.TFrame', background='#F0F4F8')
        style.configure('PlotButton.TButton', font=('Helvetica', 11, 'bold'), padding=10)
        style.configure('Header.TLabel', font=('Helvetica', 16, 'bold'), background='#F0F4F8')
        style.configure('SubHeader.TLabel', font=('Helvetica', 10), background='#F0F4F8', foreground='#555')

        # Header section
        header_frame = ttk.Frame(main_frame, style='Plots.TFrame')
        header_frame.pack(pady=(20, 10), padx=20, fill=tk.X)

        ttk.Label(header_frame, text="HRV Visualizations", style='Header.TLabel').pack(anchor=tk.W)
        ttk.Label(header_frame, text="Select a subject and generate interactive plots.", style='SubHeader.TLabel').pack(anchor=tk.W)

        # Controls section
        controls_frame = ttk.Frame(main_frame, style='Plots.TFrame')
        controls_frame.pack(pady=10, padx=20, fill=tk.X)

        # Subject selection
        subject_frame = ttk.Frame(controls_frame, style='Plots.TFrame')
        subject_frame.pack(fill=tk.X)
        ttk.Label(subject_frame, text="Subject:", font=('Helvetica', 10, 'bold'), background='#F0F4F8').pack(side=tk.LEFT, padx=(0, 10))
        self.plot_subject_var = tk.StringVar(master=self.root)
        self.plot_subject_combo = ttk.Combobox(subject_frame, textvariable=self.plot_subject_var, state='readonly', width=40)
        self.plot_subject_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Create plot buttons in a grid layout
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(pady=10, padx=10, fill=tk.X)
        buttons_frame.columnconfigure((0, 1, 2), weight=1)
        
        # Row 1 - Individual plot types
        ttk.Button(buttons_frame, text="Poincaré Plot", 
                  command=self._generate_poincare_plot,
                  style='Accent.TButton').grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        ttk.Button(buttons_frame, text="Power Spectral Density", 
                  command=self._generate_psd_plot,
                  style='Accent.TButton').grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Button(buttons_frame, text="RR Interval Time Series", 
                  command=self._generate_timeseries_plot,
                  style='Accent.TButton').grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        # Row 2 - Advanced analysis types
        ttk.Button(buttons_frame, text="HRV Dashboard", 
                  command=self._generate_all_plots,
                  style='Accent.TButton').grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        ttk.Button(buttons_frame, text="Combined Time Series", 
                  command=self._generate_combined_time_series,
                  style='Accent.TButton').grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Button(buttons_frame, text="Custom Time Series", 
                  command=self._generate_custom_time_series,
                  style='Accent.TButton').grid(row=1, column=2, padx=5, pady=5, sticky="ew")
        
        # Row 3 - GAM Analysis (New)
        ttk.Button(buttons_frame, text="GAM Crew Analysis", 
                  command=self._generate_gam_crew_analysis,
                  style='Info.TButton').grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        ttk.Button(buttons_frame, text="GAM Custom Metrics", 
                  command=self._generate_gam_custom_analysis,
                  style='Info.TButton').grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        # Row 3 - Mission Phases Boxplots (New Feature)
        ttk.Button(buttons_frame, text="Mission Phases - Individual", 
                  command=self._generate_individual_mission_phases,
                  style='Success.TButton').grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        
        ttk.Button(buttons_frame, text="Mission Phases - Group", 
                  command=self._generate_group_mission_phases,
                  style='Success.TButton').grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Button(buttons_frame, text="Mission Phases - Report", 
                  command=self._generate_mission_phases_report,
                  style='Success.TButton').grid(row=3, column=2, padx=5, pady=5, sticky="ew")
        
        # Description labels
        gam_desc_label = ttk.Label(main_frame,
                                  text="GAM Analysis: Advanced statistical modeling with trend lines and confidence intervals for crew-wide analysis",
                                  font=('Helvetica', 9),
                                  foreground='#555555')
        gam_desc_label.pack(pady=(5, 0))
        
        # Mission phases description
        phases_desc_label = ttk.Label(main_frame,
                                     text="Mission Phases: Compare physiological adaptation across Early, Mid, and Late mission phases",
                                     font=('Helvetica', 9),
                                     foreground='#2D5A27')
        phases_desc_label.pack(pady=(2, 0))
        
        # Status display
        self.plot_status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        self.plot_status_frame.pack(pady=10, padx=20, fill=tk.X)

        self.plot_status_label = ttk.Label(self.plot_status_frame, text="Ready to generate plots.", wraplength=700)
        self.plot_status_label.pack(fill=tk.X)
    
    def _set_plot_preview_instructions(self):
        """Set instructions in the plot preview area."""
        pass
    
    def _update_plot_preview(self, plot_type, subject, file_path, rr_intervals):
        """Update the plot preview area with plot information."""
        pass
    
    def _setup_stats_tab(self):
        """Setup the statistics display tab."""
        # Statistics text area
        self.stats_text = scrolledtext.ScrolledText(self.stats_frame,
                                                   wrap=tk.WORD,
                                                   width=80, height=30)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.stats_frame.columnconfigure(0, weight=1) 
        self.stats_frame.rowconfigure(0, weight=1)
        
    def _setup_menu(self):
        """Setup application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Refresh Valquiria Data", command=self._auto_load_valquiria_data)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results...", command=self._export_results)
        file_menu.add_command(label="Export Plots...", command=self._export_plots)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run HRV Analysis", command=self._run_analysis)
        analysis_menu.add_command(label="Clear Results", command=self._clear_results)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        # Removed Settings menu item - was causing errors and showing no values
        tools_menu.add_command(label="Performance Monitor", command=self._toggle_performance_monitor)
        tools_menu.add_command(label="Clear Cache", command=self._clear_cache)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="HRV Metrics Explained", command=self._show_hrv_explanations)
        help_menu.add_command(label="Reference Ranges & Citations", command=self._show_hrv_citations)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)
        
    def _setup_status_bar(self):
        """Setup status bar."""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var,
                                     style='Status.TLabel')
        self.status_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
    def _center_window(self):
        """Center the application window on screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
    def _update_status(self, message: str):
        """Update status bar and async status display with thread safety."""
        if not self._gui_active:
            return
            
        try:
            self.status_var.set(message)
            
            # Also update async status display if it exists
            if hasattr(self, 'async_status_var'):
                self.async_status_var.set(message)
                
            self.root.update_idletasks()
        except Exception as e:
            if "main thread is not in main loop" in str(e) or "invalid command name" in str(e):
                logger.warning("Status update failed - main thread unavailable")
                self._gui_active = False
            else:
                logger.warning(f"Status update error: {e}")
        
    def _update_progress(self, value: float, message: str = ""):
        """Update progress bar, status, and progress details with thread safety."""
        if not self._gui_active:
            return
            
        def update():
            try:
                if self._gui_active and hasattr(self, 'progress_var'):
                    self.progress_var.set(value)
                    if message:
                        self._update_status(message)
                        
                        # Also update progress details if available
                        if hasattr(self, 'progress_details_var'):
                            # Add percentage to details if meaningful
                            if value > 0:
                                detail_msg = f"{message} ({value:.0f}%)"
                            else:
                                detail_msg = message
                            self.progress_details_var.set(detail_msg)
            except Exception as e:
                if "main thread is not in main loop" in str(e) or "invalid command name" in str(e):
                    logger.warning("Progress update failed - main thread unavailable")
                    self._gui_active = False
                else:
                    logger.warning(f"Progress update error: {e}")
        
        try:
            self.root.after(0, update)
        except Exception as e:
            if "main thread is not in main loop" in str(e):
                logger.warning("Cannot schedule progress update - main thread unavailable")
                self._gui_active = False
            else:
                logger.warning(f"Scheduling progress update error: {e}")
    
    def _update_init_progress(self, percentage: float, message: str):
        """Update progress during initialization if callback is available."""
        if self.progress_callback:
            self.progress_callback(percentage, message)
    
    def _on_settings_changed(self, new_settings: Dict[str, Any]):
        """Handle settings changes."""
        try:
            self._apply_settings(new_settings)
            logger.info("Settings applied successfully")
        except Exception as e:
            logger.error(f"Error applying settings: {e}")
            messagebox.showerror("Settings Error", f"Error applying settings: {e}")
    
    def _apply_settings(self, settings: Dict[str, Any]):
        """Apply settings to the application components."""
        try:
            # Update cache settings
            if hasattr(self, 'results_cache'):
                # Cache settings are applied at initialization, would need cache restart for changes
                pass
            
            # Update analysis settings
            self.analysis_timeout = settings.get('analysis_timeout_seconds', 300.0)
            logger.info("Analysis will run only while GUI is open (no background processing)")
            
            # Update analysis settings
            if hasattr(self, 'advanced_stats'):
                bootstrap_samples = settings.get('analysis_max_bootstrap_samples', 50)
                # Would need to reinitialize advanced_stats for this change
            
            # Update UI settings
            if settings.get('analysis_fast_mode_default', False):
                self.fast_mode_var.set(True)
            
            # Update performance monitor
            if hasattr(self, 'performance_monitor') and self.performance_monitor:
                self.performance_monitor.update_interval = settings.get('monitor_update_interval', 2.0)
                self.performance_monitor.max_history = settings.get('monitor_max_history', 60)
            
            # Check for pending notifications from background processing
            self._check_pending_notifications()
            
        except Exception as e:
            logger.error(f"Error in _apply_settings: {e}")
    
    def _check_pending_notifications(self):
        """Check for and display pending notifications from background processing."""
        try:
            if hasattr(self, 'async_processor'):
                notifications = self.async_processor.get_pending_notifications()
                
                for notification in notifications:
                    task_id = notification.get('task_id', 'Unknown')
                    success = notification.get('success', False)
                    message = notification.get('message', 'Task completed')
                    timestamp = notification.get('timestamp', 0)
                    
                    # Format timestamp
                    import datetime
                    time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                    
                    # Show notification to user
                    if success:
                        messagebox.showinfo(
                            "Background Analysis Complete", 
                            f"HRV analysis completed in background at {time_str}!\n\n"
                            f"Task: {task_id}\n"
                            f"Status: {message}\n\n"
                            f"Results have been automatically saved and cached."
                        )
                        logger.info(f"Displayed success notification for background task {task_id}")
                    else:
                        error_msg = notification.get('error', 'Unknown error')
                        messagebox.showwarning(
                            "Background Analysis Issue",
                            f"Background analysis encountered an issue at {time_str}.\n\n"
                            f"Task: {task_id}\n"
                            f"Error: {error_msg}\n\n"
                            f"Partial results may have been saved for recovery."
                        )
                        logger.warning(f"Displayed error notification for background task {task_id}")
                
                if notifications:
                    # Refresh results display in case we have new data
                    self._refresh_analysis_results()
                    
        except Exception as e:
            logger.error(f"Error checking pending notifications: {e}")
    
    def _refresh_analysis_results(self):
        """Refresh analysis results from persisted data."""
        try:
            if hasattr(self, 'async_processor'):
                persisted_tasks = self.async_processor.get_persisted_tasks()
                
                if persisted_tasks:
                    logger.info(f"Found {len(persisted_tasks)} persisted results from background processing")
                    
                    # Try to load and merge persisted results
                    merged_results = {}
                    for task_id in persisted_tasks:
                        result = self.async_processor._load_persisted_result(task_id)
                        if result and isinstance(result, dict):
                            merged_results.update(result)
                    
                    if merged_results:
                        # Merge with existing results
                        if self.analysis_results:
                            self.analysis_results.update(merged_results)
                        else:
                            self.analysis_results = merged_results
                        
                        # Update displays
                        self._update_results_display()
                        self._update_plots_display()
                        
                        logger.info("Successfully loaded persisted analysis results")
                        
        except Exception as e:
            logger.error(f"Error refreshing analysis results: {e}")
    
    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics for performance monitoring."""
        try:
            if hasattr(self, 'results_cache'):
                stats = self.results_cache.get_cache_stats()
                
                # Add additional derived metrics
                stats['cache_efficiency'] = 'Excellent' if stats.get('hit_ratio', 0) > 0.8 else \
                                          'Good' if stats.get('hit_ratio', 0) > 0.6 else \
                                          'Fair' if stats.get('hit_ratio', 0) > 0.4 else 'Poor'
                
                stats['memory_utilization'] = stats.get('memory_usage_mb', 0) / stats.get('memory_limit_mb', 1)
                
                # Format compression stats for display
                compression_stats = stats.get('compression_stats', {})
                total_compressions = sum(compression_stats.values())
                if total_compressions > 0:
                    stats['compression_breakdown'] = {
                        method: f"{count} ({count/total_compressions:.1%})" 
                        for method, count in compression_stats.items()
                    }
                
                return stats
            return {}
        except Exception as e:
            logger.warning(f"Error getting comprehensive cache stats: {e}")
            return {}
    
    def _get_async_processor_stats(self) -> Dict[str, Any]:
        """Get detailed async processor statistics."""
        try:
            if hasattr(self, 'async_processor'):
                base_stats = self.async_processor.get_progress_info()
                
                # Add performance calculations
                if hasattr(self.async_processor, '_task_history'):
                    completed_tasks = [t for t in self.async_processor._task_history if t.get('status') == 'completed']
                    if completed_tasks:
                        avg_duration = sum(t.get('duration', 0) for t in completed_tasks) / len(completed_tasks)
                        base_stats['average_task_duration'] = avg_duration
                        base_stats['total_completed_tasks'] = len(completed_tasks)
                
                return base_stats
            return {}
        except Exception as e:
            logger.warning(f"Error getting async processor stats: {e}")
            return {}

    def _get_data_loader_stats(self) -> Dict[str, Any]:
        """Get data loader performance statistics."""
        try:
            if hasattr(self, 'data_loader') and hasattr(self.data_loader, 'optimized_loader'):
                if self.data_loader.optimized_loader:
                    return self.data_loader.optimized_loader.get_performance_stats()
            return {}
        except Exception as e:
            logger.warning(f"Error getting data loader stats: {e}")
            return {}

    def _show_advanced_cache_stats(self):
        """Show detailed cache performance statistics in a new window."""
        try:
            cache_stats = self._get_cache_stats()
            if not cache_stats:
                messagebox.showinfo("Cache Statistics", "No cache statistics available")
                return
            
            # Create new window for detailed stats
            stats_window = tk.Toplevel(self.root)
            stats_window.title("Advanced Cache Performance Statistics")
            stats_window.geometry("800x600")
            
            # Create scrollable text widget
            text_frame = ttk.Frame(stats_window)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 10))
            scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Generate comprehensive report
            if hasattr(self.results_cache, 'get_performance_report'):
                report = self.results_cache.get_performance_report()
            else:
                report = self._generate_cache_stats_report(cache_stats)
            
            text_widget.insert(tk.END, report)
            text_widget.configure(state=tk.DISABLED)
            
            # Add refresh button
            button_frame = ttk.Frame(stats_window)
            button_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Button(button_frame, text="Refresh", 
                      command=lambda: self._refresh_stats_window(text_widget, cache_stats)).pack(side=tk.LEFT)
            ttk.Button(button_frame, text="Export Report", 
                      command=lambda: self._export_performance_report(report)).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Close", 
                      command=stats_window.destroy).pack(side=tk.RIGHT)
            
        except Exception as e:
            logger.error(f"Error showing advanced cache stats: {e}")
            messagebox.showerror("Error", f"Failed to show cache statistics: {e}")

    def _generate_cache_stats_report(self, stats: Dict[str, Any]) -> str:
        """Generate a comprehensive cache statistics report."""
        report = f"""
=== Enhanced HRV Cache Performance Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 CACHE EFFICIENCY METRICS:
• Cache Hit Ratio: {stats.get('hit_ratio', 0):.1%} ({stats.get('cache_efficiency', 'Unknown')})
• Total Requests: {stats.get('total_requests', 0):,}
• Cache Hits: {stats.get('cache_hits', 0):,}
• Cache Misses: {stats.get('cache_misses', 0):,}
• Average Load Time: {stats.get('average_load_time_ms', 0):.1f}ms

💾 MEMORY UTILIZATION:
• Current Usage: {stats.get('memory_usage_mb', 0):.1f}MB / {stats.get('memory_limit_mb', 0):.1f}MB
• Utilization: {stats.get('memory_utilization', 0):.1%}
• Total Entries: {stats.get('total_entries', 0):,}
• Memory-Resident: {stats.get('memory_entries', 0):,}
• Evictions: {stats.get('evictions', 0):,}

🗜️ COMPRESSION ANALYTICS:
• Total Savings: {stats.get('compression_savings_mb', 0):.2f}MB
• Compression Methods:
"""
        
        # Add compression breakdown
        compression_breakdown = stats.get('compression_breakdown', {})
        for method, count_info in compression_breakdown.items():
            report += f"  • {method.upper()}: {count_info}\n"
        
        report += f"""
🚀 PREDICTIVE CACHING:
• Prefetch Candidates: {stats.get('prefetch_candidates', 0)}
• Prefetch Efficiency: {stats.get('prefetch_efficiency', 0):.1%}
• Access Patterns: {stats.get('access_patterns_tracked', 0)} tracked

📈 PERFORMANCE TRENDS:
• Recent Memory Trend: {', '.join(str(m.get('memory_mb', 0)) + 'MB' for m in stats.get('memory_usage_trend', [])[-5:])}

🔧 CACHE CONFIGURATION:
• Cache Directory: {stats.get('cache_dir', 'Unknown')}
• Entry Limit: {stats.get('entry_limit', 0):,}
"""
        return report

    def _refresh_stats_window(self, text_widget: tk.Text, old_stats: Dict[str, Any]):
        """Refresh the statistics window with updated data."""
        try:
            new_stats = self._get_cache_stats()
            report = self._generate_cache_stats_report(new_stats)
            
            text_widget.configure(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, report)
            text_widget.configure(state=tk.DISABLED)
            
        except Exception as e:
            logger.error(f"Error refreshing stats window: {e}")

    def _export_performance_report(self, report: str):
        """Export performance report to file."""
        try:
            from tkinter import filedialog
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Export Performance Report"
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(report)
                messagebox.showinfo("Export Complete", f"Performance report exported to:\n{file_path}")
                
        except Exception as e:
            logger.error(f"Error exporting performance report: {e}")
            messagebox.showerror("Export Error", f"Failed to export report: {e}")

    def _show_database_stats(self):
        """Show database optimization statistics."""
        try:
            db_stats = self._get_data_loader_stats()
            
            if not db_stats:
                messagebox.showinfo("Database Statistics", "No database statistics available.\nLoad data first to see optimization metrics.")
                return
            
            # Create stats window
            stats_window = tk.Toplevel(self.root)
            stats_window.title("Database Optimization Statistics")
            stats_window.geometry("750x500")
            
            # Create text display
            text_frame = ttk.Frame(stats_window)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 10))
            scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Generate database performance report
            if hasattr(self.data_loader, 'optimized_loader') and self.data_loader.optimized_loader:
                try:
                    report = self.data_loader.optimized_loader.get_performance_report()
                except AttributeError:
                    report = self._generate_db_stats_report(db_stats)
            else:
                report = self._generate_db_stats_report(db_stats)
            
            text_widget.insert(tk.END, report)
            text_widget.configure(state=tk.DISABLED)
            
            # Close button
            ttk.Button(stats_window, text="Close", command=stats_window.destroy).pack(pady=5)
            
        except Exception as e:
            logger.error(f"Error showing database stats: {e}")
            messagebox.showerror("Error", f"Failed to show database statistics: {e}")

    def _generate_db_stats_report(self, stats: Dict[str, Any]) -> str:
        """Generate database performance report."""
        return f"""
=== Database Optimization Performance Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 LOADING PERFORMANCE:
• Total Records: {stats.get('total_records', 0):,}
• Loading Speed: {stats.get('records_per_second', 0):.0f} records/sec
• Chunks Processed: {stats.get('chunks_loaded', 0):,}
• Peak Memory: {stats.get('memory_peak_mb', 0):.1f}MB

🔗 CONNECTION POOLING:
• Active Connections: {stats.get('connection_pool_active', 'N/A')}
• Total Connections: {stats.get('connection_pool_total', 'N/A')}
• Connection Requests: {stats.get('connection_requests', 'N/A')}
• Average Wait Time: {stats.get('average_connection_wait_ms', 0):.1f}ms

🚀 QUERY OPTIMIZATION:
• Queries Executed: {stats.get('queries_executed', 0):,}
• Index Hits: {stats.get('index_hits', 0):,}
• Table Scans: {stats.get('table_scans', 0):,}
• Optimizations Applied: {stats.get('optimization_applied', 0):,}

📈 ADAPTIVE FEATURES:
• Adaptive Adjustments: {stats.get('adaptive_adjustments', 0):,}
• Performance Trend: {stats.get('performance_trend_slope', 0):+.0f} rec/s change
• Stability Score: {stats.get('performance_stability', 0):.1%}
"""

    def _clear_all_caches(self):
        """Clear all caches and reset performance counters."""
        try:
            cleared_items = []
            
            # Clear HRV results cache
            if hasattr(self, 'results_cache'):
                stats_before = self.results_cache.get_cache_stats()
                self.results_cache.clear_cache()
                cleared_items.append(f"HRV Cache: {stats_before.get('total_entries', 0)} entries cleared")
            
            # Clear data loader caches
            if hasattr(self, 'data_loader') and hasattr(self.data_loader, 'optimized_loader'):
                if self.data_loader.optimized_loader and hasattr(self.data_loader.optimized_loader, 'query_cache'):
                    if self.data_loader.optimized_loader.query_cache:
                        query_count = len(self.data_loader.optimized_loader.query_cache)
                        self.data_loader.optimized_loader.query_cache.clear()
                        cleared_items.append(f"Query Cache: {query_count} queries cleared")
            
            # Force garbage collection
            gc.collect()
            
            if cleared_items:
                message = "Successfully cleared:\n" + "\n".join(f"• {item}" for item in cleared_items)
                message += "\n\nMemory freed and performance counters reset."
            else:
                message = "No caches found to clear."
            
            messagebox.showinfo("Cache Cleared", message)
            
            # Update status
            self._update_status("All caches cleared - memory freed")
            
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")
            messagebox.showerror("Error", f"Failed to clear caches: {e}")

    def _warm_cache_for_analysis(self):
        """Warm cache with commonly used analysis patterns."""
        try:
            if not hasattr(self, 'results_cache'):
                messagebox.showwarning("Warning", "No cache system available")
                return
            
            if self.loaded_data is None:
                messagebox.showwarning("Warning", "Load data first before warming cache")
                return
            
            # Get unique subjects
            subjects = self.loaded_data['subject'].unique().tolist()
            
            # Common analysis configurations
            common_configs = [
                {'fast_mode': True, 'bootstrap_ci': False},
                {'fast_mode': False, 'bootstrap_ci': True},
                {'fast_mode': False, 'bootstrap_ci': False}
            ]
            
            # Start cache warming
            self.results_cache.warm_cache_for_subjects(subjects, common_configs)
            
            messagebox.showinfo("Cache Warming", 
                               f"Cache warming initiated for {len(subjects)} subjects "
                               f"with {len(common_configs)} analysis configurations.\n\n"
                               "This will improve performance for future analysis runs.")
            
        except Exception as e:
            logger.error(f"Error warming cache: {e}")
            messagebox.showerror("Error", f"Failed to warm cache: {e}")
    
    def _toggle_performance_monitor(self):
        """Toggle performance monitor visibility."""
        if hasattr(self, 'performance_monitor') and self.performance_monitor:
            if self.performance_monitor.monitoring_active:
                self.performance_monitor.stop_monitoring()
                messagebox.showinfo("Performance Monitor", "Performance monitoring stopped")
            else:
                self.performance_monitor.start_monitoring()
                messagebox.showinfo("Performance Monitor", "Performance monitoring started")
        else:
            messagebox.showwarning("Performance Monitor", "Performance monitor not available")
    
    def _clear_cache(self):
        """Clear the intelligent cache."""
        try:
            if hasattr(self, 'results_cache'):
                result = messagebox.askyesno("Clear Cache", 
                                           "Are you sure you want to clear the analysis cache?\n" +
                                           "This will remove all cached results.")
                if result:
                    self.results_cache.clear_cache()
                    messagebox.showinfo("Cache Cleared", "Analysis cache has been cleared successfully")
            else:
                messagebox.showwarning("Cache", "Cache system not available")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            messagebox.showerror("Error", f"Error clearing cache: {e}")
    
    def _auto_load_valquiria_data(self):
        """Automatically load all Valquiria subject CSV files."""
        try:
            self._update_status("Loading Valquiria dataset...")
            self.data_status_label.configure(text="Loading all subject data...")
            self.root.update_idletasks()
            
            # Use the configured data directory (root /Data folder)
            import os
            from pathlib import Path
            
            # Use the pre-configured data directory pointing to root /Data folder
            data_folder = self.data_directory
            
            logger.info(f"Looking for data in: {data_folder}")
            
            if not data_folder.exists():
                # Fallback: try alternative paths to find the root Data folder
                possible_paths = [
                    Path(__file__).parent.parent.parent.parent.parent / "Data",  # Root /Data
                    Path("../../../../Data"),  # Relative path to root /Data
                    Path("../../../../../Data"),  # Alternative relative path
                    Path(r"C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\Data"),  # Windows absolute path
                ]
                
                data_folder = None
                for path in possible_paths:
                    if path.exists() and (path / "merged_data.db").exists():
                        data_folder = path
                        logger.info(f"Found Data folder at: {data_folder}")
                        break
                
                if data_folder is None:
                    raise FileNotFoundError("Cannot locate root /Data folder with merged_data.db")
            
            # Define all Valquiria subject files
            subject_files = [
                "T01_Mara.csv",
                "T02_Laura.csv", 
                "T03_Nancy.csv",
                "T04_Michelle.csv",
                "T05_Felicitas.csv",
                "T06_Mara_Selena.csv",
                "T07_Geraldinn.csv",
                "T08_Karina.csv"
            ]
            
            # Try to load from database first (if available)
            db_path = data_folder / "merged_data.db"
            if db_path.exists():
                logger.info("Loading from Valquiria database...")
                
                # Use optimized loader for large datasets
                try:
                    # Create a progress callback that uses either init progress or regular progress
                    if self.progress_callback:
                        # During initialization, map to 60-95% range
                        progress_fn = lambda msg, pct: self._update_init_progress(60 + (pct * 0.35), msg)
                    else:
                        # During normal operation, use regular progress
                        progress_fn = lambda msg, pct: self._update_progress(pct * 0.8, msg)
                    
                    self.loaded_data = self.data_loader.load_database_data_optimized(
                        str(db_path),
                        progress_callback=progress_fn
                    )
                except Exception as optimized_error:
                    logger.warning(f"Optimized loading failed: {optimized_error}, trying standard loading")
                    self.loaded_data = self.data_loader.load_database_data(str(db_path))
                
                if self.loaded_data is not None and not self.loaded_data.empty:
                    self._update_data_info_valquiria("database")
                    self._update_subject_list()
                    self.data_status_label.configure(text="✅ Valquiria Dataset Loaded (Database - Optimized)")
                    self._update_status("Valquiria dataset loaded from database using optimized loader")
                    logger.info(f"Successfully loaded {len(self.loaded_data):,} records from database")
                    return
            
            # Load from CSV files (fallback if database not available)
            logger.info("Loading from Valquiria CSV files...")
            available_files = []
            for subject_file in subject_files:
                file_path = data_folder / subject_file
                if file_path.exists():
                    available_files.append(str(file_path))
                    logger.info(f"Found: {subject_file}")
                else:
                    logger.warning(f"Subject file not found: {subject_file}")
            
            if not available_files:
                logger.warning("No Valquiria subject CSV files found in root /Data folder")
                # Try to look in the embedded Data folder as additional fallback
                embedded_data_folder = Path(__file__).parent.parent / "Data"
                if embedded_data_folder.exists():
                    logger.info(f"Trying embedded data folder: {embedded_data_folder}")
                    for subject_file in subject_files:
                        file_path = embedded_data_folder / subject_file
                        if file_path.exists():
                            available_files.append(str(file_path))
                            logger.info(f"Found in embedded folder: {subject_file}")
                    
                    if available_files:
                        data_folder = embedded_data_folder  # Use embedded folder for CSV loading
                    else:
                        raise FileNotFoundError("No Valquiria subject CSV files found in any location")
                else:
                    raise FileNotFoundError("No Valquiria subject CSV files found")
            
            # Load all available CSV files
            self.loaded_data = self.data_loader.load_csv_data(csv_files=[Path(f).name for f in available_files], 
                                                             data_dir=str(data_folder))
            
            if self.loaded_data is not None and not self.loaded_data.empty:
                self._update_data_info_valquiria("csv")
                self._update_subject_list()
                self.data_status_label.configure(text="✅ Valquiria Dataset Loaded (CSV Files)")
                self._update_status("Valquiria dataset loaded successfully")
                logger.info(f"Successfully loaded {len(self.loaded_data):,} records from {len(available_files)} CSV files")
            else:
                raise Exception("Failed to load any data from CSV files")
                
        except Exception as e:
            logger.error(f"Error loading Valquiria data: {e}")
            self.data_status_label.configure(text="❌ Failed to load Valquiria dataset")
            
            # Fallback to sample data for demonstration
            try:
                logger.info("Falling back to sample data...")
                self.loaded_data = DataLoader.create_sample_data(n_subjects=8, n_sols=6)
                self._update_data_info_valquiria("sample")
                self._update_subject_list()
                self.data_status_label.configure(text="⚠️ Using Sample Data (Valquiria files not found)")
                self._update_status("Using sample data - Valquiria files not found")
                logger.info("Sample data generated as fallback")
            except Exception as sample_error:
                logger.error(f"Failed to generate sample data: {sample_error}")
                self.data_status_label.configure(text="❌ Data loading failed")
                self._update_status("Data loading failed")
                
    def _use_sample_data(self):
        """Use sample data for demonstration - kept for compatibility."""
        try:
            self._update_status("Generating sample data...")
            self.loaded_data = DataLoader.create_sample_data(n_subjects=5, n_sols=6)
            
            self._update_data_info()
            self._update_subject_list()
            self._update_status("Sample data loaded")
            messagebox.showinfo("Success", "Sample data generated for demonstration")
            
        except Exception as e:
            logger.error(f"Error generating sample data: {e}")
            messagebox.showerror("Error", f"Error generating sample data: {e}")
            self._update_status("Ready")
            
    def _update_data_info(self):
        """Update data information display."""
        if self.loaded_data is None:
            return
            
        try:
            info_text = f"Dataset Information:\n"
            info_text += f"Total records: {len(self.loaded_data):,}\n"
            
            if 'subject' in self.loaded_data.columns:
                n_subjects = self.loaded_data['subject'].nunique()
                info_text += f"Subjects: {n_subjects}\n"
                
            if 'Sol' in self.loaded_data.columns:
                n_sols = self.loaded_data['Sol'].nunique()
                info_text += f"SOLs: {n_sols}\n"
                
            # Data quality info
            if hasattr(self.data_loader, 'data_quality_metrics'):
                quality = self.data_loader.data_quality_metrics
                info_text += f"HR Quality: {quality.hr_quality_ratio:.1%}\n"
                info_text += f"Mean HR: {quality.mean_hr:.1f} BPM\n"
                
            self.data_info_text.configure(state=tk.NORMAL)
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(tk.END, info_text)
            self.data_info_text.configure(state=tk.DISABLED)
            
        except Exception as e:
            logger.error(f"Error updating data info: {e}")
            
    def _update_data_info_valquiria(self, source_type: str):
        """Update data information display with Valquiria-specific details."""
        if self.loaded_data is None:
            return
            
        try:
            # Build comprehensive info text
            info_text = "🚀 VALQUIRIA SPACE ANALOG SIMULATION\n"
            info_text += "="*45 + "\n\n"
            
            info_text += f"Data Source: {source_type.upper()}\n"
            info_text += f"Total Records: {len(self.loaded_data):,}\n"
            
            if 'subject' in self.loaded_data.columns:
                subjects = sorted(self.loaded_data['subject'].unique())
                info_text += f"Subjects: {len(subjects)}\n"
                
                # Show subject details
                for subject in subjects:
                    subject_data = self.loaded_data[self.loaded_data['subject'] == subject]
                    if 'Sol' in subject_data.columns:
                        sols = sorted(subject_data['Sol'].unique())
                        info_text += f"  • {subject}: Sol {min(sols)}-{max(sols)} ({len(subject_data):,} records)\n"
                    else:
                        info_text += f"  • {subject}: {len(subject_data):,} records\n"
                
            if 'Sol' in self.loaded_data.columns:
                sol_range = (self.loaded_data['Sol'].min(), self.loaded_data['Sol'].max())
                info_text += f"\nSOL Range: {sol_range[0]} to {sol_range[1]}\n"
                
            # Data quality info
            if hasattr(self.data_loader, 'data_quality_metrics') and self.data_loader.data_quality_metrics:
                quality = self.data_loader.data_quality_metrics
                info_text += f"\nDATA QUALITY METRICS:\n"
                info_text += f"HR Quality: {quality.hr_quality_ratio:.1%}\n"
                info_text += f"Mean HR: {quality.mean_hr:.1f} ± {quality.std_hr:.1f} BPM\n"
                info_text += f"HR Range: {quality.hr_range[0]:.1f} - {quality.hr_range[1]:.1f} BPM\n"
                info_text += f"Recording Coverage: {quality.temporal_coverage_hours:.1f} hours\n"
                
                # Quality assessment
                if quality.hr_quality_ratio >= 0.9:
                    info_text += "Status: ✅ EXCELLENT data quality\n"
                elif quality.hr_quality_ratio >= 0.8:
                    info_text += "Status: ✅ GOOD data quality\n"
                elif quality.hr_quality_ratio >= 0.7:
                    info_text += "Status: ⚠️ MODERATE data quality\n"
                else:
                    info_text += "Status: ⚠️ Poor data quality\n"
            
            info_text += f"\n🎯 Ready for HRV Analysis!"
                
            # Update the text widget
            self.data_info_text.configure(state=tk.NORMAL)
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(tk.END, info_text)
            self.data_info_text.configure(state=tk.DISABLED)
            
        except Exception as e:
            logger.error(f"Error updating Valquiria data info: {e}")
            # Fallback to basic info
            self._update_data_info()
            
    def _update_subject_list(self):
        """Update subject selection combobox.""" 
        if self.loaded_data is None:
            return
            
        try:
            if 'subject' in self.loaded_data.columns:
                subjects = ['All'] + sorted(self.loaded_data['subject'].unique().tolist())
                self.subject_combo['values'] = subjects
                self.subject_combo.set('All')
            else:
                self.subject_combo['values'] = ['All']
                self.subject_combo.set('All')
                
        except Exception as e:
            logger.error(f"Error updating subject list: {e}")
            
    def _on_subject_selected(self, event=None):
        """Handle subject selection."""
        self.current_subject = self.subject_var.get()
        logger.info(f"Selected subject: {self.current_subject}")
        
    def _run_analysis(self):
        """Run HRV analysis (DISABLED - redirects to simple analysis)."""
        messagebox.showinfo(
            "Advanced Analysis Disabled", 
            "Advanced Analysis is temporarily disabled.\n\nUsing Simple Analysis instead..."
        )
        self._run_simple_analysis()
    
    def _monitor_analysis_task(self, task_id: str):
        """Monitor async analysis task progress with enhanced status messaging."""
        try:
            task_status = self.async_processor.get_task_status(task_id)
            
            if task_status is None:
                # Task not found, something went wrong
                self.analysis_running = False
                self.process_button.configure(state='normal')
                self._update_status("Analysis task lost")
                messagebox.showerror("Error", "Analysis task lost - this shouldn't happen with the new optimizations!")
                return
            
            if task_status.value == "running":
                # Provide more informative status based on elapsed time
                elapsed_time = time.time() - getattr(self, '_analysis_start_time', time.time())
                
                if elapsed_time < 30:
                    status_msg = "Processing subjects... (Fast analysis phase)"
                elif elapsed_time < 120:
                    status_msg = "Processing nonlinear metrics... (This may take a few minutes)"
                elif elapsed_time < 300:
                    status_msg = "Computing complex metrics (DFA, entropy)... Please wait"
                else:
                    status_msg = "Advanced processing in progress... (Approaching timeout)"
                
                self._update_status(status_msg)
                
                # Task still running, check again in 1 second
                self.root.after(1000, lambda: self._monitor_analysis_task(task_id))
                return
            
            elif task_status.value == "completed":
                # Task completed, get results
                try:
                    results = self.async_processor.get_task_result(task_id)
                    if results:
                        self.analysis_results = results
                        self._update_results_display()
                        self._update_plots_display()
                        self._update_progress(100, "Analysis complete - All metrics computed successfully!")
                        messagebox.showinfo("Success", f"HRV analysis completed successfully!\n\n" +
                                          f"✅ All subjects processed\n" +
                                          f"✅ All HRV domains computed\n" +
                                          f"✅ Results ready for visualization")
                    else:
                        self._update_status("Analysis completed but no results returned")
                        messagebox.showerror("Error", "Analysis completed but no results returned")
                except Exception as e:
                    logger.error(f"Error retrieving analysis results: {e}")
                    messagebox.showerror("Error", f"Error retrieving results: {e}")
                    
            elif task_status.value == "failed":
                # Task failed
                try:
                    self.async_processor.get_task_result(task_id)  # This will raise the exception
                except Exception as e:
                    logger.error(f"Analysis task failed: {e}")
                    messagebox.showerror("Error", f"Analysis failed: {e}")
                    
            elif task_status.value == "timeout":
                messagebox.showerror("Error", "Analysis timed out")
                
            else:
                messagebox.showerror("Error", f"Analysis task status: {task_status.value}")
            
        except Exception as e:
            logger.error(f"Error monitoring analysis task: {e}")
            messagebox.showerror("Error", f"Monitoring error: {e}")
        
        finally:
            # Cleanup
            self.analysis_running = False
            self.process_button.configure(state='normal')
            self._update_progress(0, "Ready")
            if task_id in self.current_analysis_tasks:
                self.current_analysis_tasks.remove(task_id)
    
    def _run_simple_analysis(self):
        """Run HRV analysis synchronously in the main thread (simple, no threading issues)."""
        if self.analysis_running:
            messagebox.showwarning("Warning", "Analysis is already running")
            return
            
        if self.loaded_data is None:
            messagebox.showwarning("Warning", "No data loaded")
            return
            
        # Set analysis state
        self.analysis_running = True
        self.simple_process_button.configure(state='disabled')
        self.process_button.configure(state='disabled')
        
        try:
            # Capture analysis configuration
            analysis_config = self._capture_analysis_config()
            selected_domains = analysis_config.get('selected_domains', [])
            
            if not selected_domains:
                messagebox.showwarning("Warning", "No analysis domains selected")
                return
            
            # Check if nonlinear analysis is selected and warn user
            has_nonlinear = any(domain == HRVDomain.NONLINEAR for domain in selected_domains)
            if has_nonlinear:
                result = messagebox.askyesno(
                    "Nonlinear Analysis Selected",
                    "⚠️ WARNING: Nonlinear analysis is enabled!\n\n" +
                    "This will include:\n" +
                    "• Sample Entropy calculations (very slow)\n" +
                    "• Approximate Entropy calculations (slow)\n" +
                    "• Detrended Fluctuation Analysis (moderate)\n\n" +
                    "Expected time: 5-10 minutes per subject\n" +
                    "Total estimated time: " + str(len(self._prepare_analysis_data_from_config(analysis_config)) * 7) + " minutes\n\n" +
                    "The GUI may appear frozen but the analysis IS running.\n" +
                    "Check the status display for progress updates.\n\n" +
                    "Do you want to continue with nonlinear analysis?"
                )
                if not result:
                    return
            
            # Get analysis data
            analysis_data = self._prepare_analysis_data_from_config(analysis_config)
            if not analysis_data:
                messagebox.showwarning("Warning", "No analysis data prepared")
                return
            
            logger.info(f"Starting simple analysis of {len(analysis_data)} subjects")
            self._update_status("Starting simple analysis...")
            self._update_progress(5, "Processing subjects...")
            self.overall_progress_var.set(f"0/{len(analysis_data)} subjects completed")
            
            # Process each subject synchronously
            all_results = {}
            total_subjects = len(analysis_data)
            completed_subjects = 0
            
            for i, (key, data_segment) in enumerate(analysis_data.items()):
                current_subject = i + 1  # Track current subject number
                try:
                    # Update progress with domain information
                    progress_pct = (i / total_subjects) * 85 + 10  # 10-95% range
                    
                    # Check if nonlinear analysis is enabled and warn user
                    nonlinear_enabled = analysis_config.get('selected_domains', [])
                    has_nonlinear = any(domain == HRVDomain.NONLINEAR for domain in nonlinear_enabled)
                    
                    if has_nonlinear:
                        status_msg = f"Processing {key}... ⚠️ NONLINEAR ANALYSIS - This may take several minutes"
                        self.async_status_var.set("⚠️ NONLINEAR ANALYSIS RUNNING - GUI may appear frozen but it's working")
                    else:
                        status_msg = f"Processing {key}... (Fast analysis)"
                        self.async_status_var.set("✅ Running fast analysis - Should complete quickly")
                    
                    self._update_progress(progress_pct, status_msg)
                    self.root.update_idletasks()  # Allow GUI to update
                    
                    # Additional GUI updates during processing
                    for update_cycle in range(5):  # More frequent updates
                        self.root.update()
                        import time
                        time.sleep(0.05)  # Shorter delay for more responsive updates
                        
                        # Show preparation progress
                        prep_progress = (update_cycle / 5) * 5  # 0-5% for preparation
                        self._update_progress(progress_pct + prep_progress, f"Preparing {key} for analysis...")
                        self.async_status_var.set(f"🔄 Preparing {key} (step {update_cycle + 1}/5)")
                    
                    # Use cached analysis with progress callback
                    def progress_callback(step_name, step_progress):
                        # Update progress for current subject
                        subject_progress = (i / total_subjects) * 85 + 10  # Base progress
                        detailed_progress = subject_progress + (step_progress / total_subjects) * 0.8  # Add step progress
                        
                        status_message = f"Processing {key}: {step_name} ({step_progress:.0f}%)"
                        if has_nonlinear:
                            status_message += " ⚠️ NONLINEAR - May take time"
                        
                        self._update_progress(detailed_progress, status_message)
                        self.async_status_var.set(f"🔄 {key}: {step_name}")
                        
                        # Force GUI update
                        self.root.update_idletasks()
                        self.root.update()
                    
                    result = self._perform_cached_analysis_from_config_with_progress(
                        key, data_segment, selected_domains, analysis_config, progress_callback)
                    
                    if result is not None:
                        all_results[key] = result
                        completed_subjects += 1
                        logger.info(f"Simple: Successfully processed {key}")
                        self.async_status_var.set(f"✅ Completed {key}")
                        self.overall_progress_var.set(f"{completed_subjects}/{total_subjects} subjects completed")
                    else:
                        logger.warning(f"Simple: Failed to process {key}")
                        self.async_status_var.set(f"❌ Failed {key}")
                    
                    # Update overall progress display
                    self.progress_details_var.set(f"Subject {i+1} of {total_subjects}: {key}")
                    
                except Exception as e:
                    logger.error(f"Simple: Error processing {key}: {e}")
                    self.async_status_var.set(f"❌ Error processing {key}: {str(e)[:50]}...")
                    self.progress_details_var.set(f"Error on subject {i+1}: {key}")
                    # Continue with next subject instead of stopping entire analysis
                    self.root.update_idletasks()  # Keep GUI responsive
                    continue
            
            if not all_results:
                messagebox.showerror("Error", "No subjects could be processed successfully")
                return
            
            # Store results and update display
            self.analysis_results = all_results
            self._update_progress(95, "Updating displays...")
            self.root.update_idletasks()
            
            # Update result displays
            self._update_results_display()
            self._update_plots_display()
            
            # Complete
            self._update_progress(100, "Analysis complete!")
            self.async_status_var.set(f"🎉 Analysis Complete! {len(all_results)} subjects processed")
            logger.info(f"Simple analysis completed: {len(all_results)} subjects processed")
            
            # Final GUI update
            self.root.update_idletasks()
            self.root.update()
            
            # Show completion message with plot instructions
            success_message = (
                f"🎉 HRV Analysis Completed Successfully!\n\n"
                f"✅ {len(all_results)} subjects processed\n"
                f"✅ All requested HRV domains computed\n"
                f"✅ Results ready for visualization\n\n"
                f"📊 TO CREATE PLOTS:\n"
                f"1. Click on the 'Visualizations' tab above\n"
                f"2. Select a subject from the dropdown\n"
                f"3. Click any plot generation button\n"
                f"4. Plots will open automatically in your browser\n\n"
                f"Available plot types:\n"
                f"• Poincaré Plot\n"
                f"• Power Spectral Density\n" 
                f"• Time Series\n"
                f"• Complete Dashboard"
            )
            
            messagebox.showinfo("Analysis Complete", success_message)
            
            # Switch to plots tab automatically
            try:
                self.results_notebook.select(1)  # Select plots tab
                logger.info("Automatically switched to Visualizations tab")
            except:
                pass
            
        except Exception as e:
            logger.error(f"Error in simple analysis: {e}")
            messagebox.showerror("Error", f"Analysis failed: {e}")
            
        finally:
            # Reset state
            self.analysis_running = False
            self.simple_process_button.configure(state='normal')
            self.process_button.configure(state='normal')
            self._update_progress(0, "Ready")
    
    def _show_advanced_disabled_message(self):
        """Show message explaining why advanced analysis is disabled."""
        messagebox.showinfo(
            "Advanced Analysis Disabled",
            "Advanced Analysis is disabled to eliminate threading issues.\n\n" +
            "The 'Run HRV Analysis' button provides all analysis capabilities:\n" +
            "• All HRV metrics and domains\n" +
            "• Real-time progress updates\n" +
            "• Reliable processing in main thread\n" +
            "• No background processing complexity\n\n" +
            "Analysis runs only while the GUI is open and stops\n" +
            "immediately if you close the application."
        )
    
    def _capture_analysis_config(self) -> Dict[str, Any]:
        """
        Capture all GUI state needed for analysis.
        This must be called from the main thread before async processing.
        
        Returns:
            Dictionary containing all analysis configuration
        """
        try:
            # Capture domain selections
            selected_domains = []
            for domain, var in self.domain_vars.items():
                if var.get():
                    selected_domains.append(domain)
            
            # Capture other analysis options
            config = {
                'selected_domains': selected_domains,
                'fast_mode': self.fast_mode_var.get() if hasattr(self, 'fast_mode_var') else False,
                'bootstrap_ci': self.bootstrap_ci_var.get() if hasattr(self, 'bootstrap_ci_var') else False,
                'clustering_enabled': self.clustering_var.get() if hasattr(self, 'clustering_var') else False,
                'forecasting_enabled': self.forecasting_var.get() if hasattr(self, 'forecasting_var') else False,
                'current_subject': self.current_subject,
                'max_bootstrap_samples': self.max_bootstrap_samples,
                'analysis_timeout': self.analysis_timeout,
                'cache_version': '2.1'
            }
            
            logger.info(f"Captured analysis config: {len(selected_domains)} domains, "
                       f"fast_mode={config['fast_mode']}, bootstrap_ci={config['bootstrap_ci']}")
            
            return config
            
        except Exception as e:
            logger.error(f"Error capturing analysis config: {e}")
            # Return default config
            return {
                'selected_domains': [HRVDomain.TIME, HRVDomain.FREQUENCY],
                'fast_mode': False,
                'bootstrap_ci': False,
                'clustering_enabled': False,
                'forecasting_enabled': False,
                'current_subject': None,
                'max_bootstrap_samples': 50,
                'analysis_timeout': 300,
                'cache_version': '2.1'
            }
    
    def _perform_analysis_async(self, analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform the actual HRV analysis in async context.
        This method runs in a separate thread and should not update GUI directly.
        
        Args:
            analysis_config: Analysis configuration captured from GUI in main thread
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Get analysis parameters using config instead of GUI variables
            analysis_data = self._prepare_analysis_data_from_config(analysis_config)
            if not analysis_data:
                raise Exception("No analysis data prepared")
            
            # Get selected domains from config instead of GUI
            selected_domains = analysis_config.get('selected_domains', [])
            if not selected_domains:
                raise Exception("No analysis domains selected")
            
            logger.info(f"Starting async analysis of {len(analysis_data)} subjects")
            
            # Process each subject using cached analysis
            all_results = {}
            total_subjects = len(analysis_data)
            
            for i, (key, data_segment) in enumerate(analysis_data.items()):
                try:
                    # Update progress through the callback (thread-safe)
                    progress_pct = (i / total_subjects) * 80 + 10  # 10-90% range
                    # Use the safe callback mechanism instead of direct callback
                    if self.async_processor.progress_callback and self.async_processor._gui_connected:
                        try:
                            self.async_processor.progress_callback(progress_pct, f"Processing {key}...")
                        except Exception as e:
                            if "main thread is not in main loop" in str(e):
                                logger.warning("Progress callback failed - GUI unavailable, continuing analysis")
                                self.async_processor._gui_connected = False
                            else:
                                logger.warning(f"Progress callback error: {e}")
                    
                    # Use cached analysis with config
                    result = self._perform_cached_analysis_from_config(key, data_segment, selected_domains, analysis_config)
                    
                    if result is not None:
                        all_results[key] = result
                        logger.info(f"Async: Successfully processed {key}")
                    else:
                        logger.warning(f"Async: Failed to process {key}")
                    
                except Exception as e:
                    logger.error(f"Async: Error processing {key}: {e}")
                    continue
            
            if not all_results:
                raise Exception("No subjects could be processed successfully")
            
            # Advanced analysis if requested
            if analysis_config.get('clustering_enabled', False):
                try:
                    if self.async_processor.progress_callback and self.async_processor._gui_connected:
                        self.async_processor.progress_callback(90, "Running clustering analysis...")
                    self._perform_clustering_analysis(all_results)
                except Exception as e:
                    logger.warning(f"Clustering analysis failed: {e}")
                
            if analysis_config.get('forecasting_enabled', False):
                try:
                    if self.async_processor.progress_callback and self.async_processor._gui_connected:
                        self.async_processor.progress_callback(95, "Running forecasting analysis...")
                    self._perform_forecasting_analysis(all_results)
                except Exception as e:
                    logger.warning(f"Forecasting analysis failed: {e}")
            
            logger.info(f"Async analysis completed: {len(all_results)} subjects processed")
            return all_results
            
        except Exception as e:
            logger.error(f"Error in async analysis: {e}")
            raise
    
    def _prepare_analysis_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Prepare data for analysis based on subject selection."""
        return self._prepare_analysis_data_from_config({'current_subject': self.current_subject})
    
    def _prepare_analysis_data_from_config(self, analysis_config: Dict[str, Any]) -> Optional[Dict[str, pd.DataFrame]]:
        """Prepare data for analysis based on config (thread-safe version)."""
        try:
            current_subject = analysis_config.get('current_subject')
            
            if current_subject == 'All' or current_subject is None:
                # Group by subject and Sol if available
                if 'subject' in self.loaded_data.columns and 'Sol' in self.loaded_data.columns:
                    groups = self.loaded_data.groupby(['subject', 'Sol'])
                    return {f"{subj}_Sol{sol}": group for (subj, sol), group in groups}
                elif 'subject' in self.loaded_data.columns:
                    groups = self.loaded_data.groupby('subject')
                    return {str(subj): group for subj, group in groups}
                else:
                    return {'All': self.loaded_data}
            else:
                # Single subject analysis
                if 'subject' in self.loaded_data.columns:
                    subject_data = self.loaded_data[self.loaded_data['subject'] == current_subject]
                    if 'Sol' in subject_data.columns:
                        groups = subject_data.groupby('Sol')
                        return {f"{current_subject}_Sol{sol}": group for sol, group in groups}
                    else:
                        return {current_subject: subject_data}
                else:
                    return {'Selected': self.loaded_data}
                    
        except Exception as e:
            logger.error(f"Error preparing analysis data: {e}")
            return None
            
    def _perform_clustering_analysis(self, results: Dict[str, Any]):
        """Perform clustering analysis on HRV results.""" 
        try:
            # Extract HRV metrics for clustering
            hrv_data_list = []
            
            for key, result in results.items():
                if 'hrv_results' in result:
                    hrv_result = result['hrv_results']
                    
                    # Flatten HRV metrics
                    metrics_row = {'subject_session': key}
                    
                    for domain, metrics in hrv_result.items():
                        if isinstance(metrics, dict):
                            for metric_name, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    metrics_row[f"{domain}_{metric_name}"] = value
                                    
                    hrv_data_list.append(metrics_row)
                    
            if len(hrv_data_list) < 3:
                logger.warning("Insufficient data for clustering analysis")
                return
                
            hrv_df = pd.DataFrame(hrv_data_list)
            hrv_df = hrv_df.set_index('subject_session')
            
            # Perform clustering
            cluster_result = self.hrv_clustering.perform_kmeans_clustering(hrv_df)
            cluster_interpretation = self.hrv_clustering.interpret_clusters(cluster_result)
            
            # Store clustering results
            self.analysis_results['clustering'] = {
                'cluster_result': cluster_result,
                'interpretation': cluster_interpretation
            }
            
            logger.info(f"Clustering analysis completed: {cluster_result.n_clusters} clusters")
            
        except Exception as e:
            logger.error(f"Error in clustering analysis: {e}")
            
    def _perform_forecasting_analysis(self, results: Dict[str, Any]):
        """Perform forecasting analysis on HRV trends."""
        try:
            # Create time series data
            time_series_data = {}
            
            for key, result in results.items():
                if 'hrv_results' in result and 'time_domain' in result['hrv_results']:
                    rmssd = result['hrv_results']['time_domain'].get('rmssd', 0)
                    
                    # Extract subject and SOL
                    if '_Sol' in key:
                        subject, sol_str = key.split('_Sol')
                        sol = int(sol_str)
                        
                        if subject not in time_series_data:
                            time_series_data[subject] = {}
                        time_series_data[subject][sol] = rmssd
                        
            # Convert to time series format
            for subject, sol_data in time_series_data.items():
                sols = sorted(sol_data.keys())
                values = [sol_data[sol] for sol in sols]
                ts = pd.Series(values, index=sols, name=f"{subject}_rmssd")
                
                if len(ts) >= 4:  # Minimum for forecasting
                    comparison = self.hrv_forecasting.compare_models(ts)
                    
                    if not hasattr(self, 'forecasting_results'):
                        self.analysis_results['forecasting'] = {}
                    
                    self.analysis_results['forecasting'][subject] = {
                        'time_series': ts,
                        'model_comparison': comparison
                    }
                    
            logger.info("Forecasting analysis completed")
            
        except Exception as e:
            logger.error(f"Error in forecasting analysis: {e}")
            
    def _update_results_display(self):
        """Update the results display with analysis results."""
        try:
            if not self.analysis_results:
                return
                
            # Clear existing results
            self.results_text.delete(1.0, tk.END)
            self.stats_text.delete(1.0, tk.END)
            
            # Results summary
            results_text = "HRV Analysis Results\n"
            results_text += "=" * 50 + "\n\n"
            
            # Individual subject results
            for key, result in self.analysis_results.items():
                if key in ['clustering', 'forecasting']:
                    continue
                    
                results_text += f"Subject/Session: {key}\n"
                results_text += "-" * 30 + "\n"
                
                if 'hrv_results' in result:
                    hrv_results = result['hrv_results']
                    
                    # Time domain results
                    if 'time_domain' in hrv_results:
                        time_domain = hrv_results['time_domain']
                        results_text += "Time Domain Metrics:\n"
                        results_text += f"  SDNN: {time_domain.get('sdnn', 0):.1f} ms\n"
                        results_text += f"  RMSSD: {time_domain.get('rmssd', 0):.1f} ms\n" 
                        results_text += f"  pNN50: {time_domain.get('pnn50', 0):.1f}%\n"
                        results_text += f"  Mean HR: {time_domain.get('mean_hr', 0):.1f} BPM\n\n"
                        
                    # Frequency domain results
                    if 'frequency_domain' in hrv_results:
                        freq_domain = hrv_results['frequency_domain']
                        results_text += "Frequency Domain Metrics:\n"
                        results_text += f"  LF Power: {freq_domain.get('lf_power', 0):.0f} ms²\n"
                        results_text += f"  HF Power: {freq_domain.get('hf_power', 0):.0f} ms²\n"
                        results_text += f"  LF/HF Ratio: {freq_domain.get('lf_hf_ratio', 0):.2f}\n"
                        results_text += f"  LF nu: {freq_domain.get('lf_nu', 0):.1f}%\n"
                        results_text += f"  HF nu: {freq_domain.get('hf_nu', 0):.1f}%\n\n"
                        
                results_text += "\n"
                
            # Clustering results
            if 'clustering' in self.analysis_results:
                results_text += "Clustering Analysis\n"
                results_text += "=" * 30 + "\n"
                
                cluster_result = self.analysis_results['clustering']['cluster_result']
                results_text += f"Number of clusters: {cluster_result.n_clusters}\n"
                results_text += f"Silhouette score: {cluster_result.silhouette_score:.3f}\n\n"
                
                interpretation = self.analysis_results['clustering']['interpretation']
                if 'overall_analysis' in interpretation:
                    overall = interpretation['overall_analysis']
                    results_text += f"Cluster quality: {overall.get('cluster_quality', 'Unknown')}\n"
                    results_text += f"Phenotype distribution: {overall.get('phenotype_distribution', {})}\n\n"
                    
            # Forecasting results  
            if 'forecasting' in self.analysis_results:
                results_text += "Forecasting Analysis\n"
                results_text += "=" * 30 + "\n"
                
                for subject, forecast_data in self.analysis_results['forecasting'].items():
                    results_text += f"{subject}:\n"
                    if 'model_comparison' in forecast_data:
                        comparison = forecast_data['model_comparison']
                        results_text += f"  Best model: {comparison.best_model_name}\n"
                        if not comparison.performance_metrics.empty:
                            best_rmse = comparison.performance_metrics.iloc[0]['rmse']
                            results_text += f"  RMSE: {best_rmse:.2f}\n"
                    results_text += "\n"
                    
            self.results_text.insert(tk.END, results_text)
            
            # Statistical summary
            self._update_statistics_display()
            
        except Exception as e:
            logger.error(f"Error updating results display: {e}")
            
    def _update_statistics_display(self):
        """Update the statistics display."""
        try:
            stats_text = "Statistical Summary\n"
            stats_text += "=" * 40 + "\n\n"
            
            # Collect all metrics for statistical analysis
            all_metrics = {}
            
            for key, result in self.analysis_results.items():
                if key in ['clustering', 'forecasting']:
                    continue
                    
                if 'hrv_results' in result:
                    hrv_results = result['hrv_results']
                    
                    for domain, metrics in hrv_results.items():
                        if isinstance(metrics, dict):
                            for metric_name, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    full_name = f"{domain}_{metric_name}"
                                    if full_name not in all_metrics:
                                        all_metrics[full_name] = []
                                    all_metrics[full_name].append(value)
                                    
            # Compute descriptive statistics
            for metric_name, values in all_metrics.items():
                if len(values) > 1:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)
                    
                    stats_text += f"{metric_name}:\n"
                    stats_text += f"  Mean ± SD: {mean_val:.2f} ± {std_val:.2f}\n"
                    stats_text += f"  Range: {min_val:.2f} - {max_val:.2f}\n"
                    stats_text += f"  N: {len(values)}\n\n"
                    
            self.stats_text.insert(tk.END, stats_text)
            
        except Exception as e:
            logger.error(f"Error updating statistics display: {e}")
    
    def _update_plots_display(self):
        """Update the plots display with visualization controls."""
        try:
            logger.info("Updating plots display...")
            if not self.analysis_results:
                logger.warning("No analysis results available for plots display")
                self.plot_subject_combo['values'] = ["No subjects analyzed"]
                self.plot_subject_combo.set("No subjects analyzed")
                return

            # Populate subject combobox with only valid subjects for plotting
            subjects = sorted([
                key for key, result in self.analysis_results.items()
                if isinstance(result, dict) and 'rr_intervals' in result
            ])

            if not subjects:
                self.plot_subject_combo['values'] = ["No plot-able subjects found"]
                self.plot_subject_combo.set("No plot-able subjects found")
                return

            self.plot_subject_combo['values'] = subjects
            if subjects:
                self.plot_subject_combo.set(subjects[0])
            
            logger.info(f"Populated plot controls for {len(subjects)} subjects")
            
        except Exception as e:
            logger.error(f"Error updating plots display: {e}")
            
    def _create_disabled_plot_interface(self):
        """Create a disabled state for the plot interface."""
        try:
            main_frame = ttk.Frame(self.plots_scrollable_frame)
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
            
            ttk.Label(main_frame, text="⚠️ NO ANALYSIS RESULTS", 
                     style='Title.TLabel').grid(row=0, column=0, pady=20)
            
            ttk.Label(main_frame, 
                     text="Run the HRV analysis first to enable plot generation.\n\n" +
                          "1. Go to 'Analysis Configuration'\n" +
                          "2. Select desired HRV domains\n" + 
                          "3. Click 'Run HRV Analysis'\n" +
                          "4. Return here when analysis completes",
                     justify=tk.CENTER).grid(row=1, column=0, pady=20)
                     
        except Exception as e:
            logger.error(f"Error creating disabled interface: {e}")
    
    def _create_emergency_plot_interface(self, error_msg):
        """Create emergency plot interface when main interface fails."""
        try:
            main_frame = ttk.Frame(self.plots_scrollable_frame)
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
            
            ttk.Label(main_frame, text="❌ PLOT INTERFACE ERROR", 
                     style='Title.TLabel', foreground='red').grid(row=0, column=0, pady=10)
            
            error_text = f"Plot interface creation failed:\n{error_msg}\n\n"
            error_text += "Emergency plot generation (basic functionality):"
            
            ttk.Label(main_frame, text=error_text, 
                     justify=tk.LEFT, foreground='red').grid(row=1, column=0, pady=10)
            
            # Basic emergency buttons
            if hasattr(self, 'analysis_results') and self.analysis_results:
                subjects = list(self.analysis_results.keys())
                
                self.emergency_subject_var = tk.StringVar(master=self.root, value=subjects[0] if subjects else "")
                emergency_combo = ttk.Combobox(main_frame, textvariable=self.emergency_subject_var,
                                             values=subjects, state='readonly')
                emergency_combo.grid(row=2, column=0, pady=10)
                
                ttk.Button(main_frame, text="Emergency Dashboard Plot", 
                          command=self._emergency_generate_plot).grid(row=3, column=0, pady=10)
                          
        except Exception as e:
            logger.error(f"Emergency interface also failed: {e}")
    
    def _emergency_generate_plot(self):
        """Emergency plot generation method."""
        try:
            subject = self.emergency_subject_var.get()
            if subject and subject in self.analysis_results:
                result = self.analysis_results[subject]
                if 'rr_intervals' in result:
                    # Generate basic dashboard
                    fig = self.interactive_plotter.create_hrv_dashboard(
                        result['rr_intervals'],
                        result.get('hrv_results', {}),
                        subject_id=subject,
                        session_id="Emergency"
                    )
                    
                    plot_path = Path(f"emergency_plot_{subject.replace('/', '_')}.html")
                    self.interactive_plotter.export_html(fig, str(plot_path))
                    self._open_plot_file(plot_path)
                    
                    messagebox.showinfo("Success", f"Emergency plot created: {plot_path.name}")
                else:
                    messagebox.showerror("Error", "No RR interval data available")
            else:
                messagebox.showerror("Error", "No subject selected or data unavailable")
                
        except Exception as e:
            logger.error(f"Emergency plot generation failed: {e}")
            messagebox.showerror("Error", f"Emergency plot generation failed: {e}")
    
    def _clear_plots_display(self):
        """Clear the plot display area."""
        pass
    
    def _clear_plot_buttons(self):
        """Clear any dynamically added plot buttons."""
        pass
    
    def _generate_poincare_plot(self):
        """Generate and display Poincaré plot."""
        try:
            selected_subject = self.plot_subject_var.get()
            if not self.analysis_results or not selected_subject or "No subject" in selected_subject:
                messagebox.showwarning("Plotting Error", "Please run an analysis and select a subject first.")
                return

            result = self.analysis_results.get(selected_subject)
            if not isinstance(result, dict) or 'rr_intervals' not in result:
                messagebox.showwarning("Plotting Error", f"No plottable data found for '{selected_subject}'. It may be a summary entry.")
                return
            
            rr_intervals = result['rr_intervals']
            
            self.plot_status_label.configure(text=f"Generating Poincaré plot for {selected_subject}...")
            self.root.update_idletasks()
            
            fig = self.interactive_plotter.create_poincare_plot(
                rr_intervals, 
                title=f"Poincaré Plot - {selected_subject}"
            )
            
            safe_subject = selected_subject.replace('/', '_').replace('\\', '_')
            plots_dir = Path("plots_output")
            plot_path = plots_dir / f"poincare_plot_{safe_subject}.html"
            export_success = self.interactive_plotter.export_html(fig, str(plot_path))
            
            if export_success:
                success_text = f"✅ Poincaré plot generated for {selected_subject} and saved to {plot_path.absolute()}"
                self.plot_status_label.configure(text=success_text)
                self._open_plot_file(plot_path)
            else:
                error_text = f"❌ Failed to generate Poincaré plot for {selected_subject}"
                self.plot_status_label.configure(text=error_text)
            
        except Exception as e:
            logger.error(f"Error generating Poincaré plot: {e}")
            self.plot_status_label.configure(text=f"Error generating Poincaré plot: {e}")
    
    def _generate_psd_plot(self):
        """Generate and display Power Spectral Density plot."""
        try:
            selected_subject = self.plot_subject_var.get()
            if not self.analysis_results or not selected_subject or "No subject" in selected_subject:
                messagebox.showwarning("Plotting Error", "Please run an analysis and select a subject first.")
                return

            result = self.analysis_results.get(selected_subject)
            if not isinstance(result, dict) or 'rr_intervals' not in result:
                messagebox.showwarning("Plotting Error", f"No plottable data found for '{selected_subject}'. It may be a summary entry.")
                return
            
            rr_intervals = result['rr_intervals']
            
            self.plot_status_label.configure(text=f"Generating PSD plot for {selected_subject}...")
            self.root.update_idletasks()
            
            fig = self.interactive_plotter.create_psd_plot(
                rr_intervals,
                title=f"Power Spectral Density - {selected_subject}"
            )
            
            safe_subject = selected_subject.replace('/', '_').replace('\\', '_')
            plots_dir = Path("plots_output")
            plot_path = plots_dir / f"psd_plot_{safe_subject}.html"
            export_success = self.interactive_plotter.export_html(fig, str(plot_path))
            
            if export_success:
                success_text = f"✅ PSD plot generated for {selected_subject} and saved to {plot_path.absolute()}"
                self.plot_status_label.configure(text=success_text)
                self._open_plot_file(plot_path)
            else:
                error_text = f"❌ Failed to generate PSD plot for {selected_subject}"
                self.plot_status_label.configure(text=error_text)
            
        except Exception as e:
            logger.error(f"Error generating PSD plot: {e}")
            self.plot_status_label.configure(text=f"Error generating PSD plot: {e}")
    
    def _generate_timeseries_plot(self):
        """Generate and display RR interval time series plot."""
        try:
            selected_subject = self.plot_subject_var.get()
            if not self.analysis_results or not selected_subject or "No subject" in selected_subject:
                messagebox.showwarning("Plotting Error", "Please run an analysis and select a subject first.")
                return

            result = self.analysis_results.get(selected_subject)
            if not isinstance(result, dict) or 'rr_intervals' not in result:
                messagebox.showwarning("Plotting Error", f"No plottable data found for '{selected_subject}'. It may be a summary entry.")
                return
            
            rr_intervals = result['rr_intervals']
            
            self.plot_status_label.configure(text=f"Generating time series plot for {selected_subject}...")
            self.root.update_idletasks()
            
            fig = self.interactive_plotter.create_time_series_plot(
                rr_intervals,
                title=f"RR Interval Time Series - {selected_subject}"
            )
            
            safe_subject = selected_subject.replace('/', '_').replace('\\', '_')
            plots_dir = Path("plots_output")
            plot_path = plots_dir / f"timeseries_plot_{safe_subject}.html"
            export_success = self.interactive_plotter.export_html(fig, str(plot_path))
            
            if export_success:
                success_text = f"✅ Time series plot generated for {selected_subject} and saved to {plot_path.absolute()}"
                self.plot_status_label.configure(text=success_text)
                self._open_plot_file(plot_path)
            else:
                error_text = f"❌ Failed to generate time series plot for {selected_subject}"
                self.plot_status_label.configure(text=error_text)
            
        except Exception as e:
            logger.error(f"Error generating time series plot: {e}")
            self.plot_status_label.configure(text=f"Error generating time series plot: {e}")
    
    def _generate_all_plots(self):
        """Generate all available plots for the selected subject."""
        try:
            selected_subject = self.plot_subject_var.get()
            if not self.analysis_results or not selected_subject or "No subject" in selected_subject:
                messagebox.showwarning("Plotting Error", "Please run an analysis and select a subject first.")
                return

            result = self.analysis_results.get(selected_subject)
            if not isinstance(result, dict) or 'rr_intervals' not in result:
                messagebox.showwarning("Plotting Error", f"No plottable data found for '{selected_subject}'. It may be a summary entry.")
                return
            
            self.plot_status_label.configure(text=f"Generating comprehensive dashboard for {selected_subject}...")
            self.root.update_idletasks()
            
            rr_intervals = result['rr_intervals']
            hrv_results = result.get('hrv_results', {})
            
            dashboard_fig = self.interactive_plotter.create_hrv_dashboard(
                rr_intervals,
                hrv_results,
                subject_id=selected_subject,
                session_id="Analysis"
            )
            
            safe_subject = selected_subject.replace('/', '_').replace('\\', '_')
            plots_dir = Path("plots_output")
            dashboard_path = plots_dir / f"hrv_dashboard_{safe_subject}.html"
            self.interactive_plotter.export_html(dashboard_fig, str(dashboard_path))
            
            success_text = f"✅ HRV Analysis Dashboard generated for {selected_subject} and saved to {dashboard_path.absolute()}"
            self.plot_status_label.configure(text=success_text)
            
            self._open_plot_file(dashboard_path)
            
        except Exception as e:
            logger.error(f"Error generating HRV dashboard: {e}")
            self.plot_status_label.configure(text=f"Error generating HRV dashboard: {e}")
    
    def _generate_combined_time_series(self):
        """Generate combined time series analysis for all subjects and HRV metrics."""
        try:
            if not self.analysis_results:
                messagebox.showwarning("Warning", "No analysis results available")
                return
            
            # Update status
            self.plot_status_label.configure(text="Generating combined time series analysis for all subjects...")
            self.root.update_idletasks()
            
            # Generate comprehensive time series analysis
            combined_fig = self.interactive_plotter.create_combined_time_series_analysis(
                analysis_results=self.analysis_results
            )
            
            # Save the combined analysis with unique filename
            plot_path = Path("plots_output") / "hrv_combined_time_series_analysis.html"
            self.interactive_plotter.export_html(combined_fig, str(plot_path))
            
            # Update status with success message
            success_text = "✅ Combined Time Series Analysis Generated!\n"
            success_text += f"Comprehensive analysis saved as: {plot_path.absolute()}"
            self.plot_status_label.configure(text=success_text)

            # Automatically open the plot
            self._open_plot_file(plot_path)
            
        except Exception as e:
            logger.error(f"Error in combined time series generation: {e}")
            messagebox.showerror("Error", f"Error generating combined time series: {e}")
    
    def _generate_custom_time_series(self):
        """Generate custom time series plot based on user selection."""
        try:
            if not self.analysis_results:
                messagebox.showwarning("Warning", "No analysis results available")
                return

            custom_dialog = tk.Toplevel(self.root)
            custom_dialog.title("Select Metrics for Time Series")
            custom_dialog.geometry("400x500")
            custom_dialog.transient(self.root)
            custom_dialog.grab_set()

            ttk.Label(custom_dialog, text="Select metrics to plot:", font=('Helvetica', 12, 'bold')).pack(pady=10)

            # --- Button Frame ---
            button_frame = ttk.Frame(custom_dialog)
            button_frame.pack(side=tk.BOTTOM, pady=10, fill=tk.X, padx=10)
            button_frame.columnconfigure((0, 1), weight=1)

            # --- Scrollable List Frame ---
            list_container = ttk.Frame(custom_dialog)
            list_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10)
            
            canvas = tk.Canvas(list_container)
            scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)

            scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            available_metrics = self.interactive_plotter.get_available_metrics(self.analysis_results)
            metric_vars = {}
            for metric in sorted(available_metrics):
                var = tk.BooleanVar(master=self.root)
                chk = ttk.Checkbutton(scrollable_frame, text=metric, variable=var)
                chk.pack(anchor=tk.W, padx=10, pady=2)
                metric_vars[metric] = var

            def generate_custom_plot():
                selected_metrics = [metric for metric, var in metric_vars.items() if var.get()]
                
                if not selected_metrics:
                    messagebox.showwarning("Warning", "Please select at least one metric")
                    return
                
                custom_dialog.destroy()
                
                self.plot_status_label.configure(text=f"Generating custom time series for {len(selected_metrics)} metrics...")
                self.root.update_idletasks()
                
                custom_fig = self.interactive_plotter.create_combined_time_series_analysis(
                    analysis_results=self.analysis_results,
                    metrics_to_plot=selected_metrics
                )
                
                plot_path = Path("plots_output") / "hrv_custom_time_series_analysis.html"
                self.interactive_plotter.export_html(custom_fig, str(plot_path))
                
                success_text = f"✅ Custom Time Series Analysis Generated!\n"
                success_text += f"Analysis saved as: {plot_path.absolute()}"
                self.plot_status_label.configure(text=success_text)

                self._open_plot_file(plot_path)
            
            def cancel_dialog():
                custom_dialog.destroy()
            
            ttk.Button(button_frame, text="Generate Plot", command=generate_custom_plot).grid(row=0, column=0, padx=5, sticky="ew")
            ttk.Button(button_frame, text="Cancel", command=cancel_dialog).grid(row=0, column=1, padx=5, sticky="ew")
            
        except Exception as e:
            logger.error(f"Error in custom time series generation: {e}")
            messagebox.showerror("Error", f"Error generating custom time series: {e}")
    
    def _generate_gam_crew_analysis(self):
        """Generate GAM (Generalized Additive Model) analysis for crew-wide trends."""
        try:
            if not self.analysis_results:
                messagebox.showwarning("Warning", "No analysis results available for GAM analysis")
                return

            self.plot_status_label.configure(text="Generating comprehensive GAM crew analysis with trend lines and confidence intervals...")
            self.root.update_idletasks()

            # Generate GAM analysis using comprehensive HRV metrics (will use defaults from interactive_plotter)
            gam_fig = self.interactive_plotter.create_gam_crew_analysis(
                analysis_results=self.analysis_results,
                metrics_to_plot=None,  # Use comprehensive default metrics
                show_individual_points=True,
                show_crew_median=True,
                confidence_level=0.95
            )

            plot_path = Path("plots_output") / "hrv_gam_crew_analysis.html"
            self.interactive_plotter.export_html(gam_fig, str(plot_path))

            success_text = f"✅ Comprehensive GAM Crew Analysis Generated!\n"
            success_text += f"Professional analysis with comprehensive HRV metrics across all domains\n"
            success_text += f"Includes trend lines, confidence intervals, and crew median calculations\n"
            success_text += f"Analysis saved as: {plot_path.absolute()}"
            self.plot_status_label.configure(text=success_text)

            self._open_plot_file(plot_path)

        except Exception as e:
            logger.error(f"Error in GAM crew analysis generation: {e}")
            messagebox.showerror("Error", f"Error generating GAM crew analysis: {e}")
    
    def _generate_gam_custom_analysis(self):
        """Generate custom GAM analysis with user-selected metrics."""
        try:
            if not self.analysis_results:
                messagebox.showwarning("Warning", "No analysis results available")
                return

            custom_dialog = tk.Toplevel(self.root)
            custom_dialog.title("GAM Analysis - Select Metrics")
            custom_dialog.geometry("600x700")
            custom_dialog.transient(self.root)
            custom_dialog.grab_set()

            # Header
            header_label = ttk.Label(custom_dialog, 
                                   text="Select HRV Metrics for GAM Crew Analysis:",
                                   font=('Helvetica', 14, 'bold'))
            header_label.pack(pady=10)

            # Info label
            info_label = ttk.Label(custom_dialog,
                                 text="GAM analysis provides trend lines with confidence intervals for temporal analysis",
                                 font=('Helvetica', 10),
                                 foreground='#555555')
            info_label.pack(pady=(0, 10))

            # Get metrics organized by domain
            available_metrics_by_domain = self.interactive_plotter.get_available_metrics_by_domain(self.analysis_results)
            recommended_metrics = self.interactive_plotter._get_recommended_gam_metrics()
            metric_descriptions = self.interactive_plotter._get_metric_descriptions()

            # --- Button Frame ---
            button_frame = ttk.Frame(custom_dialog)
            button_frame.pack(side=tk.BOTTOM, pady=10, fill=tk.X, padx=10)
            button_frame.columnconfigure((0, 1, 2), weight=1)

            # --- Tabbed Interface for Metric Selection ---
            notebook = ttk.Notebook(custom_dialog)
            notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

            metric_vars = {}
            
            # Tab 1: Recommended Metrics
            recommended_frame = ttk.Frame(notebook)
            notebook.add(recommended_frame, text="Recommended")
            
            # Create canvas and scrollbar for recommended
            rec_canvas = tk.Canvas(recommended_frame)
            rec_scrollbar = ttk.Scrollbar(recommended_frame, orient="vertical", command=rec_canvas.yview)
            rec_scrollable_frame = ttk.Frame(rec_canvas)

            rec_scrollable_frame.bind("<Configure>", lambda e: rec_canvas.configure(scrollregion=rec_canvas.bbox("all")))
            rec_canvas.create_window((0, 0), window=rec_scrollable_frame, anchor="nw")
            rec_canvas.configure(yscrollcommand=rec_scrollbar.set)

            rec_canvas.pack(side="left", fill="both", expand=True)
            rec_scrollbar.pack(side="right", fill="y")

            # Add recommended metrics by category
            for domain_name, metrics in recommended_metrics.items():
                # Domain header
                domain_label = ttk.Label(rec_scrollable_frame, 
                                       text=f"📊 {domain_name}",
                                       font=('Helvetica', 12, 'bold'),
                                       foreground='#2E86C1')
                domain_label.pack(anchor=tk.W, padx=10, pady=(15, 5))
                
                # Metrics in this domain
                for metric in metrics:
                    if metric in available_metrics_by_domain.get(domain_name.split(' ')[-1] if ' ' in domain_name else domain_name, []):
                        var = tk.BooleanVar(master=self.root)
                        var.set(True)  # Pre-select recommended metrics
                        
                        # Get metric description
                        description = metric_descriptions.get(metric, metric)
                        
                        # Create frame for metric
                        metric_frame = ttk.Frame(rec_scrollable_frame)
                        metric_frame.pack(fill=tk.X, padx=20, pady=2)
                        
                        chk = ttk.Checkbutton(metric_frame, text=description[:60] + "..." if len(description) > 60 else description, variable=var)
                        chk.pack(anchor=tk.W)
                        metric_vars[metric] = var

            # Tab 2: All Available Metrics
            all_metrics_frame = ttk.Frame(notebook)
            notebook.add(all_metrics_frame, text="All Available")
            
            # Create canvas and scrollbar for all metrics
            all_canvas = tk.Canvas(all_metrics_frame)
            all_scrollbar = ttk.Scrollbar(all_metrics_frame, orient="vertical", command=all_canvas.yview)
            all_scrollable_frame = ttk.Frame(all_canvas)

            all_scrollable_frame.bind("<Configure>", lambda e: all_canvas.configure(scrollregion=all_canvas.bbox("all")))
            all_canvas.create_window((0, 0), window=all_scrollable_frame, anchor="nw")
            all_canvas.configure(yscrollcommand=all_scrollbar.set)

            all_canvas.pack(side="left", fill="both", expand=True)
            all_scrollbar.pack(side="right", fill="y")

            # Add all available metrics organized by domain
            for domain_name, metrics in available_metrics_by_domain.items():
                if metrics:  # Only show domains with available metrics
                    # Domain header
                    domain_label = ttk.Label(all_scrollable_frame, 
                                           text=f"🔬 {domain_name}",
                                           font=('Helvetica', 12, 'bold'),
                                           foreground='#138D75')
                    domain_label.pack(anchor=tk.W, padx=10, pady=(15, 5))
                    
                    # Metrics in this domain
                    for metric in metrics:
                        # Skip if already added in recommended
                        if metric not in metric_vars:
                            var = tk.BooleanVar(master=self.root)
                            
                            # Get metric description
                            description = metric_descriptions.get(metric, metric.replace('_', ' ').title())
                            
                            # Create frame for metric
                            metric_frame = ttk.Frame(all_scrollable_frame)
                            metric_frame.pack(fill=tk.X, padx=20, pady=2)
                            
                            chk = ttk.Checkbutton(metric_frame, text=description[:60] + "..." if len(description) > 60 else description, variable=var)
                            chk.pack(anchor=tk.W)
                            metric_vars[metric] = var

            def select_all_recommended():
                """Select all recommended metrics."""
                for domain_name, metrics in recommended_metrics.items():
                    for metric in metrics:
                        if metric in metric_vars:
                            metric_vars[metric].set(True)
            
            def clear_all_selections():
                """Clear all metric selections."""
                for var in metric_vars.values():
                    var.set(False)

            def generate_gam_plot():
                selected_metrics = [metric for metric, var in metric_vars.items() if var.get()]
                
                if not selected_metrics:
                    messagebox.showwarning("Warning", "Please select at least one metric")
                    return
                
                custom_dialog.destroy()
                
                self.plot_status_label.configure(text=f"Generating comprehensive GAM analysis for {len(selected_metrics)} metrics...")
                self.root.update_idletasks()
                
                gam_fig = self.interactive_plotter.create_gam_crew_analysis(
                    analysis_results=self.analysis_results,
                    metrics_to_plot=selected_metrics,
                    show_individual_points=True,
                    show_crew_median=True,
                    confidence_level=0.95
                )
                
                plot_path = Path("plots_output") / "hrv_gam_comprehensive_analysis.html"
                self.interactive_plotter.export_html(gam_fig, str(plot_path))
                
                success_text = f"✅ Comprehensive GAM Analysis Generated!\n"
                success_text += f"Advanced statistical analysis with {len(selected_metrics)} HRV metrics\n"
                success_text += f"Covers all HRV domains with trend analysis and confidence intervals\n" 
                success_text += f"Analysis saved as: {plot_path.absolute()}"
                self.plot_status_label.configure(text=success_text)

                self._open_plot_file(plot_path)
            
            def cancel_dialog():
                custom_dialog.destroy()
            
            # Button row
            ttk.Button(button_frame, text="Select Recommended", command=select_all_recommended).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
            ttk.Button(button_frame, text="Clear All", command=clear_all_selections).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            
            ttk.Button(button_frame, text="Generate GAM Analysis", command=generate_gam_plot, style='Accent.TButton').grid(row=1, column=0, padx=5, pady=5, sticky="ew")
            ttk.Button(button_frame, text="Cancel", command=cancel_dialog).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
            
        except Exception as e:
            logger.error(f"Error in GAM custom analysis generation: {e}")
            messagebox.showerror("Error", f"Error generating GAM custom analysis: {e}")

    def _open_plot_file(self, file_path):
        """Open plot file in default browser."""
        try:
            import webbrowser
            webbrowser.open(f"file://{file_path.absolute()}")
        except Exception as e:
            logger.error(f"Error opening plot file: {e}")
            messagebox.showerror("Error", f"Could not open plot file: {e}")
            
    def _clear_results(self):
        """Clear all analysis results.""" 
        self.analysis_results = {}
        self.results_text.delete(1.0, tk.END)
        self.stats_text.delete(1.0, tk.END)
        self._clear_plots_display()
        self._update_status("Results cleared")
        
    def _export_results(self):
        """Export analysis results to file."""
        try:
            if not self.analysis_results:
                messagebox.showwarning("Warning", "No results to export")
                return
                
            export_path = filedialog.asksaveasfilename(
                title="Export Results",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if export_path:
                if export_path.endswith('.csv'):
                    self._export_results_csv(export_path)
                else:
                    self._export_results_json(export_path)
                    
                messagebox.showinfo("Success", f"Results exported to {export_path}")
                
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            messagebox.showerror("Error", f"Export failed: {e}")
            
    def _export_results_json(self, file_path: str):
        """Export results to JSON format."""
        # Convert numpy arrays to lists for JSON serialization
        exportable_results = {}
        
        for key, result in self.analysis_results.items():
            exportable_results[key] = self._make_json_serializable(result)
            
        with open(file_path, 'w') as f:
            json.dump(exportable_results, f, indent=2, default=str)
            
    def _export_results_csv(self, file_path: str):
        """Export results to CSV format."""
        # Flatten results for CSV export
        flattened_data = []
        
        for key, result in self.analysis_results.items():
            if key in ['clustering', 'forecasting']:
                continue
                
            if 'hrv_results' in result:
                row = {'subject_session': key}
                hrv_results = result['hrv_results']
                
                for domain, metrics in hrv_results.items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                row[f"{domain}_{metric_name}"] = value
                                
                flattened_data.append(row)
                
        if flattened_data:
            df = pd.DataFrame(flattened_data)
            df.to_csv(file_path, index=False)
            
    def _export_plots(self):
        """Export generated plots."""
        try:
            if not self.analysis_results:
                messagebox.showwarning("Warning", "No results to export plots from")
                return
                
            export_dir = filedialog.askdirectory(title="Select Directory for Plot Export")
            
            if export_dir:
                # Export individual plots for each subject
                plot_count = 0
                
                for key, result in self.analysis_results.items():
                    if key in ['clustering', 'forecasting']:
                        continue
                        
                    if 'rr_intervals' in result:
                        rr_intervals = result['rr_intervals']
                        
                        # Create Poincaré plot
                        poincare_fig = self.interactive_plotter.create_poincare_plot(
                            rr_intervals, title=f"Poincaré Plot - {key}"
                        )
                        
                        plot_path = Path(export_dir) / f"poincare_{key}.html"
                        self.interactive_plotter.export_html(poincare_fig, str(plot_path))
                        plot_count += 1
                        
                        # Create PSD plot
                        psd_fig = self.interactive_plotter.create_psd_plot(
                            rr_intervals, title=f"Power Spectral Density - {key}"
                        )
                        
                        plot_path = Path(export_dir) / f"psd_{key}.html"
                        self.interactive_plotter.export_html(psd_fig, str(plot_path))
                        plot_count += 1
                        
                messagebox.showinfo("Success", f"Exported {plot_count} plots to {export_dir}")
                
        except Exception as e:
            logger.error(f"Error exporting plots: {e}")
            messagebox.showerror("Error", f"Plot export failed: {e}")
            
    def _generate_report(self):
        """Generate comprehensive analysis report."""
        try:
            if not self.analysis_results:
                messagebox.showwarning("Warning", "No results to generate report from") 
                return
                
            report_path = filedialog.asksaveasfilename(
                title="Save Analysis Report",
                defaultextension=".html",
                filetypes=[("HTML files", "*.html"), ("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if report_path:
                if report_path.endswith('.html'):
                    self._generate_html_report(report_path)
                else:
                    self._generate_text_report(report_path)
                    
                messagebox.showinfo("Success", f"Report generated: {report_path}")
                
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            messagebox.showerror("Error", f"Report generation failed: {e}")
            
    def _generate_html_report(self, file_path: str):
        """Generate HTML report with embedded plots."""
        html_content = f"""
        <html>
        <head>
            <title>HRV Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Enhanced HRV Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Analysis Summary</h2>
            <p>Total subjects analyzed: {len([k for k in self.analysis_results.keys() if k not in ['clustering', 'forecasting']])}</p>
            
            <h2>Individual Results</h2>
        """
        
        for key, result in self.analysis_results.items():
            if key in ['clustering', 'forecasting']:
                continue
                
            html_content += f"<h3>{key}</h3>\n"
            
            if 'hrv_results' in result:
                hrv_results = result['hrv_results']
                html_content += "<table>\n<tr><th>Metric</th><th>Value</th></tr>\n"
                
                for domain, metrics in hrv_results.items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                html_content += f"<tr><td>{domain}_{metric_name}</td><td>{value:.2f}</td></tr>\n"
                                
                html_content += "</table>\n"
                
        html_content += """
        </body>
        </html>
        """
        
        with open(file_path, 'w') as f:
            f.write(html_content)
            
    def _generate_text_report(self, file_path: str):
        """Generate text-based report."""
        report_text = self.results_text.get(1.0, tk.END)
        
        with open(file_path, 'w') as f:
            f.write(report_text)
            
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON export."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
            
    def _show_about(self):
        """Show about dialog."""
        about_text = """Enhanced HRV Analysis System
Version 2.0.0

A comprehensive tool for Heart Rate Variability analysis 
featuring advanced statistical methods, machine learning, 
and interactive visualizations.

Author: Dr. Diego Malpica - Aerospace Medicine Specialist
Organization: DIMAE / FAC / Colombia

Created for the Valquiria Space Analog Simulation project.

Special Thanks:
• Women AeroSTEAM
• Centro de Telemedicina de Colombia  
• Valquiria Space Analog Crew

© 2025 Dr. Diego Malpica"""
        
        messagebox.showinfo("About Enhanced HRV Analysis", about_text)
        
    def _apply_memory_protection(self, hr_data: pd.Series, subject_key: str) -> pd.Series:
        """Apply memory protection and data validation with improved RR interval alignment."""
        try:
            original_size = len(hr_data)
            
            # Remove invalid data points
            hr_data = hr_data.dropna()
            hr_data = hr_data[hr_data > 0]  # Remove zero or negative values
            
            # Apply fast mode if enabled or data is too large
            if self.fast_mode or len(hr_data) > 10000:
                # Use smart sampling instead of simple truncation to preserve temporal patterns
                if len(hr_data) > 10000:
                    logger.info(f"Memory protection: Limited {subject_key} to 10000 samples")
                    # Sample at regular intervals to maintain temporal structure
                    step_size = max(1, len(hr_data) // 10000)
                    hr_data = hr_data.iloc[::step_size].head(10000)
                else:
                    logger.info(f"Fast mode: Limited {subject_key} to {len(hr_data)} samples")
            
            # Additional data quality checks
            if len(hr_data) < 50:
                logger.warning(f"Very short data segment for {subject_key}: {len(hr_data)} samples")
                return hr_data
            
            # Fix common data alignment issues that cause Time-RR mismatch
            # Ensure data is properly indexed with integer index
            hr_data = hr_data.reset_index(drop=True)
            
            # Remove outliers that might cause processing issues
            mean_hr = hr_data.mean()
            std_hr = hr_data.std()
            
            # Define reasonable heart rate bounds (30-200 BPM)
            lower_bound = max(30, mean_hr - 4 * std_hr)
            upper_bound = min(200, mean_hr + 4 * std_hr)
            
            before_outlier_removal = len(hr_data)
            hr_data = hr_data[(hr_data >= lower_bound) & (hr_data <= upper_bound)]
            after_outlier_removal = len(hr_data)
            
            if before_outlier_removal != after_outlier_removal:
                removed_count = before_outlier_removal - after_outlier_removal
                logger.info(f"Removed {removed_count} outliers from {subject_key} "
                           f"({removed_count/before_outlier_removal*100:.1f}%)")
            
            # Ensure minimum data requirement after cleaning
            if len(hr_data) < 50:
                logger.warning(f"Insufficient data after cleaning for {subject_key}: {len(hr_data)} samples")
                return pd.Series(dtype=float)  # Return empty series
            
            # Reset index again to ensure clean integer indexing for RR calculation
            hr_data = hr_data.reset_index(drop=True)
            
            # Log final data statistics
            logger.info(f"Data processing for {subject_key}: "
                       f"{original_size} → {len(hr_data)} samples "
                       f"(mean: {hr_data.mean():.1f} BPM, "
                       f"std: {hr_data.std():.1f} BPM)")
            
            return hr_data
            
        except Exception as e:
            logger.error(f"Error in memory protection for {subject_key}: {e}")
            # Return original data if processing fails
            return hr_data if isinstance(hr_data, pd.Series) else pd.Series(hr_data)
    
    def _perform_cached_analysis(self, subject_key: str, data_segment: pd.DataFrame, selected_domains: List[HRVDomain]) -> Optional[Dict[str, Any]]:
        """
        Perform HRV analysis with intelligent caching for faster repeated processing.
        
        Args:
            subject_key: Identifier for the subject/session
            data_segment: Data for this subject/session
            selected_domains: HRV domains to analyze
            
        Returns:
            Analysis results dictionary or None if analysis failed
        """
        # Check if we're in the main thread - if not, we can't capture GUI state safely
        import threading
        if threading.current_thread() != threading.main_thread():
            # We're in a background thread, use a default config to avoid GUI access
            logger.warning("_perform_cached_analysis called from background thread - using default config")
            default_config = {
                'selected_domains': selected_domains,
                'fast_mode': False,
                'bootstrap_ci': False,
                'clustering_enabled': False,
                'forecasting_enabled': False,
                'current_subject': None,
                'max_bootstrap_samples': 50,
                'analysis_timeout': 300,
                'cache_version': '2.1'
            }
            return self._perform_cached_analysis_from_config(subject_key, data_segment, selected_domains, default_config)
        else:
            # We're in the main thread, safe to capture GUI state
            current_config = self._capture_analysis_config()
            return self._perform_cached_analysis_from_config(subject_key, data_segment, selected_domains, current_config)
    
    def _perform_cached_analysis_from_config(self, subject_key: str, data_segment: pd.DataFrame, selected_domains: List[HRVDomain], analysis_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Perform HRV analysis with intelligent caching (thread-safe version).
        
        Args:
            subject_key: Identifier for the subject/session
            data_segment: Data for this subject/session
            selected_domains: HRV domains to analyze
            analysis_config: Analysis configuration from main thread
            
        Returns:
            Analysis results dictionary or None if analysis failed
        """
        try:
            # Generate cache config from provided analysis config
            cache_config = {
                'fast_mode': analysis_config.get('fast_mode', False),
                'selected_domains': [domain.value for domain in selected_domains],
                'bootstrap_ci': analysis_config.get('bootstrap_ci', False) and not analysis_config.get('fast_mode', False),
                'max_bootstrap_samples': analysis_config.get('max_bootstrap_samples', 50),
                'analysis_timeout': analysis_config.get('analysis_timeout', 300),
                'cache_version': analysis_config.get('cache_version', '2.1')
            }
            
            # Extract subject ID and session info
            if '_Sol' in subject_key:
                subject_id = subject_key.split('_Sol')[0]
            else:
                subject_id = subject_key
            
            # Check cache first  
            logger.debug(f"Checking cache for {subject_key}")
            cached_result = self.results_cache.get(
                subject_id=subject_id,
                session_id=subject_key,
                data=data_segment,
                analysis_config=cache_config
            )
            
            if cached_result is not None:
                logger.info(f"🚀 Cache HIT: Using cached results for {subject_key}")
                return cached_result
            
            # Cache miss - perform fresh analysis
            logger.info(f"💾 Cache MISS: Computing fresh analysis for {subject_key}")
            
            # CRITICAL FIX: Smart memory management
            hr_data = data_segment['heart_rate [bpm]']
            hr_data = self._apply_memory_protection(hr_data, subject_key)
            
            # Signal processing with timeout
            def process_subject():
                rr_intervals, processing_info = self.signal_processor.compute_rr_intervals(hr_data)
                
                if len(rr_intervals) < 50:
                    logger.warning(f"Insufficient RR intervals for {subject_key}: {len(rr_intervals)}")
                    return None
                    
                # HRV analysis with performance optimizations
                include_ci = cache_config['bootstrap_ci']
                hrv_results = self.hrv_processor.compute_hrv_metrics(
                    rr_intervals,
                    domains=selected_domains,
                    include_confidence_intervals=include_ci
                )
                
                return {
                    'rr_intervals': rr_intervals,
                    'processing_info': processing_info,
                    'hrv_results': hrv_results,
                    'data_info': {
                        'original_samples': len(data_segment),
                        'processed_samples': len(hr_data),
                        'rr_intervals_count': len(rr_intervals)
                    },
                    'subject_key': subject_key
                }
            
            # Execute with timeout protection (reduced timeout for reliability)
            result = None
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(process_subject)
                try:
                    result = future.result(timeout=30)  # 30 second timeout per subject (reduced)
                except FutureTimeoutError:
                    logger.warning(f"⏱️ Timeout processing {subject_key} after 30 seconds, skipping...")
                    return None
            
            if result is None:
                return None
            
            # Cache the result for future use
            cache_metadata = {
                'analysis_timestamp': datetime.now().isoformat(),
                'original_samples': result['data_info']['original_samples'],
                'processed_samples': result['data_info']['processed_samples'],
                'rr_intervals': result['data_info']['rr_intervals_count'],
                'domains_analyzed': [domain.value for domain in selected_domains]
            }
            
            cache_success = self.results_cache.put(
                subject_id=subject_id,
                session_id=subject_key,
                data=data_segment,
                analysis_config=cache_config,
                result=result,
                ttl_hours=24.0,  # 24 hour cache TTL
                metadata=cache_metadata
            )
            
            if cache_success:
                logger.info(f"✅ Successfully cached results for {subject_key}")
            else:
                logger.warning(f"⚠️ Failed to cache results for {subject_key}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in cached analysis for {subject_key}: {e}")
            return None
    
    def _perform_cached_analysis_from_config_with_progress(self, subject_key: str, data_segment: pd.DataFrame, 
                                                          selected_domains: List[HRVDomain], analysis_config: Dict[str, Any], 
                                                          progress_callback) -> Optional[Dict[str, Any]]:
        """
        Perform HRV analysis with progress callback for GUI updates.
        
        Args:
            subject_key: Identifier for the subject/session
            data_segment: Data for this subject/session
            selected_domains: HRV domains to analyze
            analysis_config: Analysis configuration from main thread
            progress_callback: Function to call with progress updates (step_name, percentage)
            
        Returns:
            Analysis results dictionary or None if analysis failed
        """
        try:
            progress_callback("Starting analysis", 0)
            
            # Generate cache config from provided analysis config
            cache_config = {
                'fast_mode': analysis_config.get('fast_mode', False),
                'selected_domains': [domain.value for domain in selected_domains],
                'bootstrap_ci': analysis_config.get('bootstrap_ci', False) and not analysis_config.get('fast_mode', False),
                'max_bootstrap_samples': analysis_config.get('max_bootstrap_samples', 50),
                'analysis_timeout': analysis_config.get('analysis_timeout', 300),
                'cache_version': analysis_config.get('cache_version', '2.1')
            }
            
            # Extract subject ID and session info
            if '_Sol' in subject_key:
                subject_id = subject_key.split('_Sol')[0]
            else:
                subject_id = subject_key
            
            progress_callback("Checking cache", 5)
            
            # Check cache first  
            cached_result = self.results_cache.get(
                subject_id=subject_id,
                session_id=subject_key,
                data=data_segment,
                analysis_config=cache_config
            )
            
            if cached_result is not None:
                progress_callback("Using cached results", 100)
                logger.info(f"🚀 Cache HIT: Using cached results for {subject_key}")
                return cached_result
            
            progress_callback("Processing heart rate data", 10)
            logger.info(f"💾 Cache MISS: Computing fresh analysis for {subject_key}")
            
            # CRITICAL FIX: Smart memory management
            hr_data = data_segment['heart_rate [bpm]']
            hr_data = self._apply_memory_protection(hr_data, subject_key)
            
            progress_callback("Computing RR intervals", 20)
            
            # Signal processing
            rr_intervals, processing_info = self.signal_processor.compute_rr_intervals(hr_data)
            
            if len(rr_intervals) < 50:
                logger.warning(f"Insufficient RR intervals for {subject_key}: {len(rr_intervals)}")
                progress_callback("Insufficient data", 100)
                return None
            
            progress_callback("Starting HRV calculations", 30)
            
            # HRV analysis with progress updates
            include_ci = cache_config['bootstrap_ci']
            
            # Create a progress-aware HRV processor call
            def hrv_progress_callback(domain_name, domain_progress):
                # Map domain progress to overall progress (30-95%)
                overall_progress = 30 + (domain_progress * 0.65)
                progress_callback(f"Computing {domain_name} metrics", overall_progress)
            
            # Call HRV processor with progress updates
            hrv_results = self._compute_hrv_with_progress(
                rr_intervals, selected_domains, include_ci, hrv_progress_callback)
            
            progress_callback("Finalizing results", 95)
            
            result = {
                'rr_intervals': rr_intervals,
                'processing_info': processing_info,
                'hrv_results': hrv_results,
                'data_info': {
                    'original_samples': len(data_segment),
                    'processed_samples': len(hr_data),
                    'rr_intervals_count': len(rr_intervals)
                },
                'subject_key': subject_key
            }
            
            # Cache the result for future use
            cache_metadata = {
                'analysis_timestamp': datetime.now().isoformat(),
                'original_samples': result['data_info']['original_samples'],
                'processed_samples': result['data_info']['processed_samples'],
                'rr_intervals': result['data_info']['rr_intervals_count'],
                'domains_analyzed': [domain.value for domain in selected_domains]
            }
            
            cache_success = self.results_cache.put(
                subject_id=subject_id,
                session_id=subject_key,
                data=data_segment,
                analysis_config=cache_config,
                result=result,
                ttl_hours=24.0,  # 24 hour cache TTL
                metadata=cache_metadata
            )
            
            progress_callback("Complete", 100)
            
            if cache_success:
                logger.info(f"✅ Successfully cached results for {subject_key}")
            else:
                logger.warning(f"⚠️ Failed to cache results for {subject_key}")
            
            return result
            
        except Exception as e:
            progress_callback(f"Error: {str(e)[:20]}...", 100)
            logger.error(f"Error in cached analysis for {subject_key}: {e}")
            return None
    
    def _compute_hrv_with_progress(self, rr_intervals, selected_domains, include_ci, progress_callback):
        """Compute HRV metrics with progress updates."""
        try:
            results = {}
            total_domains = len(selected_domains)
            
            for i, domain in enumerate(selected_domains):
                domain_progress_start = (i / total_domains) * 100
                domain_progress_end = ((i + 1) / total_domains) * 100
                
                # Update progress for this domain
                progress_callback(domain.value, domain_progress_start)
                
                # Force GUI update before expensive computation
                self.root.update_idletasks()
                self.root.update()
                
                # Compute this domain's metrics
                try:
                    if domain == HRVDomain.NONLINEAR:
                        # Nonlinear is slow - update progress more frequently
                        progress_callback("Nonlinear (DFA)", domain_progress_start + 10)
                        self.root.update_idletasks()
                        self.root.update()
                        
                        progress_callback("Nonlinear (Entropy)", domain_progress_start + 30)
                        self.root.update_idletasks()
                        self.root.update()
                        
                    # Use timeout for individual domain computation
                    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            self.hrv_processor.compute_hrv_metrics,
                            rr_intervals,
                            domains=[domain],
                            include_confidence_intervals=include_ci
                        )
                        
                        # Different timeout for different domains
                        timeout = 60 if domain == HRVDomain.NONLINEAR else 15
                        domain_result = future.result(timeout=timeout)
                        
                        if domain_result and domain.value in domain_result:
                            results[domain.value] = domain_result[domain.value]
                            
                except (FutureTimeoutError, Exception) as e:
                    logger.warning(f"Timeout or error computing {domain.value}: {e}")
                    # Continue with other domains
                    
                # Update progress for completed domain
                progress_callback(domain.value, domain_progress_end)
                
            return results
            
        except Exception as e:
            logger.error(f"Error in HRV computation with progress: {e}")
            return {}
    
    def _get_cache_status_info(self) -> Dict[str, Any]:
        """Get current cache statistics for display."""
        try:
            return self.results_cache.get_cache_stats()
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}
    
    def _setup_performance_monitor(self):
        """Performance monitor disabled by user request."""
        # Performance monitor has been removed as it was not working properly
        self.performance_monitor = None
        logger.info("Performance monitor disabled")
    
    def _on_window_closing(self):
        """Handle window closing event properly."""
        try:
            logger.info("Window closing requested by user")
            
            # Check if analysis is running
            if self.analysis_running:
                result = messagebox.askyesnocancel(
                    "Analysis Running", 
                    "HRV analysis is currently running. \n\n" +
                    "⚠️ WARNING: Closing now will stop the analysis and lose progress.\n\n" +
                    "What would you like to do?\n" +
                    "• Yes: Stop analysis and close application\n" +
                    "• No: Keep application open and continue analysis\n" +
                    "• Cancel: Return to application"
                )
                
                if result is True:  # Yes - close anyway
                    logger.info("User confirmed stopping analysis and closing")
                    self.analysis_running = False
                    # Force stop any ongoing analysis
                    if hasattr(self, 'current_analysis_tasks'):
                        self.current_analysis_tasks.clear()
                elif result is False:  # No - keep open
                    logger.info("User chose to keep application open")
                    return  # Don't close
                else:  # Cancel - do nothing
                    logger.info("User cancelled window closing")
                    return  # Don't close
            
            # Clean shutdown
            logger.info("Closing application - no analysis running")
            
            # Disable GUI updates to prevent errors during shutdown
            self._gui_active = False
            self._shutdown_in_progress = True
            
            # Clear any remaining variables to prevent "no default root window" errors
            try:
                if hasattr(self, 'plot_subject_var'):
                    del self.plot_subject_var
                if hasattr(self, 'emergency_subject_var'):
                    del self.emergency_subject_var
            except:
                pass
            
            # Close any open file handles or connections
            try:
                if hasattr(self, 'results_cache'):
                    self.results_cache.cleanup()
            except:
                pass
            
            # Force garbage collection before closing
            import gc
            gc.collect()
            
            # Destroy the root window properly
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            logger.error(f"Error during window closing: {e}")
            # Force close anyway
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass
    
    def _shutdown_analysis(self):
        """Properly shutdown analysis and async processor."""
        try:
            # Set GUI as disconnected
            self._gui_active = False
            
            # Shutdown async processor (if enabled)
            if hasattr(self, 'async_processor') and self.async_processor is not None:
                if hasattr(self.async_processor, '_is_running') and self.async_processor._is_running:
                    logger.info("Shutting down async processor...")
                    
                    # Set GUI connection status to false to prevent callback errors
                    self.async_processor.set_gui_connection_status(False)
                    
                    # Update status safely
                    try:
                        if hasattr(self, 'async_status_var'):
                            self.async_status_var.set("Shutting down...")
                    except Exception as e:
                        logger.debug(f"Could not update status during shutdown: {e}")
                    
                    # Stop processor
                    self.async_processor.shutdown(wait_for_completion=False, timeout=10.0)
                    logger.info("Analysis shutdown complete")
            
            # Reset state
            self.analysis_running = False
            try:
                if hasattr(self, 'process_button'):
                    self.process_button.configure(state='normal')
                if hasattr(self, 'simple_process_button'):
                    self.simple_process_button.configure(state='normal')
            except Exception as e:
                logger.debug(f"Could not update buttons during shutdown: {e}")
            
            # Shutdown performance monitor
            if hasattr(self, 'performance_monitor') and self.performance_monitor:
                logger.info("Shutting down performance monitor...")
                self.performance_monitor.set_gui_connection_status(False)
                self.performance_monitor.shutdown()
                logger.info("Performance monitor shutdown complete")
                
        except Exception as e:
            logger.error(f"Error shutting down analysis: {e}")
    
    def _on_window_focus(self, event):
        """Handle window focus event."""
        if event.widget == self.root:  # Only handle main window events
            if not self._gui_active:
                self._gui_active = True
                if hasattr(self, 'async_processor') and self.async_processor is not None:
                    self.async_processor.set_gui_connection_status(True)
                if hasattr(self, 'performance_monitor') and self.performance_monitor:
                    self.performance_monitor.set_gui_connection_status(True)
                logger.info("GUI reconnected - resuming normal operation")
                
                # Check for any pending notifications
                self._check_pending_notifications()
    
    def _on_window_unfocus(self, event):
        """Handle window unfocus event."""
        if event.widget == self.root:  # Only handle main window events
            # Don't immediately disconnect - user might just be switching windows
            # Only disconnect if window is actually closed/minimized
            pass
    
    def _generate_individual_mission_phases(self):
        """Generate individual mission phases boxplots."""
        try:
            if not self.analysis_results:
                messagebox.showwarning("Warning", "Please run an analysis first to generate mission phases boxplots.")
                return
            
            if not self.mission_phases_generator:
                messagebox.showerror("Error", "Mission phases boxplot generator is not available.")
                return
            
            self.plot_status_label.configure(text="Generating individual mission phases boxplots...")
            self.root.update_idletasks()
            
            # Prepare data from analysis results
            df, mission_phases = self.mission_phases_generator.prepare_mission_data(self.analysis_results)
            
            # Generate individual boxplots
            individual_plot_path = self.mission_phases_generator.generate_individual_boxplots(df, mission_phases)
            
            success_text = f"✅ Individual mission phases boxplots generated!\nSaved to: {individual_plot_path}"
            self.plot_status_label.configure(text=success_text)
            
            # Open the plot file
            self._open_plot_file(Path(individual_plot_path))
            
        except Exception as e:
            logger.error(f"Error generating individual mission phases boxplots: {e}")
            self.plot_status_label.configure(text=f"❌ Error: {e}")
            messagebox.showerror("Error", f"Failed to generate individual mission phases boxplots: {e}")
    
    def _generate_group_mission_phases(self):
        """Generate group mission phases boxplots."""
        try:
            if not self.analysis_results:
                messagebox.showwarning("Warning", "Please run an analysis first to generate mission phases boxplots.")
                return
            
            if not self.mission_phases_generator:
                messagebox.showerror("Error", "Mission phases boxplot generator is not available.")
                return
            
            self.plot_status_label.configure(text="Generating group mission phases boxplots...")
            self.root.update_idletasks()
            
            # Prepare data from analysis results
            df, mission_phases = self.mission_phases_generator.prepare_mission_data(self.analysis_results)
            
            # Generate group boxplots
            group_plot_path = self.mission_phases_generator.generate_group_boxplots(df, mission_phases)
            
            success_text = f"✅ Group mission phases boxplots generated!\nSaved to: {group_plot_path}"
            self.plot_status_label.configure(text=success_text)
            
            # Open the plot file
            self._open_plot_file(Path(group_plot_path))
            
        except Exception as e:
            logger.error(f"Error generating group mission phases boxplots: {e}")
            self.plot_status_label.configure(text=f"❌ Error: {e}")
            messagebox.showerror("Error", f"Failed to generate group mission phases boxplots: {e}")
    
    def _generate_mission_phases_report(self):
        """Generate comprehensive mission phases analysis report."""
        try:
            if not self.analysis_results:
                messagebox.showwarning("Warning", "Please run an analysis first to generate mission phases report.")
                return
            
            if not self.mission_phases_generator:
                messagebox.showerror("Error", "Mission phases boxplot generator is not available.")
                return
            
            self.plot_status_label.configure(text="Generating mission phases analysis (individual + group + report)...")
            self.root.update_idletasks()
            
            # Prepare data from analysis results
            df, mission_phases = self.mission_phases_generator.prepare_mission_data(self.analysis_results)
            
            # Generate both plots
            individual_plot_path = self.mission_phases_generator.generate_individual_boxplots(df, mission_phases)
            group_plot_path = self.mission_phases_generator.generate_group_boxplots(df, mission_phases)
            
            # Generate comprehensive report
            report_path = self.mission_phases_generator.generate_comprehensive_report(
                df, mission_phases, individual_plot_path, group_plot_path
            )
            
            success_text = f"✅ Complete mission phases analysis generated!\n"
            success_text += f"Individual plots: {individual_plot_path}\n"
            success_text += f"Group plots: {group_plot_path}\n"
            success_text += f"Report: {report_path}"
            self.plot_status_label.configure(text=success_text)
            
            # Show completion message
            completion_message = (
                "🎉 Mission Phases Analysis Complete!\n\n"
                "Generated Files:\n"
                f"• Individual Boxplots: {Path(individual_plot_path).name}\n"
                f"• Group Boxplots: {Path(group_plot_path).name}\n"
                f"• Comprehensive Report: {Path(report_path).name}\n\n"
                "All files saved to plots_output/ folder.\n"
                "The plots and report will open automatically."
            )
            
            messagebox.showinfo("Analysis Complete", completion_message)
            
            # Open the plot files and report
            self._open_plot_file(Path(individual_plot_path))
            self._open_plot_file(Path(group_plot_path))
            self._open_plot_file(Path(report_path))
            
        except Exception as e:
            logger.error(f"Error generating mission phases report: {e}")
            self.plot_status_label.configure(text=f"❌ Error: {e}")
            messagebox.showerror("Error", f"Failed to generate mission phases analysis: {e}")

    def _show_hrv_explanations(self):
        """Show comprehensive HRV metrics scientific explanations."""
        try:
            if show_hrv_explanations is None:
                messagebox.showerror(
                    "Feature Unavailable",
                    "HRV explanations module is not available.\n"
                    "Please check your installation."
                )
                return
            
            # Pass current analysis results if available for context
            metrics_data = {}
            if self.analysis_results:
                # Extract some basic metrics for context
                for subject, result in self.analysis_results.items():
                    if isinstance(result, dict) and 'hrv_results' in result:
                        metrics_data[subject] = result['hrv_results']
            
            show_hrv_explanations(self.root, metrics_data)
            
        except Exception as e:
            logger.error(f"Error showing HRV explanations: {e}")
            messagebox.showerror(
                "Error",
                f"Could not open HRV explanations: {e}"
            )
    
    def _show_hrv_citations(self):
        """Show HRV reference ranges with scientific citations."""
        try:
            if show_hrv_citations is None:
                messagebox.showerror(
                    "Feature Unavailable",
                    "HRV citations module is not available.\n"
                    "Please check your installation."
                )
                return
            
            show_hrv_citations(self.root)
            logger.info("HRV citations displayed")
        except Exception as e:
            logger.error(f"Error showing HRV citations: {e}")
            messagebox.showerror(
                "Error",
                f"Could not open HRV citations: {e}"
            )


def main():
    """Main entry point for the application.""" 
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('hrv_analysis.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create and run application
    root = tk.Tk()
    app = HRVAnalysisApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        messagebox.showerror("Fatal Error", f"Application error: {e}")
        
if __name__ == "__main__":
    main() 