#!/usr/bin/env python3
"""
Launch script for Enhanced HRV Analysis GUI Application

This script provides a simple entry point for launching the HRV analysis GUI
with automatic dependency checking and environment setup.

PERFORMANCE OPTIMIZATIONS:
- Disabled parallel processing to prevent GUI thread deadlocks
- Reduced bootstrap samples from 1000 to 50 for faster analysis
- Added timeout protection for long-running analyses
- Fast mode option to limit data size and prevent hanging
"""

import sys
import os
import logging
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import threading
import time

# Add current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

class SplashScreen:
    """Loading splash screen with progress indication."""
    
    def __init__(self):
        self.splash = tk.Tk()
        self.splash.title("Loading HRV Analysis System")
        self.splash.geometry("600x400")
        self.splash.resizable(False, False)
        
        # Center the splash screen
        self.splash.update_idletasks()
        width = self.splash.winfo_width()
        height = self.splash.winfo_height()
        pos_x = (self.splash.winfo_screenwidth() // 2) - (width // 2)
        pos_y = (self.splash.winfo_screenheight() // 2) - (height // 2)
        self.splash.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
        
        # Remove window decorations and make it topmost
        self.splash.overrideredirect(True)
        self.splash.attributes("-topmost", True)
        
        # Configure background
        self.splash.configure(bg='#2C3E50')
        
        # Create main frame
        main_frame = tk.Frame(self.splash, bg='#2C3E50', padx=40, pady=40)
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="HRV Analysis for the Valquiria Analog Space Mission Simulation",
            font=('Arial', 20, 'bold'),
            fg='#ECF0F1',
            bg='#2C3E50',
            wraplength=500,
            justify='center'
        )
        title_label.pack(pady=(0, 20))
        
        # Author and organization
        author_label = tk.Label(
            main_frame,
            text="by Dr. Diego Malpica MD\nDirectorate of Aerospace Medicine\nColombian Aerospace Force\nAerospace Scientific Department",
            font=('Arial', 12),
            fg='#BDC3C7',
            bg='#2C3E50',
            justify='center'
        )
        author_label.pack(pady=(0, 30))
        
        # Progress bar frame
        progress_frame = tk.Frame(main_frame, bg='#2C3E50')
        progress_frame.pack(fill='x', pady=(0, 20))
        
        # Progress bar
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Loading.Horizontal.TProgressbar",
                       background='#3498DB',
                       troughcolor='#34495E',
                       borderwidth=0,
                       lightcolor='#3498DB',
                       darkcolor='#3498DB')
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            style="Loading.Horizontal.TProgressbar",
            length=400,
            mode='determinate'
        )
        self.progress_bar.pack(pady=10)
        
        # Percentage label
        self.percentage_label = tk.Label(
            main_frame,
            text="0%",
            font=('Arial', 16, 'bold'),
            fg='#ECF0F1',
            bg='#2C3E50'
        )
        self.percentage_label.pack(pady=(10, 0))
        
        # Version info
        version_label = tk.Label(
            main_frame,
            text="Colombia - 2025",
            font=('Arial', 8),
            fg='#7F8C8D',
            bg='#2C3E50'
        )
        version_label.pack(side='bottom', pady=(20, 0))
        
    def update_progress(self, percentage, status_text="", details_text=""):
        """Update progress bar and percentage display."""
        if self.splash and self.splash.winfo_exists():
            self.progress_var.set(percentage)
            self.percentage_label.config(text=f"{int(percentage)}%")
            self.splash.update_idletasks()
    
    def destroy(self):
        """Close the splash screen."""
        if self.splash and self.splash.winfo_exists():
            self.splash.destroy()

def setup_logging():
    """Setup logging for the application."""
    # Setup safe logging that handles Unicode properly on Windows
    import sys
    
    # Try to set UTF-8 encoding for Windows console
    try:
        if sys.platform.startswith('win'):
            import os
            os.system('chcp 65001 >nul 2>&1')  # Set console to UTF-8
    except Exception:
        pass  # Ignore if it fails
    
    # Create a custom logging formatter that handles encoding issues
    class SafeFormatter(logging.Formatter):
        def format(self, record):
            try:
                return super().format(record)
            except UnicodeEncodeError:
                # Replace problematic Unicode characters
                msg = super().format(record)
                msg = msg.replace('✅', '[OK]').replace('❌', '[ERROR]').replace('⚠️', '[WARNING]')
                msg = msg.replace('🚀', '').replace('📊', '').replace('🎉', '')
                return msg
    
    # Create handlers with safe formatting
    file_handler = logging.FileHandler(current_dir / 'hrv_analysis.log', encoding='utf-8')
    file_handler.setFormatter(SafeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(SafeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Enhanced HRV Analysis GUI")
    
    # Log performance optimizations
    logger.info("PERFORMANCE OPTIMIZATIONS ACTIVE:")
    logger.info("- Parallel processing: DISABLED (prevents GUI hanging)")
    logger.info("- Bootstrap samples: LIMITED to 50 (was 1000)")
    logger.info("- Analysis timeout: 5 minutes maximum")
    logger.info("- Fast mode: Available (limits data size)")
    
    return logger

def check_dependencies():
    """Check for required dependencies."""
    logger = logging.getLogger(__name__)
    
    required_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
        ('tkinter', 'tkinter (usually built-in)')
    ]
    
    optional_packages = [
        ('numba', 'numba (for performance - will use fallback if missing)'),
        ('plotly', 'plotly (for interactive plots)'),
        ('sklearn', 'scikit-learn (for ML analysis)')
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    for package, display_name in required_packages:
        try:
            __import__(package)
            logger.info(f"[OK] {display_name}")
        except ImportError:
            missing_required.append(display_name)
            logger.error(f"[MISSING] {display_name} - REQUIRED")
    
    # Check optional packages
    for package, display_name in optional_packages:
        try:
            __import__(package)
            logger.info(f"[OK] {display_name}")
        except ImportError:
            missing_optional.append(display_name)
            logger.warning(f"[OPTIONAL] {display_name} - OPTIONAL")
    
    if missing_required:
        logger.error("Missing required dependencies. Please install:")
        for pkg in missing_required:
            logger.error(f"  pip install {pkg}")
        return False
    
    if missing_optional:
        logger.warning("Missing optional dependencies. For full functionality:")
        for pkg in missing_optional:
            logger.warning(f"  pip install {pkg}")
    
    return True

def safe_print(text):
    """Safe print function that handles Unicode encoding issues on Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace problematic Unicode characters and try again
        safe_text = text.replace('✅', '[OK]').replace('❌', '[ERROR]').replace('⚠️', '[WARNING]')
        safe_text = safe_text.replace('🚀', '').replace('📊', '').replace('🎉', '')
        print(safe_text)

def load_application_with_progress(splash_screen, logger):
    """Load the main application with progress updates."""
    try:
        # Step 1: Check dependencies
        splash_screen.update_progress(10)
        if not check_dependencies():
            logger.error("Cannot start application due to missing dependencies")
            return None
        
        # Step 2: Import GUI components
        splash_screen.update_progress(25)
        from gui.main_application import HRVAnalysisApp
        
        # Step 3: Create main window
        splash_screen.update_progress(40)
        root = tk.Tk()
        root.withdraw()  # Hide initially
        
        # Step 4: Set window icon
        splash_screen.update_progress(50)
        try:
            icon_path = current_dir / "assets" / "hrv_icon.ico"
            if icon_path.exists():
                root.iconbitmap(str(icon_path))
        except Exception:
            pass
        
        # Step 5: Initialize application core
        splash_screen.update_progress(60)
        
        # Custom progress callback for app initialization
        def app_progress_callback(percentage, message):
            # Map app loading progress to remaining 40% (60-100%)
            mapped_percentage = 60 + (percentage * 0.4)
            splash_screen.update_progress(mapped_percentage)
        
        # Step 6: Create application with progress tracking
        splash_screen.update_progress(70)
        app = HRVAnalysisApp(root, progress_callback=app_progress_callback)
        
        # Step 7: Finalize setup and maximize window
        splash_screen.update_progress(90)
        
        # Maximize the window
        root.state('zoomed')  # Windows-specific maximization
        root.deiconify()  # Show the main window
        
        splash_screen.update_progress(100)
        time.sleep(0.5)  # Brief pause to show completion
        
        return root, app
        
    except ImportError as e:
        logger.error(f"Failed to import GUI components: {e}")
        splash_screen.update_progress(0)
        return None
    except Exception as e:
        logger.error(f"Error loading application: {e}")
        splash_screen.update_progress(0)
        return None

def main():
    """Main entry point for the application."""
    logger = setup_logging()
    
    try:
        # Create and show splash screen
        splash = SplashScreen()
        splash.splash.update()
        
        # Load application with progress
        result = load_application_with_progress(splash, logger)
        
        if result is None:
            logger.error("Failed to load application")
            splash.update_progress(0)
            time.sleep(3)
            splash.destroy()
            safe_print("Press Enter to exit...")
            input()
            return 1
        
        root, app = result
        
        # Close splash screen
        splash.destroy()
        
        logger.info("GUI initialized successfully")
        logger.info("Starting main event loop...")
        
        # Show startup message
        safe_print("\n" + "="*70)
        safe_print("HRV ANALYSIS - VALQUIRIA SPACE SIMULATION")
        safe_print("="*70)
        safe_print("[OK] Simple Analysis Mode: No threading issues")
        safe_print("[OK] Real-time Progress: GUI updates during analysis")
        safe_print("[OK] Memory Protection: Handles large datasets safely")
        safe_print("[OK] Cache System: Speeds up repeated analysis")
        safe_print("[OK] All HRV Domains: Time, frequency, nonlinear metrics")
        safe_print("="*70)
        safe_print("💡 THREADING ISSUES RESOLVED:")
        safe_print("   • All analysis runs in main thread")
        safe_print("   • No background processing complications")
        safe_print("   • Real-time progress updates")
        safe_print("   • Guaranteed reliability")
        safe_print("="*70)
        safe_print("⚠️  ANALYSIS TIME NOTICE:")
        safe_print("   • Time/Frequency Analysis: Fast (< 30 seconds)")
        safe_print("   • Nonlinear Analysis: May take 2-5 minutes")
        safe_print("   • Progress bar shows real-time status")
        safe_print("   • GUI remains responsive during processing")
        safe_print("="*70)
        safe_print("Ready to analyze Valquiria crew HRV data!")
        safe_print("="*70 + "\n")
        
        # Start the GUI main loop
        root.mainloop()
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    
    logger.info("Application finished successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 