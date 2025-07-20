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

# Add current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

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

def main():
    """Main entry point for the application."""
    logger = setup_logging()
    
    try:
        # Check dependencies
        if not check_dependencies():
            logger.error("Cannot start application due to missing dependencies")
            safe_print("Press Enter to exit...")
            input()
            return 1
        
        logger.info("All required dependencies found")
        
        # Import and launch the GUI
        try:
            from gui.main_application import HRVAnalysisApp
            
            # Create the main window
            root = tk.Tk()
            
            # Set window icon if available
            try:
                # Look for icon file
                icon_path = current_dir / "assets" / "hrv_icon.ico"
                if icon_path.exists():
                    root.iconbitmap(str(icon_path))
            except Exception:
                pass  # Icon not critical
            
            # Create and run the application
            logger.info("Initializing HRV Analysis GUI...")
            app = HRVAnalysisApp(root)
            
            logger.info("GUI initialized successfully")
            logger.info("Starting main event loop...")
            
            # Show startup message
            safe_print("\n" + "="*60)
            safe_print("ENHANCED HRV ANALYSIS - ALL SUBJECTS MODE")
            safe_print("="*60)
            safe_print("[OK] Analyze All Subjects: Enabled by default")
            safe_print("[OK] Bootstrap CI: Disabled by default (can enable if needed)")
            safe_print("[OK] Timeout Protection: 5 minutes per analysis")
            safe_print("[OK] Memory Protection: Limits extremely large datasets")
            safe_print("[OK] Data Limiting: Available as option if needed")
            safe_print("="*60)
            safe_print("Ready to analyze ALL Valquiria subjects!")
            safe_print("="*60 + "\n")
            
            # Start the GUI main loop
            root.mainloop()
            
        except ImportError as e:
            logger.error(f"Failed to import GUI components: {e}")
            logger.error("Make sure all files are in the correct location")
            return 1
        except Exception as e:
            logger.error(f"Error starting GUI: {e}")
            return 1
            
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