#!/usr/bin/env python3
"""
Launch Script for Enhanced HRV Analysis App with Mission Phases Boxplots

This script launches the Enhanced HRV Analysis application with the newly
integrated mission phases boxplot functionality for comparing crew physiological
adaptation across Early, Mid, and Late mission phases.

Usage: python launch_enhanced_hrv_app.py

Features:
- Complete HRV Analysis System
- Individual crew member boxplots across mission phases
- Group boxplots comparing all crew members
- Comprehensive statistical analysis and reporting
- Integration with Valquiria space analog simulation data

Author: AI Assistant
Date: 2025-01-14
Integration: Enhanced HRV Analysis System
"""

import sys
from pathlib import Path
import logging


def setup_environment():
    """Setup the environment for running the Enhanced HRV Analysis app."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Add the enhanced HRV analysis path to Python path
    hrv_analysis_path = script_dir / "src" / "hrv_analysis" / "enhanced_hrv_analysis"
    
    if hrv_analysis_path.exists():
        sys.path.insert(0, str(hrv_analysis_path))
        sys.path.insert(0, str(hrv_analysis_path.parent))
        print(f"✅ Added {hrv_analysis_path} to Python path")
    else:
        print(f"❌ Enhanced HRV Analysis path not found: {hrv_analysis_path}")
        print("Please ensure you're running this from the project root "
              "directory.")
        return False
    
    # Ensure required directories exist
    required_dirs = [
        script_dir / "plots_output",
        hrv_analysis_path / "hrv_cache",
    ]
    
    for dir_path in required_dirs:
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Ensured directory exists: {dir_path}")
    
    return True


def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        'tkinter',
        'pandas', 
        'numpy',
        'matplotlib',
        'seaborn',
        'scipy',
        'pathlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package)
            print(f"✅ {package} - Available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print("pip install " + " ".join(missing_packages))
        return False
    
    print("\n✅ All required dependencies are available!")
    return True

def launch_app():
    """Launch the Enhanced HRV Analysis application."""
    try:
        print("\n🚀 Launching Enhanced HRV Analysis App with Mission Phases Boxplots...")
        
        # Import and run the application
        from gui.main_application import HRVAnalysisApp
        import tkinter as tk
        
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
        print("🎯 Initializing GUI...")
        root = tk.Tk()
        app = HRVAnalysisApp(root)
        
        print("✅ Enhanced HRV Analysis App launched successfully!")
        print("\n📊 NEW FEATURE: Mission Phases Boxplots")
        print("   Navigate to the 'Visualizations' tab and look for:")
        print("   • Mission Phases - Individual")
        print("   • Mission Phases - Group") 
        print("   • Mission Phases - Report")
        print("\n🔬 These tools compare physiological adaptation across Early, Mid, and Late mission phases")
        print("\n🚪 Close this window or press Ctrl+C to exit the application")
        
        # Start the GUI main loop
        root.mainloop()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure you're running from the correct directory with all required files.")
        return False
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        return False
    
    print("\n👋 Enhanced HRV Analysis App closed")
    return True

def main():
    """Main entry point for the launch script."""
    print("="*60)
    print("Enhanced HRV Analysis App Launcher")
    print("Mission Phases Boxplots Integration")
    print("="*60)
    
    # Setup environment
    print("\n📁 Setting up environment...")
    if not setup_environment():
        sys.exit(1)
    
    # Check dependencies  
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Launch the application
    success = launch_app()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 