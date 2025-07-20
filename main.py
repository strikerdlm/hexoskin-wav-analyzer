#!/usr/bin/env python3
"""
Valquiria Data Analysis Suite - Main Entry Point

This script provides easy access to both analysis systems:
1. Hexoskin WAV File Analyzer
2. Enhanced HRV Analysis System

Author: Dr. Diego Malpica MD - Aerospace Medicine Specialist
Organization: Colombian Aerospace Force (FAC) / DIMAE
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

def show_banner():
    """Display the application banner."""
    banner = """
    ╔════════════════════════════════════════════════════════════════╗
    ║           Valquiria Space Analog Data Analysis Suite          ║
    ║                                                                ║
    ║   Author: Dr. Diego Malpica MD - Aerospace Medicine           ║
    ║   Organization: Colombian Aerospace Force (FAC) / DIMAE       ║
    ║                                                                ║
    ║   🚀 Advanced Physiological Data Analysis for Space Research  ║
    ╚════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def launch_hexoskin_analyzer():
    """Launch the Hexoskin WAV File Analyzer."""
    try:
        from hexoskin_analyzer.hexoskin_wav_loader import main as hexoskin_main
        print("🔬 Starting Hexoskin WAV File Analyzer...")
        hexoskin_main()
    except ImportError as e:
        print(f"❌ Error importing Hexoskin analyzer: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting Hexoskin analyzer: {e}")
        sys.exit(1)

def launch_hrv_analysis():
    """Launch the Enhanced HRV Analysis System."""
    try:
        from hrv_analysis.launch_hrv_analysis import main as hrv_main
        print("❤️  Starting Enhanced HRV Analysis System...")
        hrv_main()
    except ImportError as e:
        print(f"❌ Error importing HRV analysis system: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting HRV analysis system: {e}")
        sys.exit(1)

def run_example(wav_file):
    """Run example analysis on a WAV file."""
    try:
        # Import the example script from the examples directory
        examples_path = Path(__file__).parent / 'examples'
        sys.path.insert(0, str(examples_path))
        
        from hexoskin_wav_example import main as example_main
        print(f"📊 Running example analysis on: {wav_file}")
        example_main(wav_file)
    except ImportError as e:
        print(f"❌ Error importing example script: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running example analysis: {e}")
        sys.exit(1)

def show_system_info():
    """Show system and dependency information."""
    print("🖥️  System Information:")
    print(f"   Python version: {sys.version}")
    print(f"   Python path: {sys.executable}")
    print(f"   Working directory: {os.getcwd()}")
    
    # Check key dependencies
    dependencies = [
        'numpy', 'pandas', 'scipy', 'matplotlib', 'tkinter'
    ]
    
    print("\n📦 Dependency Status:")
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} - Not installed")
    
    # Check optional dependencies
    optional_deps = [
        'numba', 'plotly', 'sklearn', 'statsmodels'
    ]
    
    print("\n📦 Optional Dependencies:")
    for dep in optional_deps:
        try:
            if dep == 'sklearn':
                __import__('sklearn')
            else:
                __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ⚠️  {dep} - Not installed (optional)")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Valquiria Data Analysis Suite - Main Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py hexoskin              # Launch Hexoskin WAV analyzer GUI
  python main.py hrv                   # Launch Enhanced HRV analysis GUI
  python main.py example file.wav     # Run example analysis on WAV file
  python main.py info                  # Show system information
  
For more information, visit the docs/ directory.
        """
    )
    
    parser.add_argument(
        'command',
        choices=['hexoskin', 'hrv', 'example', 'info'],
        help='Which analysis system to launch'
    )
    
    parser.add_argument(
        'file',
        nargs='?',
        help='WAV file path (required for example command)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Valquiria Data Analysis Suite v2.0.0'
    )
    
    # Show banner
    show_banner()
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()
    
    # Execute commands
    if args.command == 'hexoskin':
        launch_hexoskin_analyzer()
    
    elif args.command == 'hrv':
        launch_hrv_analysis()
    
    elif args.command == 'example':
        if not args.file:
            print("❌ Error: WAV file path required for example command")
            print("Usage: python main.py example <wav_file>")
            sys.exit(1)
        
        if not Path(args.file).exists():
            print(f"❌ Error: WAV file not found: {args.file}")
            sys.exit(1)
        
        run_example(args.file)
    
    elif args.command == 'info':
        show_system_info()

if __name__ == '__main__':
    main() 