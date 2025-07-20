#!/usr/bin/env python3
"""
Enhanced HRV Analysis System Launcher

This script provides an easy way to launch the enhanced HRV analysis system
with various options for GUI, command-line, or batch processing modes.

Usage:
    python run_enhanced_hrv_analysis.py [options]
    
Options:
    --gui               Launch the GUI application (default)
    --cli               Run in command-line mode
    --test              Run the test suite
    --demo              Run with sample data demonstration
    --batch <config>    Run batch analysis with configuration file
    --install-deps      Install required dependencies
    --version           Show version information
    --help              Show this help message

Examples:
    python run_enhanced_hrv_analysis.py --gui
    python run_enhanced_hrv_analysis.py --test
    python run_enhanced_hrv_analysis.py --demo
    python run_enhanced_hrv_analysis.py --batch config.json
"""

import sys
import os
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Optional

# Add enhanced_hrv_analysis to path
current_dir = Path(__file__).parent
enhanced_path = current_dir / "enhanced_hrv_analysis"
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(enhanced_path))

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'scikit-learn',
        'plotly', 'statsmodels', 'joblib'
    ]
    
    optional_packages = [
        'hrvanalysis', 'pingouin', 'pmdarima', 'prophet', 'hdbscan', 'umap-learn'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_required.append(package)
            
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_optional.append(package)
            
    return missing_required, missing_optional

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    requirements_file = current_dir / "requirements.txt"
    
    if requirements_file.exists():
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    else:
        print("❌ requirements.txt not found")
        return False

def launch_gui():
    """Launch the GUI application."""
    try:
        print("🚀 Launching Enhanced HRV Analysis GUI...")
        
        # Check critical dependencies
        missing_required, missing_optional = check_dependencies()
        
        if missing_required:
            print(f"❌ Missing required dependencies: {missing_required}")
            print("Run with --install-deps to install them")
            return False
            
        if missing_optional:
            print(f"⚠️ Missing optional dependencies: {missing_optional}")
            print("Some features may be limited")
            
        # Import and run GUI
        import tkinter as tk
        from enhanced_hrv_analysis.gui.main_application import HRVAnalysisApp
        
        root = tk.Tk()
        app = HRVAnalysisApp(root)
        root.mainloop()
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed (run with --install-deps)")
        return False
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        return False

def run_command_line(data_path: Optional[str] = None):
    """Run command-line analysis."""
    try:
        print("🔧 Running Enhanced HRV Analysis (Command Line)...")
        
        from enhanced_hrv_analysis.core.data_loader import DataLoader
        from enhanced_hrv_analysis.core.signal_processing import SignalProcessor
        from enhanced_hrv_analysis.core.hrv_processor import HRVProcessor, HRVDomain
        
        # Initialize components
        data_loader = DataLoader()
        signal_processor = SignalProcessor()
        hrv_processor = HRVProcessor()
        
        # Load data
        if data_path:
            if data_path.endswith('.db'):
                data = data_loader.load_database_data(data_path)
            else:
                data = data_loader.load_csv_data(data_dir=data_path)
        else:
            print("No data path provided, using sample data...")
            data = DataLoader.create_sample_data()
            
        if data is None:
            print("❌ Failed to load data")
            return False
            
        print(f"✅ Loaded {len(data)} records")
        
        # Process first subject as example
        if 'subject' in data.columns:
            first_subject = data['subject'].iloc[0]
            subject_data = data[data['subject'] == first_subject]
        else:
            subject_data = data
            
        # Signal processing
        print("🔍 Processing RR intervals...")
        rr_intervals, processing_info = signal_processor.compute_rr_intervals(
            subject_data['heart_rate [bpm]']
        )
        
        if len(rr_intervals) < 50:
            print("❌ Insufficient RR intervals for analysis")
            return False
            
        print(f"✅ Processed {len(rr_intervals)} RR intervals")
        
        # HRV analysis
        print("📊 Computing HRV metrics...")
        hrv_results = hrv_processor.compute_hrv_metrics(
            rr_intervals,
            domains=[HRVDomain.TIME, HRVDomain.FREQUENCY, HRVDomain.NONLINEAR]
        )
        
        # Display results
        print("\n" + "="*50)
        print("HRV ANALYSIS RESULTS")
        print("="*50)
        
        if 'time_domain' in hrv_results:
            td = hrv_results['time_domain']
            print(f"\nTime Domain Metrics:")
            print(f"  SDNN: {td.get('sdnn', 0):.1f} ms")
            print(f"  RMSSD: {td.get('rmssd', 0):.1f} ms")
            print(f"  pNN50: {td.get('pnn50', 0):.1f}%")
            print(f"  Mean HR: {td.get('mean_hr', 0):.1f} BPM")
            
        if 'frequency_domain' in hrv_results:
            fd = hrv_results['frequency_domain']
            print(f"\nFrequency Domain Metrics:")
            print(f"  LF Power: {fd.get('lf_power', 0):.0f} ms²")
            print(f"  HF Power: {fd.get('hf_power', 0):.0f} ms²")
            print(f"  LF/HF Ratio: {fd.get('lf_hf_ratio', 0):.2f}")
            
        if 'quality_assessment' in hrv_results:
            qa = hrv_results['quality_assessment']
            print(f"\nQuality Assessment:")
            print(f"  Data Quality: {qa.get('data_quality', 'unknown')}")
            print(f"  Analysis Reliability: {qa.get('analysis_reliability', 'unknown')}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error in command-line analysis: {e}")
        return False

def run_tests():
    """Run the test suite.""" 
    try:
        print("🧪 Running Enhanced HRV Analysis Test Suite...")
        
        # Check if pytest is available
        try:
            import pytest
        except ImportError:
            print("❌ pytest not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest"])
            import pytest
            
        # Run tests
        test_dir = enhanced_path / "tests"
        if test_dir.exists():
            exit_code = pytest.main([str(test_dir), "-v", "--tb=short"])
            return exit_code == 0
        else:
            print("❌ Test directory not found")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def run_demo():
    """Run demonstration with sample data."""
    try:
        print("🎯 Running Enhanced HRV Analysis Demo...")
        
        from enhanced_hrv_analysis.core.data_loader import DataLoader
        from enhanced_hrv_analysis.core.hrv_processor import HRVProcessor, HRVDomain
        from enhanced_hrv_analysis.ml_analysis.clustering import HRVClustering
        from enhanced_hrv_analysis.visualization.interactive_plots import InteractivePlotter
        
        # Generate sample data
        print("📊 Generating sample data...")
        data_loader = DataLoader()
        sample_data = data_loader.create_sample_data(n_subjects=5, n_sols=4)
        
        print(f"✅ Generated {len(sample_data)} data points for {sample_data['subject'].nunique()} subjects")
        
        # Process all subjects
        print("🔍 Processing all subjects...")
        hrv_processor = HRVProcessor()
        
        all_results = {}
        for subject in sample_data['subject'].unique():
            subject_data = sample_data[sample_data['subject'] == subject]
            
            # Combine all SOLs for this subject
            combined_hr = subject_data['heart_rate [bpm]']
            
            # Quick RR conversion for demo
            rr_intervals = 60000 / combined_hr.dropna()
            rr_intervals = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
            
            if len(rr_intervals) > 50:
                results = hrv_processor.compute_hrv_metrics(
                    rr_intervals.values,
                    domains=[HRVDomain.TIME, HRVDomain.FREQUENCY]
                )
                all_results[subject] = results
                
        print(f"✅ Processed {len(all_results)} subjects")
        
        # Demonstrate clustering
        print("🎯 Performing clustering analysis...")
        
        # Prepare data for clustering
        clustering_data = []
        for subject, results in all_results.items():
            if 'time_domain' in results and 'frequency_domain' in results:
                row = {
                    'subject': subject,
                    'sdnn': results['time_domain'].get('sdnn', 0),
                    'rmssd': results['time_domain'].get('rmssd', 0),
                    'lf_power': results['frequency_domain'].get('lf_power', 0),
                    'hf_power': results['frequency_domain'].get('hf_power', 0),
                    'lf_hf_ratio': results['frequency_domain'].get('lf_hf_ratio', 0)
                }
                clustering_data.append(row)
                
        if len(clustering_data) >= 3:
            import pandas as pd
            cluster_df = pd.DataFrame(clustering_data).set_index('subject')
            
            clustering = HRVClustering()
            cluster_result = clustering.perform_kmeans_clustering(cluster_df)
            interpretation = clustering.interpret_clusters(cluster_result)
            
            print(f"✅ Identified {cluster_result.n_clusters} autonomic phenotypes")
            print(f"📈 Silhouette score: {cluster_result.silhouette_score:.3f}")
            
            # Show phenotype distribution
            if 'overall_analysis' in interpretation:
                phenotypes = interpretation['overall_analysis'].get('phenotype_distribution', {})
                print("🏷️ Phenotype distribution:")
                for phenotype, count in phenotypes.items():
                    print(f"  {phenotype}: {count} subjects")
                    
        # Generate sample plot
        print("📈 Creating visualization...")
        if all_results:
            first_subject = list(all_results.keys())[0]
            first_result = all_results[first_subject]
            
            # Create sample RR intervals for plotting
            import numpy as np
            np.random.seed(42)
            sample_rr = np.random.normal(800, 50, 200)
            
            plotter = InteractivePlotter()
            fig = plotter.create_poincare_plot(sample_rr, title=f"Demo: Poincaré Plot")
            
            # Export to HTML
            output_file = current_dir / "demo_poincare_plot.html"
            plotter.export_html(fig, str(output_file))
            print(f"📊 Demo plot saved to: {output_file}")
            
        print("✅ Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        return False

def show_version():
    """Show version information."""
    try:
        from enhanced_hrv_analysis import __version__, __author__
        print(f"Enhanced HRV Analysis System v{__version__}")
        print(f"Author: {__author__}")
    except ImportError:
        print("Enhanced HRV Analysis System v2.0.0")
        print("Author: Enhanced HRV Analysis Team")
    
    print("\nSystem Information:")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Show dependency status
    missing_required, missing_optional = check_dependencies()
    
    print(f"\nDependencies:")
    print(f"✅ Required packages installed: {len(missing_required) == 0}")
    print(f"⚠️ Optional packages missing: {len(missing_optional)}")
    
    if missing_required:
        print(f"❌ Missing required: {missing_required}")
    if missing_optional:
        print(f"📦 Missing optional: {missing_optional}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced HRV Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--gui', action='store_true', default=True,
                       help='Launch GUI application (default)')
    parser.add_argument('--cli', action='store_true',
                       help='Run in command-line mode')  
    parser.add_argument('--test', action='store_true',
                       help='Run test suite')
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration')
    parser.add_argument('--data', type=str, metavar='PATH',
                       help='Path to data file or directory')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install required dependencies')
    parser.add_argument('--version', action='store_true',
                       help='Show version information')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        setup_logging(logging.DEBUG)
    else:
        setup_logging(logging.INFO)
        
    # Handle version request
    if args.version:
        show_version()
        return True
        
    # Handle dependency installation
    if args.install_deps:
        return install_dependencies()
        
    # Handle test request
    if args.test:
        return run_tests()
        
    # Handle demo request
    if args.demo:
        return run_demo()
        
    # Handle CLI request
    if args.cli:
        return run_command_line(args.data)
        
    # Default: launch GUI
    return launch_gui()

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1) 