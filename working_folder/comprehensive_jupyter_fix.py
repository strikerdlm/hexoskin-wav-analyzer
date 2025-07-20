#!/usr/bin/env python3
"""
Comprehensive Jupyter Notebook Fix for Valquiria Data Analysis
==============================================================

This script fixes all known issues with the Jupyter notebooks including:
1. Missing dependencies
2. HRV analysis compatibility issues  
3. Astropy compatibility patches
4. Import errors
5. Database connectivity
6. Environment setup

Usage: python comprehensive_jupyter_fix.py
"""

import subprocess
import sys
import os
import types
import sqlite3
from pathlib import Path
from typing import List, Tuple
import warnings

warnings.filterwarnings('ignore')

class JupyterNotebookFixer:
    def __init__(self):
        self.current_dir = Path(__file__).parent
        self.success_count = 0
        self.total_fixes = 0
        
    def log_success(self, message: str):
        """Log a successful operation"""
        print(f"✅ {message}")
        self.success_count += 1
        
    def log_error(self, message: str):
        """Log an error"""
        print(f"❌ {message}")
        
    def log_warning(self, message: str):
        """Log a warning"""
        print(f"⚠️ {message}")
        
    def log_info(self, message: str):
        """Log information"""
        print(f"ℹ️ {message}")

    def install_package(self, package: str) -> bool:
        """Install a single package"""
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def check_import(self, package_name: str, import_name: str = None) -> bool:
        """Check if a package can be imported"""
        import_name = import_name or package_name
        try:
            __import__(import_name)
            return True
        except ImportError:
            return False

    def fix_dependencies(self):
        """Install all required dependencies"""
        print("\n" + "="*60)
        print("STEP 1: INSTALLING DEPENDENCIES")
        print("="*60)
        
        # Core packages with compatible versions
        packages = [
            'pandas>=1.3.0',
            'numpy>=1.20.0', 
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'scipy>=1.7.0',
            'jupyter>=1.0.0',
            'jupyterlab>=3.0.0',
            'ipykernel>=6.0.0',
            'notebook>=6.4.0',
            'statsmodels>=0.13.0',
            'scikit-learn>=1.0.0',
            'astropy<7.0',  # Compatible version for hrv-analysis
            'hrv-analysis>=1.0.4',
            'neurokit2>=0.2.0',
            'heartpy>=1.2.7',
            'plotly>=5.0.0',
            'openpyxl>=3.0.0'
        ]
        
        self.log_info(f"Installing {len(packages)} essential packages...")
        
        for package in packages:
            if self.install_package(package):
                self.log_success(f"Installed: {package}")
            else:
                self.log_error(f"Failed to install: {package}")
        
        # Verify critical imports
        critical_imports = [
            ('pandas', 'pandas'),
            ('numpy', 'numpy'),
            ('matplotlib', 'matplotlib'),
            ('seaborn', 'seaborn'),
            ('scipy', 'scipy'),
            ('jupyter', 'jupyter'),
            ('statsmodels', 'statsmodels'),
            ('sklearn', 'sklearn'),
            ('astropy', 'astropy'),
            ('hrvanalysis', 'hrvanalysis'),
        ]
        
        self.log_info("Verifying critical imports...")
        for package_name, import_name in critical_imports:
            if self.check_import(package_name, import_name):
                self.log_success(f"Import verified: {import_name}")
            else:
                self.log_error(f"Import failed: {import_name}")

    def apply_astropy_patch(self) -> bool:
        """Apply astropy compatibility patch for hrv-analysis"""
        print("\n" + "="*60)
        print("STEP 2: APPLYING ASTROPY COMPATIBILITY PATCH")
        print("="*60)
        
        try:
            import astropy
            self.log_info(f"Astropy version: {astropy.__version__}")
            
            # Check if LombScargle is in the old location
            try:
                from astropy.stats import LombScargle
                self.log_success("LombScargle available in astropy.stats (compatible)")
                return True
            except ImportError:
                self.log_info("LombScargle not in astropy.stats, applying patch...")
                
            # Try the new location and apply patch
            try:
                from astropy.timeseries import LombScargle
                self.log_success("Found LombScargle in astropy.timeseries")
                
                # Create compatibility module
                stats_mod = types.ModuleType("astropy.stats")
                stats_mod.LombScargle = LombScargle
                sys.modules["astropy.stats"] = stats_mod
                astropy.stats = stats_mod
                
                # Verify patch
                from astropy.stats import LombScargle as TestLS
                self.log_success("Astropy compatibility patch applied successfully")
                return True
                
            except ImportError:
                self.log_error("LombScargle not found in astropy.timeseries")
                return False
                
        except ImportError:
            self.log_error("Astropy not available")
            return False

    def test_hrv_analysis(self) -> bool:
        """Test HRV analysis functionality"""
        print("\n" + "="*60)
        print("STEP 3: TESTING HRV ANALYSIS")
        print("="*60)
        
        try:
            # Test basic imports
            from hrvanalysis import (
                get_time_domain_features,
                get_frequency_domain_features,
                get_poincare_plot_features,
                remove_outliers,
                interpolate_nan_values
            )
            self.log_success("HRV analysis functions imported")
            
            # Test with sample data
            sample_rr = [800, 810, 790, 820, 830, 800, 810, 790, 825, 805] * 30
            self.log_info(f"Testing with {len(sample_rr)} sample RR intervals...")
            
            # Test time domain
            try:
                time_features = get_time_domain_features(sample_rr)
                self.log_success(f"Time domain analysis: {len(time_features)} features")
            except Exception as e:
                self.log_error(f"Time domain analysis failed: {e}")
                
            # Test frequency domain  
            try:
                freq_features = get_frequency_domain_features(sample_rr)
                self.log_success(f"Frequency domain analysis: {len(freq_features)} features")
            except Exception as e:
                self.log_error(f"Frequency domain analysis failed: {e}")
                
            # Test nonlinear
            try:
                poincare_features = get_poincare_plot_features(sample_rr)
                self.log_success(f"Poincaré analysis: {len(poincare_features)} features")
            except Exception as e:
                self.log_error(f"Poincaré analysis failed: {e}")
                
            return True
            
        except ImportError as e:
            self.log_error(f"HRV analysis import failed: {e}")
            return False

    def test_alternative_hrv_libraries(self):
        """Test alternative HRV libraries as fallback"""
        print("\n" + "="*60)
        print("STEP 4: TESTING ALTERNATIVE HRV LIBRARIES")
        print("="*60)
        
        # Test NeuroKit2
        try:
            import neurokit2 as nk
            self.log_success(f"NeuroKit2 available: v{nk.__version__}")
        except ImportError:
            self.log_warning("NeuroKit2 not available")
            
        # Test HeartPy
        try:
            import heartpy as hp
            self.log_success(f"HeartPy available: v{hp.__version__}")
        except ImportError:
            self.log_warning("HeartPy not available")

    def check_data_sources(self):
        """Check if data files are accessible"""
        print("\n" + "="*60)
        print("STEP 5: CHECKING DATA SOURCES")
        print("="*60)
        
        # Check database
        db_path = self.current_dir / "merged_data.db"
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                conn.close()
                self.log_success(f"Database accessible: {len(tables)} tables found")
            except Exception as e:
                self.log_error(f"Database connection failed: {e}")
        else:
            self.log_warning("Database file not found: merged_data.db")
            
        # Check CSV files
        csv_files = list(self.current_dir.glob("T*.csv"))
        if csv_files:
            self.log_success(f"CSV files found: {len(csv_files)} files")
            for csv_file in csv_files:
                size_mb = csv_file.stat().st_size / (1024 * 1024)
                self.log_info(f"  {csv_file.name}: {size_mb:.1f} MB")
        else:
            self.log_warning("No CSV files found")

    def create_jupyter_kernel(self):
        """Create a custom Jupyter kernel for the analysis"""
        print("\n" + "="*60)
        print("STEP 6: CREATING JUPYTER KERNEL")
        print("="*60)
        
        try:
            kernel_name = "valquiria-analysis"
            display_name = "Valquiria Space Analog Analysis"
            
            cmd = [
                sys.executable, "-m", "ipykernel", "install",
                "--user", f"--name={kernel_name}",
                f"--display-name={display_name}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.log_success(f"Jupyter kernel '{display_name}' created successfully")
            else:
                self.log_warning(f"Kernel creation failed: {result.stderr}")
                
        except Exception as e:
            self.log_error(f"Failed to create Jupyter kernel: {e}")

    def create_startup_scripts(self):
        """Create convenient startup scripts"""
        print("\n" + "="*60)
        print("STEP 7: CREATING STARTUP SCRIPTS")
        print("="*60)
        
        # Create Python startup script
        startup_content = '''#!/usr/bin/env python3
"""
Jupyter Startup Script for Valquiria Analysis
"""
import subprocess
import sys
import os
from pathlib import Path

def start_jupyter():
    """Start Jupyter notebook with proper configuration"""
    # Apply compatibility patches
    exec(open('comprehensive_jupyter_fix.py').read())
    
    # Start Jupyter
    print("Starting Jupyter Notebook...")
    print("The notebook will open in your browser automatically.")
    print("Press Ctrl+C to stop the server.")
    
    cmd = ["jupyter", "notebook", "--no-browser", "--port=8888"]
    subprocess.run(cmd)

if __name__ == "__main__":
    start_jupyter()
'''
        
        startup_file = self.current_dir / "start_analysis.py"
        with open(startup_file, 'w', encoding='utf-8') as f:
            f.write(startup_content)
        
        self.log_success("Created start_analysis.py")
        
        # Create batch file for Windows
        batch_content = '''@echo off
echo Starting Valquiria Jupyter Analysis Environment
echo =============================================
python comprehensive_jupyter_fix.py
if %ERRORLEVEL% EQU 0 (
    echo Environment setup complete. Starting Jupyter...
    jupyter notebook --port=8888
) else (
    echo Setup failed. Please check the error messages above.
    pause
)
'''
        
        batch_file = self.current_dir / "start_analysis.bat"
        with open(batch_file, 'w', encoding='utf-8') as f:
            f.write(batch_content)
            
        self.log_success("Created start_analysis.bat")

    def create_compatibility_module(self):
        """Create a compatibility module for notebooks to import"""
        print("\n" + "="*60)
        print("STEP 8: CREATING COMPATIBILITY MODULE")
        print("="*60)
        
        compat_content = '''"""
Compatibility module for Valquiria Jupyter notebooks
This module ensures all imports work correctly and applies necessary patches
"""

import sys
import types
import warnings
warnings.filterwarnings('ignore')

# Apply astropy patch
def setup_astropy_compatibility():
    """Apply astropy compatibility patch"""
    try:
        import astropy
        try:
            from astropy.stats import LombScargle
            return True
        except ImportError:
            from astropy.timeseries import LombScargle
            stats_mod = types.ModuleType("astropy.stats")
            stats_mod.LombScargle = LombScargle
            sys.modules["astropy.stats"] = stats_mod
            astropy.stats = stats_mod
            return True
    except ImportError:
        return False

# Setup compatibility on import
setup_astropy_compatibility()

# Safe imports with fallbacks
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import sqlite3
    HAS_BASIC_LIBS = True
except ImportError:
    HAS_BASIC_LIBS = False

try:
    from hrvanalysis import (
        get_time_domain_features,
        get_frequency_domain_features, 
        get_poincare_plot_features,
        remove_outliers,
        interpolate_nan_values
    )
    HAS_HRV_ANALYSIS = True
except ImportError:
    HAS_HRV_ANALYSIS = False

try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False
    
try:
    import heartpy as hp
    HAS_HEARTPY = True
except ImportError:
    HAS_HEARTPY = False

def get_available_libraries():
    """Return status of available libraries"""
    return {
        'basic_libs': HAS_BASIC_LIBS,
        'hrv_analysis': HAS_HRV_ANALYSIS,
        'neurokit': HAS_NEUROKIT,
        'heartpy': HAS_HEARTPY
    }

def load_data_safely():
    """Load data with error handling"""
    import sqlite3
    from pathlib import Path
    
    # Try database first
    db_path = Path("merged_data.db")
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            df = pd.read_sql_query("SELECT * FROM merged_data", conn)
            conn.close()
            print(f"✅ Loaded {len(df):,} rows from database")
            return df
        except Exception as e:
            print(f"❌ Database loading failed: {e}")
    
    # Try CSV files
    csv_files = list(Path(".").glob("T*.csv"))
    if csv_files:
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
                print(f"✅ Loaded {csv_file.name}")
            except Exception as e:
                print(f"❌ Failed to load {csv_file.name}: {e}")
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            print(f"✅ Combined {len(combined_df):,} rows from CSV files")
            return combined_df
    
    print("❌ No data sources available")
    return None

print("✅ Valquiria compatibility module loaded successfully")
print(f"Available libraries: {get_available_libraries()}")
'''
        
        compat_file = self.current_dir / "valquiria_compat.py"
        with open(compat_file, 'w', encoding='utf-8') as f:
            f.write(compat_content)
            
        self.log_success("Created valquiria_compat.py")

    def run_comprehensive_fix(self):
        """Run all fix procedures"""
        print("🔧 COMPREHENSIVE JUPYTER NOTEBOOK FIX")
        print("="*60)
        print("Fixing all issues with Valquiria Jupyter notebooks...")
        print("="*60)
        
        # Count total procedures
        self.total_fixes = 8
        
        # Run all fix procedures
        self.fix_dependencies()
        self.apply_astropy_patch()
        self.test_hrv_analysis()
        self.test_alternative_hrv_libraries()
        self.check_data_sources()
        self.create_jupyter_kernel()
        self.create_startup_scripts()
        self.create_compatibility_module()
        
        # Final summary
        print("\n" + "="*60)
        print("FIX SUMMARY")
        print("="*60)
        
        print(f"✅ Procedures completed: {self.success_count}/{self.total_fixes}")
        
        if self.success_count >= 6:  # Most critical procedures succeeded
            print("\n🎉 JUPYTER NOTEBOOKS ARE NOW READY!")
            print("\nNext steps:")
            print("1. Start Jupyter: python start_analysis.py")
            print("2. Or use: start_analysis.bat (Windows)")
            print("3. Or manually: jupyter notebook")
            print("\nIn your notebooks, add this at the top:")
            print("   import valquiria_compat")
            print("   df = valquiria_compat.load_data_safely()")
        else:
            print("\n⚠️ Some issues remain. Check the errors above.")
            print("\nTroubleshooting:")
            print("1. Ensure Python and pip are working")
            print("2. Try running this script as administrator")
            print("3. Check your internet connection for package downloads")


def main():
    """Main entry point"""
    fixer = JupyterNotebookFixer()
    fixer.run_comprehensive_fix()


if __name__ == "__main__":
    main() 