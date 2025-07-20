"""
Setup script for HRV Analysis in Valquiria project
Installs all required packages and verifies the installation
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

def upgrade_package(package_name):
    """Upgrade a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
        print(f"✅ Successfully upgraded {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to upgrade {package_name}: {e}")
        return False

def check_package_import(package_name, import_name=None):
    """Check if a package can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} can be imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {package_name}: {e}")
        return False

def install_requirements_from_file(requirements_file):
    """Install packages from requirements file"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print(f"✅ Successfully installed packages from {requirements_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install from {requirements_file}: {e}")
        return False

def main():
    print("="*60)
    print("HRV Analysis Setup for Valquiria Project")
    print("="*60)
    
    # Get current directory
    current_dir = Path(__file__).parent
    requirements_file = current_dir / "requirements_jupyter.txt"
    
    print(f"Installing packages from: {requirements_file}")
    
    # Install from requirements file
    if requirements_file.exists():
        print("\n1. Installing packages from requirements file...")
        install_requirements_from_file(str(requirements_file))
    else:
        print(f"⚠️ Requirements file not found: {requirements_file}")
        
        # Install essential packages manually
        print("\n1. Installing essential packages manually...")
        essential_packages = [
            "numpy>=1.20.0",
            "pandas>=1.3.0", 
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "scipy>=1.7.0",
            "hrv-analysis>=1.0.4"
        ]
        
        for package in essential_packages:
            install_package(package)
    
    # Verify critical imports
    print("\n2. Verifying package imports...")
    critical_imports = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scipy", "scipy"),
        ("hrv-analysis", "hrvanalysis"),
        ("sqlite3", "sqlite3")
    ]
    
    failed_imports = []
    for package_name, import_name in critical_imports:
        if not check_package_import(package_name, import_name):
            failed_imports.append(package_name)
    
    # Try to fix specific issues
    print("\n3. Checking for specific compatibility issues...")
    
    # Check astropy compatibility for hrvanalysis
    try:
        import astropy
        print(f"✅ astropy version: {astropy.__version__}")
        
        # Check if LombScargle is available
        try:
            from astropy.stats import LombScargle
            print("✅ astropy.stats.LombScargle available")
        except ImportError:
            try:
                from astropy.timeseries import LombScargle
                print("✅ astropy.timeseries.LombScargle available (newer version)")
            except ImportError:
                print("⚠️ LombScargle not found in astropy - this may cause issues")
                
    except ImportError:
        print("⚠️ astropy not found - installing...")
        install_package("astropy")
    
    # Test HRV analysis import specifically
    print("\n4. Testing HRV analysis functionality...")
    try:
        from hrvanalysis import (
            get_time_domain_features,
            get_frequency_domain_features,
            get_poincare_plot_features,
        )
        print("✅ Successfully imported HRV analysis functions")
        
        # Test with sample data
        sample_rr = [800, 810, 790, 820, 830, 800, 810, 790] * 10
        time_features = get_time_domain_features(sample_rr)
        print(f"✅ Time domain features computed: {len(time_features)} features")
        
        freq_features = get_frequency_domain_features(sample_rr) 
        print(f"✅ Frequency domain features computed: {len(freq_features)} features")
        
        poincare_features = get_poincare_plot_features(sample_rr)
        print(f"✅ Poincaré features computed: {len(poincare_features)} features")
        
    except Exception as e:
        print(f"❌ HRV analysis test failed: {e}")
        failed_imports.append("hrv-analysis")
    
    # Summary
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    
    if failed_imports:
        print(f"❌ Failed packages: {', '.join(failed_imports)}")
        print("❌ HRV analysis setup incomplete")
        return False
    else:
        print("✅ All packages installed successfully")
        print("✅ HRV analysis setup complete")
        return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 