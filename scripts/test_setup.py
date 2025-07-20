#!/usr/bin/env python3
"""
Test script for Jupyter setup
Verifies that all components are working correctly.
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    required_packages = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'scipy',
        'jupyter',
        'ipykernel',
        'statsmodels',
        'sklearn',  # scikit-learn
        'scikit_posthocs',  # scikit-posthocs
        'statannotations',
        'missingno',
        'tabulate'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed to import: {failed_imports}")
        print("Run: pip install -r requirements_jupyter.txt")
        return False
    else:
        print("\n✓ All packages imported successfully")
        return True

def test_data_loading():
    """Test data loading utilities."""
    print("\nTesting data loading utilities...")
    
    try:
        from scripts.load_data import load_csv_data, load_database_data, get_data_summary
        
        # Test CSV loading
        print("Testing CSV loading...")
        csv_data = load_csv_data()
        print(f"✓ Loaded {len(csv_data)} CSV files")
        
        # Test database loading
        print("Testing database loading...")
        db_data = load_database_data()
        
        if db_data is not None:
            summary = get_data_summary(db_data)
            print(f"✓ Database loaded: {summary['rows']} rows, {summary['columns']} columns")
            return True
        else:
            print("⚠ Database not found (this is OK if you haven't created it yet)")
            return True
            
    except Exception as e:
        print(f"✗ Error testing data loading: {e}")
        return False

def test_analysis_utils():
    """Test analysis utilities."""
    print("\nTesting analysis utilities...")
    
    try:
        from scripts.analysis_utils import (
            setup_plotting_style,
            analyze_variable_distribution,
            correlation_analysis,
            statistical_comparison
        )
        
        print("✓ Analysis utilities imported successfully")
        
        # Test with sample data
        import pandas as pd
        import numpy as np
        
        # Create sample data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'subject': ['A', 'A', 'B', 'B', 'C', 'C'] * 10,
            'heart_rate': np.random.normal(80, 10, 60),
            'activity': np.random.exponential(5, 60)
        })
        
        print("✓ Sample data created for testing")
        
        # Test plotting style
        setup_plotting_style()
        print("✓ Plotting style configured")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing analysis utilities: {e}")
        return False

def test_jupyter_kernel():
    """Test Jupyter kernel installation."""
    print("\nTesting Jupyter kernel...")
    
    try:
        import subprocess
        result = subprocess.run(['jupyter', 'kernelspec', 'list'], 
                              capture_output=True, text=True)
        
        if 'valquiria-analysis' in result.stdout:
            print("✓ Valquiria analysis kernel found")
            return True
        else:
            print("⚠ Valquiria analysis kernel not found")
            print("Run: python setup_jupyter_environment.py")
            return False
            
    except Exception as e:
        print(f"✗ Error testing Jupyter kernel: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("JUPYTER SETUP TEST")
    print("="*60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Analysis Utilities", test_analysis_utils),
        ("Jupyter Kernel", test_jupyter_kernel)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your Jupyter setup is ready.")
        print("\nNext steps:")
        print("1. Start Jupyter: jupyter notebook")
        print("2. Open Results.ipynb or Results_2.ipynb")
        print("3. Select the 'Valquiria Space Analog Analysis' kernel")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r requirements_jupyter.txt")
        print("2. Run setup: python setup_jupyter_environment.py")
        print("3. Check Python environment and paths")

if __name__ == "__main__":
    main() 