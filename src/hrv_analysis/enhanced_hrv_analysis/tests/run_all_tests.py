#!/usr/bin/env python3
"""
Comprehensive test runner for Enhanced HRV Analysis System.

This script runs all tests across the enhanced HRV analysis system,
providing detailed reporting and coverage analysis.
"""

import sys
import os
import pytest
from pathlib import Path
import argparse
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_core_tests(verbose=True):
    """Run core functionality tests."""
    print("=" * 60)
    print("RUNNING CORE FUNCTIONALITY TESTS")
    print("=" * 60)
    
    test_file = Path(__file__).parent / "test_core_functionality.py"
    
    if not test_file.exists():
        print(f"Warning: {test_file} not found!")
        return False
        
    args = [str(test_file)]
    if verbose:
        args.append("-v")
    args.extend(["--tb=short", "-x"])  # Stop on first failure
    
    result = pytest.main(args)
    return result == 0

def run_ans_balance_tests(verbose=True):
    """Run ANS balance analysis tests."""
    print("\n" + "=" * 60)
    print("RUNNING ANS BALANCE ANALYSIS TESTS")
    print("=" * 60)
    
    test_file = Path(__file__).parent / "test_ans_balance_analysis.py"
    
    if not test_file.exists():
        print(f"Warning: {test_file} not found!")
        return False
        
    args = [str(test_file)]
    if verbose:
        args.append("-v")
    args.extend(["--tb=short", "-x"])
    
    result = pytest.main(args)
    return result == 0

def run_statistics_tests(verbose=True):
    """Run advanced statistics tests."""
    print("\n" + "=" * 60)
    print("RUNNING ADVANCED STATISTICS TESTS")
    print("=" * 60)
    
    test_file = Path(__file__).parent / "test_advanced_statistics.py"
    
    if not test_file.exists():
        print(f"Warning: {test_file} not found!")
        return False
        
    args = [str(test_file)]
    if verbose:
        args.append("-v")
    args.extend(["--tb=short", "-x"])
    
    result = pytest.main(args)
    return result == 0

def run_visualization_tests(verbose=True):
    """Run visualization tests."""
    print("\n" + "=" * 60)
    print("RUNNING VISUALIZATION TESTS")
    print("=" * 60)
    
    test_file = Path(__file__).parent / "test_visualization.py"
    
    if not test_file.exists():
        print(f"Warning: {test_file} not found!")
        return False
        
    args = [str(test_file)]
    if verbose:
        args.append("-v")
    args.extend(["--tb=short", "-x"])
    
    result = pytest.main(args)
    return result == 0

def run_ml_analysis_tests(verbose=True):
    """Run ML analysis tests."""
    print("\n" + "=" * 60)
    print("RUNNING ML ANALYSIS TESTS")
    print("=" * 60)
    
    test_file = Path(__file__).parent / "test_ml_analysis.py"
    
    if not test_file.exists():
        print(f"Warning: {test_file} not found!")
        return False
        
    args = [str(test_file)]
    if verbose:
        args.append("-v")
    args.extend(["--tb=short", "-x"])
    
    result = pytest.main(args)
    return result == 0

def check_dependencies():
    """Check if required testing dependencies are available."""
    print("Checking testing dependencies...")
    
    missing_deps = []
    
    try:
        import numpy
        print("✓ NumPy available")
    except ImportError:
        missing_deps.append("numpy")
        print("✗ NumPy not available")
    
    try:
        import pandas
        print("✓ Pandas available")
    except ImportError:
        missing_deps.append("pandas")
        print("✗ Pandas not available")
    
    try:
        import scipy
        print("✓ SciPy available")
    except ImportError:
        missing_deps.append("scipy")
        print("✗ SciPy not available")
    
    try:
        import matplotlib
        print("✓ Matplotlib available")
    except ImportError:
        missing_deps.append("matplotlib")
        print("✗ Matplotlib not available")
    
    try:
        import sklearn
        print("✓ Scikit-learn available")
    except ImportError:
        missing_deps.append("scikit-learn")
        print("✗ Scikit-learn not available")
    
    try:
        import plotly
        print("✓ Plotly available")
    except ImportError:
        missing_deps.append("plotly")
        print("✗ Plotly not available")
    
    try:
        import joblib
        print("✓ Joblib available")
    except ImportError:
        missing_deps.append("joblib")
        print("✗ Joblib not available")
    
    try:
        import numba
        print("✓ Numba available")
    except ImportError:
        missing_deps.append("numba")
        print("✗ Numba not available")
    
    try:
        import statsmodels
        print("✓ Statsmodels available")
    except ImportError:
        print("⚠ Statsmodels not available (some advanced stats features will be skipped)")
    
    try:
        import pingouin
        print("✓ Pingouin available")
    except ImportError:
        print("⚠ Pingouin not available (some power analysis features will be skipped)")
    
    if missing_deps:
        print(f"\n❌ Missing critical dependencies: {', '.join(missing_deps)}")
        print("Please install missing dependencies before running tests.")
        return False
    else:
        print("\n✅ All critical dependencies are available.")
        return True

def run_quick_test():
    """Run a quick test to verify basic functionality."""
    print("Running quick functionality test...")
    
    try:
        # Test basic imports
        from core.hrv_processor import HRVProcessor, HRVDomain
        from core.data_loader import DataLoader
        
        # Create test data
        data_loader = DataLoader()
        sample_data = data_loader.create_sample_data(n_subjects=2, n_sols=2, samples_per_session=50)
        
        # Test basic HRV computation
        processor = HRVProcessor()
        import numpy as np
        test_rr = np.random.normal(800, 50, 100)
        
        results = processor.compute_hrv_metrics(
            test_rr, 
            domains=[HRVDomain.TIME, HRVDomain.FREQUENCY],
            include_confidence_intervals=False
        )
        
        if 'time_domain' in results and 'frequency_domain' in results:
            print("✅ Quick test passed - basic functionality working")
            return True
        else:
            print("❌ Quick test failed - missing expected results")
            return False
            
    except Exception as e:
        print(f"❌ Quick test failed with error: {e}")
        return False

def generate_test_report(results, start_time, end_time):
    """Generate a comprehensive test report."""
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST REPORT")
    print("=" * 80)
    print(f"Test run completed at: {datetime.fromtimestamp(end_time)}")
    print(f"Total duration: {duration:.2f} seconds")
    print()
    
    print("Test Results Summary:")
    print("-" * 40)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    failed_tests = total_tests - passed_tests
    
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        icon = "✅" if passed else "❌"
        print(f"{icon} {test_name:<30} {status}")
    
    print()
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests > 0:
        print("\n⚠️  Some tests failed. Check the detailed output above for error information.")
        print("Consider running individual test modules for more detailed debugging.")
    else:
        print("\n🎉 All tests passed successfully!")
        
    print("\n" + "=" * 80)
    
    return failed_tests == 0

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive tests for Enhanced HRV Analysis System"
    )
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run only quick functionality test'
    )
    parser.add_argument(
        '--module',
        choices=['core', 'ans', 'stats', 'viz', 'ml'],
        help='Run tests for specific module only'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--no-deps-check',
        action='store_true',
        help='Skip dependency checking'
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("Enhanced HRV Analysis System - Comprehensive Test Suite")
    print("=" * 60)
    print(f"Starting tests at: {datetime.now()}")
    print()
    
    # Check dependencies unless skipped
    if not args.no_deps_check:
        if not check_dependencies():
            print("\n❌ Dependency check failed. Exiting.")
            return 1
        print()
    
    # Quick test mode
    if args.quick:
        print("Running in quick test mode...")
        success = run_quick_test()
        return 0 if success else 1
    
    # Module-specific testing
    if args.module:
        if args.module == 'core':
            success = run_core_tests(args.verbose)
        elif args.module == 'ans':
            success = run_ans_balance_tests(args.verbose)
        elif args.module == 'stats':
            success = run_statistics_tests(args.verbose)
        elif args.module == 'viz':
            success = run_visualization_tests(args.verbose)
        elif args.module == 'ml':
            success = run_ml_analysis_tests(args.verbose)
        
        return 0 if success else 1
    
    # Comprehensive testing
    print("Running comprehensive test suite...")
    print()
    
    results = {}
    
    # Run all test modules
    results['Core Functionality'] = run_core_tests(args.verbose)
    results['ANS Balance Analysis'] = run_ans_balance_tests(args.verbose)
    results['Advanced Statistics'] = run_statistics_tests(args.verbose)
    results['Visualization'] = run_visualization_tests(args.verbose)
    results['ML Analysis'] = run_ml_analysis_tests(args.verbose)
    
    end_time = time.time()
    
    # Generate final report
    all_passed = generate_test_report(results, start_time, end_time)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 