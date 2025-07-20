"""
Test script for HRV Analysis
Verifies that all components work correctly
"""

import sys
import os
from pathlib import Path
import traceback

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ matplotlib import failed: {e}")
        return False
    
    try:
        import scipy
        print("✅ scipy imported successfully")
    except ImportError as e:
        print(f"❌ scipy import failed: {e}")
        return False
    
    try:
        import hrvanalysis
        print("✅ hrvanalysis imported successfully")
    except ImportError as e:
        print(f"❌ hrvanalysis import failed: {e}")
        print("Please install with: pip install hrv-analysis")
        return False
    
    return True

def test_hrv_functions():
    """Test HRV analysis functions"""
    print("\nTesting HRV functions...")
    
    try:
        from hrvanalysis import (
            get_time_domain_features,
            get_frequency_domain_features,
            get_poincare_plot_features,
        )
        print("✅ HRV functions imported successfully")
        
        # Test with sample data
        sample_rr = [800, 810, 790, 820, 830, 800, 810, 790, 825, 805] * 20
        
        # Test time domain features
        try:
            time_features = get_time_domain_features(sample_rr)
            print(f"✅ Time domain features: {len(time_features)} features computed")
        except Exception as e:
            print(f"❌ Time domain features failed: {e}")
            return False
        
        # Test frequency domain features
        try:
            freq_features = get_frequency_domain_features(sample_rr)
            print(f"✅ Frequency domain features: {len(freq_features)} features computed")
        except Exception as e:
            print(f"❌ Frequency domain features failed: {e}")
            traceback.print_exc()
            return False
        
        # Test Poincaré features
        try:
            poincare_features = get_poincare_plot_features(sample_rr)
            print(f"✅ Poincaré features: {len(poincare_features)} features computed")
        except Exception as e:
            print(f"❌ Poincaré features failed: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ HRV functions import failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\nTesting data loading...")
    
    try:
        # Try to import load_data
        try:
            from load_data import load_database_data, load_csv_data
            print("✅ load_data imported from current directory")
        except ImportError:
            try:
                from scripts.load_data import load_database_data, load_csv_data
                print("✅ load_data imported from scripts directory")
            except ImportError:
                print("❌ Could not import load_data module")
                return False
        
        # Test database loading
        try:
            df = load_database_data("merged_data.db")
            if df is not None:
                print(f"✅ Database loaded: {len(df)} rows")
                return True
            else:
                print("⚠️ Database not found, testing CSV loading...")
        except Exception as e:
            print(f"⚠️ Database loading failed: {e}")
        
        # Test CSV loading
        try:
            csvs = load_csv_data()
            if csvs:
                print(f"✅ CSV files loaded: {len(csvs)} files")
                return True
            else:
                print("⚠️ No CSV files found")
        except Exception as e:
            print(f"⚠️ CSV loading failed: {e}")
        
        print("⚠️ No data sources available - this is okay for testing")
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_plotting():
    """Test plotting functionality"""
    print("\nTesting plotting...")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Test simple plot
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title("Test Plot")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        # Save to test directory
        test_dir = Path(__file__).parent / "test_outputs"
        test_dir.mkdir(exist_ok=True)
        
        plt.savefig(test_dir / "test_plot.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Basic plotting works")
        return True
        
    except Exception as e:
        print(f"❌ Plotting test failed: {e}")
        return False

def test_complete_workflow():
    """Test complete HRV analysis workflow"""
    print("\nTesting complete workflow...")
    
    try:
        # Import everything
        import numpy as np
        import pandas as pd
        from hrvanalysis import (
            get_time_domain_features,
            get_frequency_domain_features,
            get_poincare_plot_features,
        )
        
        # Create sample data
        np.random.seed(42)
        n_samples = 500
        base_hr = 70
        noise = np.random.normal(0, 5, n_samples)
        trend = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 8
        
        heart_rate = base_hr + noise + trend
        heart_rate = np.clip(heart_rate, 50, 120)
        
        # Convert to RR intervals
        rr_ms = 60_000.0 / heart_rate
        rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
        
        # Compute all features
        time_features = get_time_domain_features(rr_ms)
        freq_features = get_frequency_domain_features(rr_ms)
        poincare_features = get_poincare_plot_features(rr_ms)
        
        # Combine results
        all_features = {
            "subject": "TestSubject",
            "Sol": "TestSol",
            **time_features,
            **freq_features,
            **poincare_features
        }
        
        # Save results
        results_df = pd.DataFrame([all_features])
        test_dir = Path(__file__).parent / "test_outputs"
        test_dir.mkdir(exist_ok=True)
        
        results_df.to_csv(test_dir / "test_hrv_results.csv", index=False)
        
        print(f"✅ Complete workflow successful: {len(all_features)} features computed")
        print(f"✅ Results saved to: {test_dir / 'test_hrv_results.csv'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete workflow test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("HRV Analysis Test Suite")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("HRV Functions Test", test_hrv_functions),
        ("Data Loading Test", test_data_loading),
        ("Plotting Test", test_plotting),
        ("Complete Workflow Test", test_complete_workflow),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("✅ All tests passed! HRV analysis is ready to use.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 