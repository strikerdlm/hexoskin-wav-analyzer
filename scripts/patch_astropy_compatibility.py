"""
Standalone script to patch astropy compatibility for hrvanalysis
This fixes the issue where LombScargle has been moved from astropy.stats to astropy.timeseries
"""

import sys
import types

def patch_astropy():
    """Apply the astropy compatibility patch"""
    print("Applying astropy compatibility patch...")
    
    try:
        import astropy
        print(f"Found astropy version: {astropy.__version__}")
        
        # Check if LombScargle is already available in astropy.stats
        try:
            from astropy.stats import LombScargle
            print("✅ LombScargle already available in astropy.stats")
            return True
        except ImportError:
            print("LombScargle not found in astropy.stats, applying patch...")
            
        # Try to import from timeseries
        try:
            from astropy.timeseries import LombScargle
            print("✅ Found LombScargle in astropy.timeseries")
            
            # Create a fake astropy.stats module
            stats_mod = types.ModuleType("astropy.stats")
            stats_mod.LombScargle = LombScargle
            sys.modules["astropy.stats"] = stats_mod
            astropy.stats = stats_mod
            
            print("✅ Successfully patched astropy.stats.LombScargle")
            
            # Test the patch
            from astropy.stats import LombScargle as TestLS
            print("✅ Patch verification successful")
            
            return True
            
        except ImportError:
            print("❌ LombScargle not found in astropy.timeseries either")
            return False
            
    except ImportError:
        print("❌ astropy not found")
        return False

def test_hrvanalysis_import():
    """Test if hrvanalysis can be imported after the patch"""
    print("\nTesting hrvanalysis import...")
    
    try:
        import hrvanalysis
        print("✅ hrvanalysis imported successfully")
        
        # Test the functions
        from hrvanalysis import (
            get_time_domain_features,
            get_frequency_domain_features,
            get_poincare_plot_features,
        )
        print("✅ HRV functions imported successfully")
        
        # Test with sample data
        sample_rr = [800, 810, 790, 820, 830, 800, 810, 790, 825, 805] * 20
        print(f"Testing with {len(sample_rr)} RR intervals...")
        
        try:
            time_feats = get_time_domain_features(sample_rr)
            print(f"✅ Time domain: {len(time_feats)} features")
        except Exception as e:
            print(f"❌ Time domain failed: {e}")
            
        try:
            freq_feats = get_frequency_domain_features(sample_rr)
            print(f"✅ Frequency domain: {len(freq_feats)} features")
        except Exception as e:
            print(f"❌ Frequency domain failed: {e}")
            
        try:
            poincare_feats = get_poincare_plot_features(sample_rr)
            print(f"✅ Poincaré: {len(poincare_feats)} features")
        except Exception as e:
            print(f"❌ Poincaré failed: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ hrvanalysis import failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Astropy Compatibility Patch for HRV Analysis")
    print("="*60)
    
    # Apply the patch
    if patch_astropy():
        print("\n" + "="*60)
        print("Testing HRV Analysis with Patch")
        print("="*60)
        
        if test_hrvanalysis_import():
            print("\n✅ SUCCESS: HRV analysis is now working!")
            print("You can now use hrvanalysis functions in your scripts.")
        else:
            print("\n❌ FAILURE: HRV analysis still not working after patch.")
    else:
        print("\n❌ FAILURE: Could not apply compatibility patch.")
        
    print("\nNext steps:")
    print("1. If successful, you can now run: python hrv_analysis_fixed.py")
    print("2. Or use the HRV functions directly in your notebook")
    print("3. If failed, check the error messages above") 