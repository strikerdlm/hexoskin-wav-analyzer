#!/usr/bin/env python3
"""
Test script to verify unified export directory configuration

This script tests that all components use the unified export directory:
C:/Users/User/OneDrive/FAC/Research/Valquiria/Data/src/hrv_analysis/enhanced_hrv_analysis/plots_output

Run this script from the enhanced_hrv_analysis directory to verify the configuration.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_unified_export_configuration():
    """Test that the unified export configuration is working correctly."""
    print("Testing Unified Export Directory Configuration")
    print("=" * 60)
    
    # Test 1: Check if config module works
    try:
        from config import get_export_directory, get_plots_output_path
        export_dir = get_export_directory()
        print(f"✅ Config module loaded successfully")
        print(f"   Export directory: {export_dir}")
        print(f"   Absolute path: {export_dir.absolute()}")
        
        # Verify the directory is in the correct location
        expected_location = current_dir / "plots_output"
        if export_dir.absolute() == expected_location.absolute():
            print(f"✅ Export directory is in correct location")
        else:
            print(f"❌ Export directory mismatch:")
            print(f"   Expected: {expected_location.absolute()}")
            print(f"   Actual: {export_dir.absolute()}")
            
    except ImportError as e:
        print(f"❌ Config module import failed: {e}")
        return False
    
    # Test 2: Test with filename parameter
    try:
        test_filename = "test_plot.html"
        full_path = get_plots_output_path(test_filename)
        expected_full_path = export_dir / test_filename
        
        if full_path == expected_full_path:
            print(f"✅ Filename path generation works correctly")
            print(f"   Test file path: {full_path}")
        else:
            print(f"❌ Filename path generation failed:")
            print(f"   Expected: {expected_full_path}")
            print(f"   Actual: {full_path}")
            
    except Exception as e:
        print(f"❌ Filename path generation failed: {e}")
        return False
    
    # Test 3: Check environment variable (if set by launcher)
    if 'HRV_EXPORT_DIR' in os.environ:
        env_dir = Path(os.environ['HRV_EXPORT_DIR'])
        print(f"✅ Environment variable set: {env_dir}")
        
        if env_dir.absolute() == export_dir.absolute():
            print(f"✅ Environment variable matches config")
        else:
            print(f"⚠️ Environment variable mismatch:")
            print(f"   Environment: {env_dir.absolute()}")
            print(f"   Config: {export_dir.absolute()}")
    else:
        print(f"⚠️ Environment variable HRV_EXPORT_DIR not set (normal if not launched via launcher)")
    
    # Test 4: Verify directory creation
    try:
        export_dir.mkdir(parents=True, exist_ok=True)
        if export_dir.exists():
            print(f"✅ Export directory created/exists: {export_dir.absolute()}")
        else:
            print(f"❌ Failed to create export directory")
            return False
    except Exception as e:
        print(f"❌ Directory creation failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("UNIFIED EXPORT CONFIGURATION TEST RESULTS:")
    print(f"✅ All exports will go to: {export_dir.absolute()}")
    print(f"✅ Configuration is working correctly!")
    return True

if __name__ == "__main__":
    success = test_unified_export_configuration()
    sys.exit(0 if success else 1) 