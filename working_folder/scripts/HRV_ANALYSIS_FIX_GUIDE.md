# HRV Analysis Fix Guide

## Problem Summary

The HRV analysis notebook was failing due to several issues:

1. **Missing Dependencies**: The `hrv-analysis` package was not included in the Jupyter requirements
2. **Import Issues**: Package name confusion (`hrv-analysis` vs `hrvanalysis`)
3. **Compatibility Issues**: Potential astropy compatibility problems
4. **Path Issues**: Import path problems in the notebook environment

## Solution Overview

I've created several tools to fix these issues:

### 1. Fixed Requirements (`requirements_jupyter.txt`)
- Added `hrv-analysis>=1.0.4` to the dependencies
- This ensures the HRV analysis package is properly installed

### 2. Setup Script (`setup_hrv_analysis.py`)
- Automatically installs all required dependencies
- Verifies installation and compatibility
- Provides detailed error reporting

### 3. Fixed HRV Analysis Script (`hrv_analysis_fixed.py`)
- Robust error handling
- Compatibility patches for astropy
- Better import management
- Fallback to sample data if real data unavailable

### 4. Test Suite (`test_hrv_analysis.py`)
- Comprehensive testing of all components
- Verifies imports, functions, and complete workflow
- Helps identify any remaining issues

## Step-by-Step Fix Instructions

### Step 1: Install Dependencies

Run the setup script to install all required packages:

```bash
cd Data/joined_data
python setup_hrv_analysis.py
```

This will:
- Install packages from `requirements_jupyter.txt`
- Verify all imports work correctly
- Test HRV analysis functionality
- Report any issues

### Step 2: Run Tests

Verify everything works with the test suite:

```bash
python test_hrv_analysis.py
```

This will run comprehensive tests and report the results.

### Step 3: Use the Fixed HRV Analysis

You can now use the fixed HRV analysis in several ways:

#### Option A: Run the Fixed Script
```bash
python hrv_analysis_fixed.py
```

#### Option B: Use in Your Notebook
```python
# In your notebook, use these imports:
from hrvanalysis import (
    get_time_domain_features,
    get_frequency_domain_features,
    get_poincare_plot_features,
)

# Load your data
from scripts.load_data import load_database_data, load_csv_data

# Use the fixed functions from hrv_analysis_fixed.py
```

## Key Fixes Applied

### 1. Dependency Management
- Added `hrv-analysis>=1.0.4` to requirements
- Ensured all scientific computing packages are properly versioned
- Added astropy for compatibility

### 2. Import Handling
```python
# Fixed import pattern:
try:
    from hrvanalysis import (
        get_time_domain_features,
        get_frequency_domain_features,
        get_poincare_plot_features,
    )
except ImportError as e:
    print(f"HRV analysis not available: {e}")
    # Handle gracefully
```

### 3. Astropy Compatibility
```python
# Compatibility patch for astropy
try:
    from astropy.stats import LombScargle
except ImportError:
    from astropy.timeseries import LombScargle
    # Apply compatibility patch
```

### 4. Robust Error Handling
- Comprehensive try-catch blocks
- Informative error messages
- Graceful degradation when components fail
- Fallback to sample data for testing

## Understanding the HRV Analysis

The fixed script computes three types of HRV features:

### Time Domain Features
- `mean_nni`: Mean of RR intervals
- `sdnn`: Standard deviation of RR intervals  
- `rmssd`: Root mean square of successive differences
- `nn50`: Number of successive RR intervals differing by > 50ms
- `pnn50`: Percentage of successive RR intervals differing by > 50ms

### Frequency Domain Features
- `lf`: Low frequency power (0.04-0.15 Hz)
- `hf`: High frequency power (0.15-0.4 Hz)
- `vlf`: Very low frequency power (0.003-0.04 Hz)
- `lf_hf_ratio`: Ratio of LF to HF power
- `total_power`: Total power in all frequency bands

### Nonlinear Features
- `sd1`: Standard deviation perpendicular to line of identity (Poincaré plot)
- `sd2`: Standard deviation along line of identity (Poincaré plot)
- `sd1_sd2_ratio`: Ratio of SD1 to SD2

## Output Files

The analysis generates several outputs in the `hrv_results/` directory:

1. **CSV Results**: `hrv_metrics_summary.csv` - All computed HRV metrics
2. **Plots**: 
   - `rr_ts_[subject]_Sol[sol].png` - RR interval time series
   - `welch_psd_[subject]_Sol[sol].png` - Power spectral density
   - `poincare_[subject]_Sol[sol].png` - Poincaré plot

## Troubleshooting

### If setup fails:
1. Check your Python environment
2. Ensure you have pip installed
3. Try installing packages manually: `pip install hrv-analysis`

### If imports fail:
1. Verify the package is installed: `pip list | grep hrv`
2. Check Python path issues
3. Try importing directly: `import hrvanalysis`

### If analysis fails:
1. Check your data has the required column: `heart_rate [bpm]`
2. Verify data quality (no NaN values, reasonable ranges)
3. Check that you have sufficient data points (>50 RR intervals)

## Next Steps

1. Run the setup script to install dependencies
2. Run the test suite to verify functionality
3. Use the fixed HRV analysis script with your data
4. Examine the generated plots and CSV results
5. Integrate the working code into your notebooks

The fixed implementation is more robust and provides better error handling than the original notebook version. It should work reliably with your Valquiria dataset.

## Support

If you encounter any issues:
1. Check the output from `setup_hrv_analysis.py`
2. Run `test_hrv_analysis.py` to identify specific problems
3. Review the error messages - they're designed to be informative
4. Ensure your data has the expected column names and format

The fixed tools provide comprehensive diagnostics to help identify and resolve any remaining issues. 