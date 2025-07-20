"""
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
