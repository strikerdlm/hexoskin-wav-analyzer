"""
Debug script to check what HRV columns are actually available
"""

import pandas as pd
import numpy as np

# This script helps you see what columns are available in your HRV DataFrame
# Run this in your Jupyter notebook to see the actual column names

def debug_hrv_columns(hrv_df):
    """Debug function to check available HRV columns"""
    
    print("="*60)
    print("HRV DATAFRAME COLUMN ANALYSIS")
    print("="*60)
    
    print(f"DataFrame shape: {hrv_df.shape}")
    print(f"Total columns: {len(hrv_df.columns)}")
    
    print("\nAll available columns:")
    for i, col in enumerate(hrv_df.columns):
        print(f"{i+1:2d}. {col}")
    
    print("\nColumn categories:")
    
    # Basic columns
    basic_cols = [col for col in hrv_df.columns if col in ['Subject', 'Sol', 'n_beats', 'mean_hr_bpm', 'std_hr_bpm', 'mean_rr_ms', 'std_rr_ms']]
    print(f"\nBasic columns available: {basic_cols}")
    
    # Time domain columns (common names)
    time_domain_names = ['mean_nni', 'sdnn', 'rmssd', 'nn50', 'pnn50', 'nn20', 'pnn20', 'cvnn', 'cvsd', 'median_nn', 'range_nn', 'mean_hr', 'std_hr', 'min_hr', 'max_hr']
    time_cols = [col for col in hrv_df.columns if col in time_domain_names]
    print(f"Time domain columns available: {time_cols}")
    
    # Frequency domain columns
    freq_domain_names = ['total_power', 'vlf', 'lf', 'hf', 'lf_hf_ratio', 'lfnu', 'hfnu', 'vlf_power', 'lf_power', 'hf_power']
    freq_cols = [col for col in hrv_df.columns if col in freq_domain_names]
    print(f"Frequency domain columns available: {freq_cols}")
    
    # Nonlinear columns
    nonlinear_names = ['sd1', 'sd2', 'sd1_sd2_ratio', 'ellipse_area', 'csi', 'cvi', 'modified_csi']
    nonlinear_cols = [col for col in hrv_df.columns if col in nonlinear_names]
    print(f"Nonlinear columns available: {nonlinear_cols}")
    
    # Show sample data
    print("\nSample data (first 3 rows):")
    print(hrv_df.head(3))
    
    return {
        'basic': basic_cols,
        'time_domain': time_cols,
        'frequency_domain': freq_cols,
        'nonlinear': nonlinear_cols
    }

def create_adaptive_tables(hrv_df):
    """Create tables with only available columns"""
    
    available_cols = debug_hrv_columns(hrv_df)
    
    tables = {}
    
    # Basic Summary Table
    basic_cols = ['Subject', 'Sol'] + available_cols['basic']
    if len(basic_cols) > 2:
        tables['basic_summary'] = hrv_df[basic_cols].round(2)
    
    # Time Domain Table
    time_cols = ['Subject', 'Sol'] + available_cols['time_domain']
    if len(time_cols) > 2:
        tables['time_domain'] = hrv_df[time_cols].round(2)
    
    # Frequency Domain Table
    freq_cols = ['Subject', 'Sol'] + available_cols['frequency_domain']
    if len(freq_cols) > 2:
        tables['frequency_domain'] = hrv_df[freq_cols].round(2)
    
    # Nonlinear Table
    nonlinear_cols = ['Subject', 'Sol'] + available_cols['nonlinear']
    if len(nonlinear_cols) > 2:
        tables['nonlinear'] = hrv_df[nonlinear_cols].round(2)
    
    return tables

# Use this code in your notebook:
print("""
COPY AND PASTE THIS INTO YOUR NOTEBOOK:

# Debug your HRV columns
available_cols = debug_hrv_columns(hrv_df)

# Create adaptive tables
tables = create_adaptive_tables(hrv_df)

# Display and save tables
for table_name, table_df in tables.items():
    print(f"\\n{'='*50}")
    print(f"{table_name.upper()} TABLE")
    print(f"{'='*50}")
    print(table_df.to_string(index=False))
    
    # Save to CSV
    table_df.to_csv(f'hrv_{table_name}.csv', index=False)
    print(f"💾 Saved to: hrv_{table_name}.csv")
""") 