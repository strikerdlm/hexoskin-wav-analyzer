"""
HRV Metrics Table Generator for Jupyter Notebooks
=================================================

This script generates organized tables of Heart Rate Variability (HRV) metrics
from the Valquiria dataset. It computes time-domain, frequency-domain, and 
nonlinear HRV features and presents them in well-formatted tables.

Usage in Jupyter Notebook:
    %run hrv_table_analysis.py
    
Or import functions:
    from hrv_table_analysis import generate_hrv_tables, get_hrv_summary_stats
    
Dependencies:
    pip install hrv-analysis pandas numpy matplotlib seaborn
"""

import sys
import types
import os
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import traceback

warnings.filterwarnings('ignore')

# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.signal import welch

# Apply astropy compatibility patch if needed
def apply_astropy_patch():
    """Apply compatibility patch for astropy/hrvanalysis"""
    try:
        import astropy
        try:
            from astropy.stats import LombScargle
            return True
        except ImportError:
            try:
                from astropy.timeseries import LombScargle
                stats_mod = types.ModuleType("astropy.stats")
                stats_mod.LombScargle = LombScargle
                sys.modules["astropy.stats"] = stats_mod
                astropy.stats = stats_mod
                return True
            except ImportError:
                pass
    except ImportError:
        pass
    return False

# Apply patch before importing HRV functions
apply_astropy_patch()

# HRV analysis functions
def check_hrv_dependencies():
    """Check if HRV analysis dependencies are available"""
    try:
        from hrvanalysis import (
            get_time_domain_features,
            get_frequency_domain_features,
            get_poincare_plot_features,
        )
        return True, (get_time_domain_features, get_frequency_domain_features, get_poincare_plot_features)
    except ImportError as e:
        print(f"❌ HRV analysis package not found: {e}")
        print("Please install: pip install hrv-analysis")
        return False, (None, None, None)

def load_data():
    """Load data from database or CSV files"""
    try:
        # Try loading from database first
        from scripts.load_data import load_database_data, load_csv_data
        
        df = load_database_data("merged_data.db")
        if df is not None:
            print(f"✅ Loaded {len(df)} rows from database")
            return df
        
        # Fallback to CSV
        csv_data = load_csv_data()
        if csv_data:
            df = pd.concat(csv_data.values(), ignore_index=True)
            print(f"✅ Loaded {len(df)} rows from CSV files")
            return df
        
        print("❌ No data could be loaded")
        return None
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def create_sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    subjects = ['T01_Mara', 'T02_Laura', 'T03_Nancy', 'T04_Michelle']
    sols = ['Sol2', 'Sol3', 'Sol4', 'Sol5', 'Sol6']
    
    data = []
    for subject in subjects:
        for sol in sols:
            # Generate realistic heart rate data
            base_hr = np.random.normal(70, 10)
            trend = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 5
            noise = np.random.normal(0, 3, n_samples)
            
            heart_rate = base_hr + trend + noise
            heart_rate = np.clip(heart_rate, 50, 120)
            
            for i, hr in enumerate(heart_rate):
                data.append({
                    'subject': subject,
                    'Sol': sol,
                    'heart_rate [bpm]': hr,
                    'time_index': i
                })
    
    return pd.DataFrame(data)

def rr_from_hr(hr_series: pd.Series) -> np.ndarray:
    """Convert heart rate (bpm) to RR intervals (ms)"""
    try:
        hr_clean = pd.to_numeric(hr_series, errors='coerce')
        hr_clean = hr_clean.dropna()
        
        if len(hr_clean) == 0:
            return np.array([])
        
        # Convert to RR intervals
        rr_ms = 60_000.0 / hr_clean
        rr_ms = rr_ms.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Filter physiologically plausible intervals (300-2000 ms)
        rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
        
        return rr_ms.values
        
    except Exception as e:
        print(f"❌ Error converting HR to RR: {e}")
        return np.array([])

def calculate_hrv_metrics(df: pd.DataFrame, subject: str, sol: str, hrv_functions) -> Optional[Dict[str, Any]]:
    """Calculate HRV metrics for a data segment"""
    get_time_domain_features, get_frequency_domain_features, get_poincare_plot_features = hrv_functions
    
    if get_time_domain_features is None:
        return None
    
    try:
        # Extract RR intervals
        rr_ms = rr_from_hr(df["heart_rate [bpm]"])
        
        if len(rr_ms) < 50:
            print(f"⚠️ Insufficient RR intervals for {subject} {sol}: {len(rr_ms)} intervals")
            return None
        
        # Calculate basic statistics
        results = {
            "Subject": subject,
            "Sol": sol,
            "N_RR_intervals": len(rr_ms),
            "RR_mean_ms": np.mean(rr_ms),
            "RR_std_ms": np.std(rr_ms),
            "HR_mean_bpm": 60000 / np.mean(rr_ms),
            "HR_std_bpm": np.std(60000 / rr_ms)
        }
        
        # Time domain features
        try:
            time_features = get_time_domain_features(rr_ms)
            results.update(time_features)
        except Exception as e:
            print(f"⚠️ Time domain analysis failed for {subject} {sol}: {e}")
        
        # Frequency domain features
        try:
            freq_features = get_frequency_domain_features(rr_ms)
            results.update(freq_features)
        except Exception as e:
            print(f"⚠️ Frequency domain analysis failed for {subject} {sol}: {e}")
        
        # Nonlinear features
        try:
            poincare_features = get_poincare_plot_features(rr_ms)
            results.update(poincare_features)
        except Exception as e:
            print(f"⚠️ Poincaré analysis failed for {subject} {sol}: {e}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error calculating HRV metrics for {subject} {sol}: {e}")
        return None

def generate_hrv_tables(df: pd.DataFrame = None, use_sample_data: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Generate organized HRV metrics tables
    
    Returns:
        Dictionary containing different HRV metric tables:
        - 'time_domain': Time domain HRV metrics
        - 'frequency_domain': Frequency domain HRV metrics  
        - 'nonlinear': Nonlinear HRV metrics
        - 'summary': Basic summary statistics
        - 'complete': All metrics combined
    """
    
    print("="*60)
    print("HRV METRICS TABLE GENERATOR")
    print("="*60)
    
    # Check dependencies
    has_hrv, hrv_functions = check_hrv_dependencies()
    if not has_hrv:
        print("Cannot proceed without HRV analysis package")
        return {}
    
    # Load data
    if df is None:
        if use_sample_data:
            df = create_sample_data()
            print("Using sample data for demonstration")
        else:
            df = load_data()
            if df is None:
                print("Using sample data as fallback")
                df = create_sample_data()
    
    # Check required columns
    if "heart_rate [bpm]" not in df.columns:
        print(f"❌ Required column 'heart_rate [bpm]' not found")
        print(f"Available columns: {list(df.columns)}")
        return {}
    
    # Process data by subject and Sol
    metrics_list = []
    
    # Determine grouping variables
    grouping = []
    if "subject" in df.columns:
        grouping.append("subject")
    if "Sol" in df.columns:
        grouping.append("Sol")
    
    if not grouping:
        print("Processing entire dataset as single segment")
        result = calculate_hrv_metrics(df, "All", "All", hrv_functions)
        if result:
            metrics_list.append(result)
    else:
        print(f"Processing by: {' × '.join(grouping)}")
        for keys, segment in df.groupby(grouping):
            if len(grouping) == 1:
                subject, sol = keys, "All"
            else:
                subject, sol = keys
            
            result = calculate_hrv_metrics(segment, str(subject), str(sol), hrv_functions)
            if result:
                metrics_list.append(result)
    
    if not metrics_list:
        print("❌ No HRV metrics could be calculated")
        return {}
    
    # Convert to DataFrame
    complete_df = pd.DataFrame(metrics_list)
    
    # Create organized tables
    tables = {}
    
    # Basic summary table
    summary_cols = [
        "Subject", "Sol", "N_RR_intervals", "RR_mean_ms", "RR_std_ms", 
        "HR_mean_bpm", "HR_std_bpm"
    ]
    tables['summary'] = complete_df[summary_cols]
    
    # Time domain metrics
    time_domain_cols = [
        "Subject", "Sol", "mean_nni", "sdnn", "rmssd", "nn50", "pnn50",
        "nn20", "pnn20", "cvnn", "cvsd", "median_nn", "range_nn",
        "mean_hr", "std_hr", "min_hr", "max_hr"
    ]
    time_domain_available = [col for col in time_domain_cols if col in complete_df.columns]
    if len(time_domain_available) > 2:  # More than just Subject and Sol
        tables['time_domain'] = complete_df[time_domain_available]
    
    # Frequency domain metrics
    freq_domain_cols = [
        "Subject", "Sol", "total_power", "vlf", "lf", "hf", "lf_hf_ratio",
        "lfnu", "hfnu", "vlf_power", "lf_power", "hf_power"
    ]
    freq_domain_available = [col for col in freq_domain_cols if col in complete_df.columns]
    if len(freq_domain_available) > 2:
        tables['frequency_domain'] = complete_df[freq_domain_available]
    
    # Nonlinear metrics
    nonlinear_cols = [
        "Subject", "Sol", "sd1", "sd2", "sd1_sd2_ratio", "ellipse_area",
        "csi", "cvi", "modified_csi"
    ]
    nonlinear_available = [col for col in nonlinear_cols if col in complete_df.columns]
    if len(nonlinear_available) > 2:
        tables['nonlinear'] = complete_df[nonlinear_available]
    
    # Complete table
    tables['complete'] = complete_df
    
    print(f"\n✅ Generated {len(tables)} HRV metric tables")
    for table_name, table_df in tables.items():
        print(f"   {table_name}: {len(table_df)} rows × {len(table_df.columns)} columns")
    
    return tables

def display_hrv_tables(tables: Dict[str, pd.DataFrame], max_rows: int = 10):
    """Display HRV tables in a formatted way"""
    
    if not tables:
        print("No tables to display")
        return
    
    for table_name, table_df in tables.items():
        print(f"\n{'='*60}")
        print(f"HRV {table_name.upper().replace('_', ' ')} METRICS")
        print(f"{'='*60}")
        
        if len(table_df) > max_rows:
            print(f"Showing first {max_rows} rows of {len(table_df)} total:")
            display_df = table_df.head(max_rows)
        else:
            display_df = table_df
        
        print(display_df.to_string(index=False))
        
        if len(table_df) > max_rows:
            print(f"... and {len(table_df) - max_rows} more rows")

def get_hrv_summary_stats(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Generate summary statistics for HRV metrics"""
    
    if not tables or 'complete' not in tables:
        return {}
    
    complete_df = tables['complete']
    
    # Numeric columns only
    numeric_cols = complete_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['Subject', 'Sol']]
    
    if not numeric_cols:
        return {}
    
    stats = {}
    
    # Overall statistics
    stats['overall'] = complete_df[numeric_cols].describe()
    
    # By subject if available
    if 'Subject' in complete_df.columns:
        subject_stats = complete_df.groupby('Subject')[numeric_cols].mean()
        stats['by_subject'] = subject_stats
    
    # By Sol if available
    if 'Sol' in complete_df.columns:
        sol_stats = complete_df.groupby('Sol')[numeric_cols].mean()
        stats['by_sol'] = sol_stats
    
    return stats

def save_hrv_tables(tables: Dict[str, pd.DataFrame], output_dir: str = "hrv_results"):
    """Save HRV tables to CSV files"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    saved_files = []
    
    for table_name, table_df in tables.items():
        filename = f"hrv_{table_name}_metrics.csv"
        filepath = output_path / filename
        table_df.to_csv(filepath, index=False)
        saved_files.append(filepath)
        print(f"✅ Saved {table_name} table to {filepath}")
    
    return saved_files

def main():
    """Main function for standalone execution"""
    
    # Generate HRV tables
    tables = generate_hrv_tables()
    
    if not tables:
        print("Failed to generate HRV tables")
        return
    
    # Display tables
    display_hrv_tables(tables)
    
    # Generate summary statistics
    summary_stats = get_hrv_summary_stats(tables)
    
    if summary_stats:
        print(f"\n{'='*60}")
        print("HRV SUMMARY STATISTICS")
        print(f"{'='*60}")
        
        for stat_type, stat_df in summary_stats.items():
            print(f"\n{stat_type.upper().replace('_', ' ')}:")
            print(stat_df.round(2))
    
    # Save tables
    saved_files = save_hrv_tables(tables)
    
    return tables, summary_stats

if __name__ == "__main__":
    main() 