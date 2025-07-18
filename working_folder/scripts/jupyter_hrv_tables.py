"""
HRV Metrics Tables for Jupyter Notebooks
=========================================

Simple, self-contained code to extract HRV metrics tables from your data.
Copy and paste this code into your Jupyter notebook cells.
"""

import pandas as pd
import numpy as np
import sqlite3
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: Setup and Data Loading
# =============================================================================

def load_valquiria_data():
    """Load the Valquiria dataset from database or CSV files"""
    
    # Try loading from database first
    db_path = Path("merged_data.db")
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query("SELECT * FROM merged_data", conn)
            conn.close()
            print(f"✅ Loaded {len(df)} rows from database")
            return df
        except:
            pass
    
    # Fallback to CSV loading
    csv_files = list(Path(".").glob("*.csv"))
    if csv_files:
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                print(f"✅ Loaded {file.name}: {len(df)} rows")
            except:
                continue
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            print(f"✅ Combined data: {len(df)} rows")
            return df
    
    print("❌ No data found - creating sample data")
    return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    subjects = ['T01_Mara', 'T02_Laura', 'T03_Nancy', 'T04_Michelle']
    sols = ['Sol2', 'Sol3', 'Sol4', 'Sol5']
    
    data = []
    for subject in subjects:
        for sol in sols:
            # Generate realistic heart rate data
            base_hr = np.random.normal(70, 5)
            n_points = 500
            trend = np.sin(np.linspace(0, 2*np.pi, n_points)) * 8
            noise = np.random.normal(0, 3, n_points)
            
            heart_rate = base_hr + trend + noise
            heart_rate = np.clip(heart_rate, 50, 120)
            
            for i, hr in enumerate(heart_rate):
                data.append({
                    'subject': subject,
                    'Sol': sol,
                    'heart_rate [bpm]': hr,
                    'time_point': i
                })
    
    return pd.DataFrame(data)

# =============================================================================
# STEP 2: HRV Calculation Functions
# =============================================================================

def calculate_basic_hrv_metrics(hr_data):
    """Calculate basic HRV metrics without external packages"""
    
    # Convert HR to RR intervals (ms)
    hr_clean = pd.to_numeric(hr_data, errors='coerce').dropna()
    
    if len(hr_clean) < 10:
        return None
    
    rr_ms = 60000 / hr_clean
    rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]  # Physiological range
    
    if len(rr_ms) < 10:
        return None
    
    # Calculate basic time-domain metrics
    metrics = {
        'n_beats': len(rr_ms),
        'mean_rr_ms': np.mean(rr_ms),
        'std_rr_ms': np.std(rr_ms),
        'mean_hr_bpm': 60000 / np.mean(rr_ms),
        'std_hr_bpm': np.std(60000 / rr_ms),
        'min_hr_bpm': 60000 / np.max(rr_ms),
        'max_hr_bpm': 60000 / np.min(rr_ms),
        'cvnn': np.std(rr_ms) / np.mean(rr_ms) * 100,  # Coefficient of variation
    }
    
    # Calculate RMSSD (root mean square of successive differences)
    if len(rr_ms) > 1:
        rr_diff = np.diff(rr_ms)
        metrics['rmssd'] = np.sqrt(np.mean(rr_diff**2))
        
        # Calculate pNN50 (percentage of successive RR intervals > 50ms)
        nn50 = np.sum(np.abs(rr_diff) > 50)
        metrics['nn50'] = nn50
        metrics['pnn50'] = (nn50 / len(rr_diff)) * 100
    
    return metrics

def calculate_advanced_hrv_metrics(hr_data):
    """Calculate advanced HRV metrics using hrv-analysis package"""
    
    try:
        # Try to import hrv-analysis
        from hrvanalysis import (
            get_time_domain_features,
            get_frequency_domain_features,
            get_poincare_plot_features,
        )
        
        # Convert HR to RR intervals
        hr_clean = pd.to_numeric(hr_data, errors='coerce').dropna()
        if len(hr_clean) < 50:
            return None
        
        rr_ms = 60000 / hr_clean
        rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
        
        if len(rr_ms) < 50:
            return None
        
        # Calculate all HRV features
        metrics = {}
        
        try:
            time_features = get_time_domain_features(rr_ms.values)
            metrics.update(time_features)
        except:
            pass
        
        try:
            freq_features = get_frequency_domain_features(rr_ms.values)
            metrics.update(freq_features)
        except:
            pass
        
        try:
            poincare_features = get_poincare_plot_features(rr_ms.values)
            metrics.update(poincare_features)
        except:
            pass
        
        return metrics
        
    except ImportError:
        print("ℹ️ hrv-analysis package not available, using basic metrics only")
        return calculate_basic_hrv_metrics(hr_data)

# =============================================================================
# STEP 3: Main HRV Table Generation
# =============================================================================

def generate_hrv_tables(df=None):
    """
    Generate HRV metrics tables
    
    Returns:
        dict: Dictionary with different HRV metric tables
    """
    
    print("="*60)
    print("GENERATING HRV METRICS TABLES")
    print("="*60)
    
    # Load data if not provided
    if df is None:
        df = load_valquiria_data()
    
    # Check for required column
    if 'heart_rate [bpm]' not in df.columns:
        print(f"❌ Required column 'heart_rate [bpm]' not found")
        print(f"Available columns: {list(df.columns)}")
        return {}
    
    # Process data by subject and Sol
    results = []
    
    # Determine grouping
    grouping = []
    if 'subject' in df.columns:
        grouping.append('subject')
    if 'Sol' in df.columns:
        grouping.append('Sol')
    
    if not grouping:
        print("Processing entire dataset as single group")
        metrics = calculate_advanced_hrv_metrics(df['heart_rate [bpm]'])
        if metrics:
            metrics['Subject'] = 'All'
            metrics['Sol'] = 'All'
            results.append(metrics)
    else:
        print(f"Processing by: {' × '.join(grouping)}")
        
        for keys, group in df.groupby(grouping):
            if len(grouping) == 1:
                subject, sol = keys, 'All'
            else:
                subject, sol = keys
            
            print(f"  Processing {subject} {sol}... ", end="")
            
            metrics = calculate_advanced_hrv_metrics(group['heart_rate [bpm]'])
            if metrics:
                metrics['Subject'] = subject
                metrics['Sol'] = sol
                results.append(metrics)
                print("✅")
            else:
                print("❌ (insufficient data)")
    
    if not results:
        print("❌ No HRV metrics could be calculated")
        return {}
    
    # Convert to DataFrame
    hrv_df = pd.DataFrame(results)
    
    # Organize into different tables
    tables = {}
    
    # Basic summary table
    basic_cols = [
        'Subject', 'Sol', 'n_beats', 'mean_hr_bpm', 'std_hr_bpm',
        'min_hr_bpm', 'max_hr_bpm', 'mean_rr_ms', 'std_rr_ms', 'cvnn'
    ]
    basic_available = [col for col in basic_cols if col in hrv_df.columns]
    if len(basic_available) > 2:
        tables['basic'] = hrv_df[basic_available].round(2)
    
    # Time domain metrics
    time_cols = [
        'Subject', 'Sol', 'mean_nni', 'sdnn', 'rmssd', 'nn50', 'pnn50',
        'nn20', 'pnn20', 'cvnn', 'cvsd'
    ]
    time_available = [col for col in time_cols if col in hrv_df.columns]
    if len(time_available) > 2:
        tables['time_domain'] = hrv_df[time_available].round(2)
    
    # Frequency domain metrics
    freq_cols = [
        'Subject', 'Sol', 'total_power', 'vlf', 'lf', 'hf', 'lf_hf_ratio',
        'lfnu', 'hfnu'
    ]
    freq_available = [col for col in freq_cols if col in hrv_df.columns]
    if len(freq_available) > 2:
        tables['frequency_domain'] = hrv_df[freq_available].round(2)
    
    # Nonlinear metrics
    nonlinear_cols = [
        'Subject', 'Sol', 'sd1', 'sd2', 'sd1_sd2_ratio', 'ellipse_area'
    ]
    nonlinear_available = [col for col in nonlinear_cols if col in hrv_df.columns]
    if len(nonlinear_available) > 2:
        tables['nonlinear'] = hrv_df[nonlinear_available].round(2)
    
    # Complete table
    tables['complete'] = hrv_df.round(2)
    
    print(f"\n✅ Generated {len(tables)} HRV tables with {len(results)} data points")
    
    return tables

# =============================================================================
# STEP 4: Display and Summary Functions
# =============================================================================

def display_hrv_tables(tables, max_rows=10):
    """Display HRV tables in a nice format"""
    
    for table_name, table_df in tables.items():
        print(f"\n{'='*60}")
        print(f"HRV {table_name.upper().replace('_', ' ')} METRICS")
        print(f"{'='*60}")
        
        if len(table_df) > max_rows:
            print(f"First {max_rows} rows of {len(table_df)} total:")
            display_df = table_df.head(max_rows)
        else:
            display_df = table_df
        
        print(display_df.to_string(index=False))
        
        if len(table_df) > max_rows:
            print(f"... and {len(table_df) - max_rows} more rows")

def get_summary_statistics(tables):
    """Get summary statistics for HRV metrics"""
    
    if 'complete' not in tables:
        return None
    
    df = tables['complete']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    summary_stats = {
        'overall': df[numeric_cols].describe().round(2),
    }
    
    if 'Subject' in df.columns:
        summary_stats['by_subject'] = df.groupby('Subject')[numeric_cols].mean().round(2)
    
    if 'Sol' in df.columns:
        summary_stats['by_sol'] = df.groupby('Sol')[numeric_cols].mean().round(2)
    
    return summary_stats

# =============================================================================
# STEP 5: Easy-to-Use Main Function
# =============================================================================

def run_hrv_analysis():
    """
    Main function to run complete HRV analysis
    
    Returns:
        tuple: (tables, summary_stats) containing all results
    """
    
    # Generate tables
    tables = generate_hrv_tables()
    
    if not tables:
        print("❌ Failed to generate HRV tables")
        return None, None
    
    # Display tables
    display_hrv_tables(tables)
    
    # Get summary statistics
    summary_stats = get_summary_statistics(tables)
    
    if summary_stats:
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        
        for stat_name, stat_df in summary_stats.items():
            print(f"\n{stat_name.upper().replace('_', ' ')}:")
            print(stat_df)
    
    # Save to CSV files
    output_dir = Path("hrv_results")
    output_dir.mkdir(exist_ok=True)
    
    saved_files = []
    for table_name, table_df in tables.items():
        filename = f"hrv_{table_name}.csv"
        filepath = output_dir / filename
        table_df.to_csv(filepath, index=False)
        saved_files.append(filepath)
    
    print(f"\n✅ Saved {len(saved_files)} CSV files to {output_dir}/")
    
    return tables, summary_stats

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    # Run complete analysis
    tables, stats = run_hrv_analysis()
    
    # You can also run individual steps:
    # tables = generate_hrv_tables()
    # display_hrv_tables(tables)
    # stats = get_summary_statistics(tables) 