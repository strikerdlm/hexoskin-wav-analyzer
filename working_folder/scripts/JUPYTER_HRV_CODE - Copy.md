# HRV Metrics Tables for Jupyter Notebooks

## Quick Start - Copy and Paste This Code

```python
# =============================================================================
# CELL 1: Install and Import Required Packages
# =============================================================================

# Install required packages (run once)
!pip install hrv-analysis pandas numpy matplotlib seaborn

# Import libraries
import pandas as pd
import numpy as np
import sqlite3
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# CELL 2: Load Your Data
# =============================================================================

def load_data():
    """Load the Valquiria dataset"""
    
    # Try loading from database first
    try:
        conn = sqlite3.connect("merged_data.db")
        df = pd.read_sql_query("SELECT * FROM merged_data", conn)
        conn.close()
        print(f"✅ Loaded {len(df)} rows from database")
        return df
    except:
        print("Database not found, trying CSV files...")
    
    # Try loading CSV files
    csv_files = list(Path(".").glob("*.csv"))
    if csv_files:
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                print(f"✅ Loaded {file.name}")
            except:
                continue
        if dfs:
            return pd.concat(dfs, ignore_index=True)
    
    print("❌ No data found")
    return None

# Load your data
df = load_data()

# Check the structure
if df is not None:
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Subjects: {df['subject'].unique() if 'subject' in df.columns else 'N/A'}")
    print(f"Sols: {df['Sol'].unique() if 'Sol' in df.columns else 'N/A'}")

# =============================================================================
# CELL 3: Calculate HRV Metrics
# =============================================================================

def calculate_hrv_metrics(hr_data):
    """Calculate HRV metrics from heart rate data"""
    
    try:
        # Import HRV analysis functions
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
        rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]  # Filter physiological range
        
        if len(rr_ms) < 50:
            return None
        
        # Calculate all HRV features
        metrics = {}
        
        # Basic stats
        metrics['n_beats'] = len(rr_ms)
        metrics['mean_hr_bpm'] = 60000 / np.mean(rr_ms)
        metrics['std_hr_bpm'] = np.std(60000 / rr_ms)
        metrics['mean_rr_ms'] = np.mean(rr_ms)
        metrics['std_rr_ms'] = np.std(rr_ms)
        
        # Advanced HRV features
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
        print("❌ hrv-analysis package not found. Install with: pip install hrv-analysis")
        return None

# Calculate HRV metrics for each subject/Sol combination
print("Calculating HRV metrics...")
results = []

if df is not None and 'heart_rate [bpm]' in df.columns:
    
    # Group by subject and Sol
    for (subject, sol), group in df.groupby(['subject', 'Sol']):
        print(f"Processing {subject} {sol}... ", end="")
        
        metrics = calculate_hrv_metrics(group['heart_rate [bpm]'])
        if metrics:
            metrics['Subject'] = subject
            metrics['Sol'] = sol
            results.append(metrics)
            print("✅")
        else:
            print("❌ (insufficient data)")

print(f"\n✅ Calculated HRV metrics for {len(results)} segments")

# =============================================================================
# CELL 4: Create HRV Tables
# =============================================================================

if results:
    # Convert to DataFrame
    hrv_df = pd.DataFrame(results)
    
    # Create organized tables
    tables = {}
    
    # 1. Basic Summary Table
    basic_cols = ['Subject', 'Sol', 'n_beats', 'mean_hr_bpm', 'std_hr_bpm', 
                  'mean_rr_ms', 'std_rr_ms']
    basic_available = [col for col in basic_cols if col in hrv_df.columns]
    if len(basic_available) > 2:
        tables['Basic_Summary'] = hrv_df[basic_available].round(2)
    
    # 2. Time Domain Metrics
    time_cols = ['Subject', 'Sol', 'mean_nni', 'sdnn', 'rmssd', 'nn50', 'pnn50',
                 'nn20', 'pnn20', 'cvnn', 'cvsd']
    time_available = [col for col in time_cols if col in hrv_df.columns]
    if len(time_available) > 2:
        tables['Time_Domain'] = hrv_df[time_available].round(2)
    
    # 3. Frequency Domain Metrics
    freq_cols = ['Subject', 'Sol', 'total_power', 'vlf', 'lf', 'hf', 'lf_hf_ratio',
                 'lfnu', 'hfnu']
    freq_available = [col for col in freq_cols if col in hrv_df.columns]
    if len(freq_available) > 2:
        tables['Frequency_Domain'] = hrv_df[freq_available].round(2)
    
    # 4. Nonlinear Metrics
    nonlinear_cols = ['Subject', 'Sol', 'sd1', 'sd2', 'sd1_sd2_ratio', 'ellipse_area']
    nonlinear_available = [col for col in nonlinear_cols if col in hrv_df.columns]
    if len(nonlinear_available) > 2:
        tables['Nonlinear'] = hrv_df[nonlinear_available].round(2)
    
    # 5. Complete Table
    tables['Complete'] = hrv_df.round(2)
    
    print(f"✅ Created {len(tables)} HRV tables")
    
    # Display tables
    for table_name, table_df in tables.items():
        print(f"\n{'='*60}")
        print(f"HRV {table_name.upper()} METRICS")
        print(f"{'='*60}")
        print(table_df.to_string(index=False))
        
        # Save to CSV
        table_df.to_csv(f"hrv_{table_name.lower()}.csv", index=False)
        print(f"💾 Saved to: hrv_{table_name.lower()}.csv")

# =============================================================================
# CELL 5: Summary Statistics
# =============================================================================

if results:
    # Calculate summary statistics
    numeric_cols = hrv_df.select_dtypes(include=[np.number]).columns
    
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    # Overall statistics
    print("\nOVERALL STATISTICS:")
    print(hrv_df[numeric_cols].describe().round(2))
    
    # By subject
    print("\nMEAN VALUES BY SUBJECT:")
    print(hrv_df.groupby('Subject')[numeric_cols].mean().round(2))
    
    # By Sol
    print("\nMEAN VALUES BY SOL:")
    print(hrv_df.groupby('Sol')[numeric_cols].mean().round(2))
    
    # Save summary statistics
    hrv_df[numeric_cols].describe().round(2).to_csv("hrv_summary_statistics.csv")
    hrv_df.groupby('Subject')[numeric_cols].mean().round(2).to_csv("hrv_by_subject.csv")
    hrv_df.groupby('Sol')[numeric_cols].mean().round(2).to_csv("hrv_by_sol.csv")
    
    print("\n💾 Summary statistics saved to CSV files")
```

## What Each Table Contains

### 1. Basic Summary Table
- **n_beats**: Number of heartbeats
- **mean_hr_bpm**: Average heart rate (beats per minute)
- **std_hr_bpm**: Standard deviation of heart rate
- **mean_rr_ms**: Average RR interval (milliseconds)
- **std_rr_ms**: Standard deviation of RR intervals

### 2. Time Domain Metrics
- **mean_nni**: Mean of RR intervals
- **sdnn**: Standard deviation of RR intervals
- **rmssd**: Root mean square of successive differences
- **nn50**: Number of successive RR intervals differing by > 50ms
- **pnn50**: Percentage of successive RR intervals differing by > 50ms
- **cvnn**: Coefficient of variation of RR intervals

### 3. Frequency Domain Metrics
- **total_power**: Total power in all frequency bands
- **vlf**: Very low frequency power (0.003-0.04 Hz)
- **lf**: Low frequency power (0.04-0.15 Hz)
- **hf**: High frequency power (0.15-0.4 Hz)
- **lf_hf_ratio**: Ratio of LF to HF power
- **lfnu**: LF power in normalized units
- **hfnu**: HF power in normalized units

### 4. Nonlinear Metrics
- **sd1**: Standard deviation perpendicular to line of identity (Poincaré plot)
- **sd2**: Standard deviation along line of identity (Poincaré plot)
- **sd1_sd2_ratio**: Ratio of SD1 to SD2
- **ellipse_area**: Area of Poincaré plot ellipse

## Quick Usage

1. **Copy the code above** into separate cells in your Jupyter notebook
2. **Run Cell 1** to install packages and import libraries
3. **Run Cell 2** to load your data
4. **Run Cell 3** to calculate HRV metrics
5. **Run Cell 4** to create and display tables
6. **Run Cell 5** to get summary statistics

## Output Files

The code will create several CSV files:
- `hrv_basic_summary.csv` - Basic heart rate statistics
- `hrv_time_domain.csv` - Time domain HRV metrics
- `hrv_frequency_domain.csv` - Frequency domain HRV metrics
- `hrv_nonlinear.csv` - Nonlinear HRV metrics
- `hrv_complete.csv` - All metrics combined
- `hrv_summary_statistics.csv` - Overall statistics
- `hrv_by_subject.csv` - Metrics averaged by subject
- `hrv_by_sol.csv` - Metrics averaged by Sol

## Troubleshooting

- **Package not found**: Run `!pip install hrv-analysis` in a cell
- **No data loaded**: Make sure your data files are in the current directory
- **Missing heart rate column**: Check that your data has a column named `heart_rate [bpm]`
- **Insufficient data**: Each segment needs at least 50 valid heartbeats for HRV analysis

## Advanced Usage

You can modify the code to:
- Filter data by specific time periods
- Calculate HRV for different groupings
- Add custom HRV metrics
- Create visualizations of the results 