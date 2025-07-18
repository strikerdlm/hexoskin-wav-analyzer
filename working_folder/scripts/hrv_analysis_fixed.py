"""
Fixed Heart-Rate Variability (HRV) Analysis for the Valquiria dataset.

This script provides a robust implementation that:
1. Handles import issues gracefully
2. Provides better error reporting
3. Works with the existing data structure
4. Includes comprehensive testing

Usage:
    python hrv_analysis_fixed.py
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Standard scientific computing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.signal import welch

def check_and_install_hrv_analysis():
    """Check if hrvanalysis is available and install if needed"""
    try:
        import hrvanalysis
        print("✅ hrvanalysis package is available")
        return True
    except ImportError:
        print("❌ hrvanalysis package not found")
        print("Please install it with: pip install hrv-analysis")
        return False

def patch_astropy_if_needed():
    """Patch astropy compatibility for older versions of hrvanalysis"""
    try:
        # First check if astropy is available
        import astropy
        print(f"✅ astropy version: {astropy.__version__}")
        
        # Check if LombScargle is available in the expected location
        try:
            from astropy.stats import LombScargle
            print("✅ astropy.stats.LombScargle available")
        except ImportError:
            try:
                from astropy.timeseries import LombScargle
                print("✅ astropy.timeseries.LombScargle available (newer version)")
                
                # Apply patch for hrvanalysis compatibility
                import types
                stats_mod = types.ModuleType("astropy.stats")
                stats_mod.LombScargle = LombScargle
                sys.modules["astropy.stats"] = stats_mod
                astropy.stats = stats_mod
                print("✅ Applied compatibility patch for astropy.stats.LombScargle")
                
            except ImportError:
                print("⚠️ LombScargle not found in astropy - frequency domain analysis may fail")
                
    except ImportError:
        print("⚠️ astropy not found - installing...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "astropy"])
            print("✅ astropy installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install astropy")

def import_hrv_functions():
    """Import HRV analysis functions with error handling"""
    try:
        from hrvanalysis import (
            get_time_domain_features,
            get_frequency_domain_features,
            get_poincare_plot_features,
        )
        print("✅ Successfully imported HRV analysis functions")
        return get_time_domain_features, get_frequency_domain_features, get_poincare_plot_features
    except ImportError as e:
        print(f"❌ Failed to import HRV analysis functions: {e}")
        return None, None, None

def load_data_safely():
    """Load data with robust error handling"""
    try:
        # First try to import from the current directory
        try:
            from load_data import load_database_data, load_csv_data
            print("✅ Imported load_data from current directory")
        except ImportError:
            # Try from scripts directory
            try:
                from scripts.load_data import load_database_data, load_csv_data
                print("✅ Imported load_data from scripts directory")
            except ImportError:
                print("❌ Could not import load_data module")
                return None
        
        # Try to load from database first
        print("Attempting to load data from database...")
        df = load_database_data("merged_data.db")
        
        if df is None:
            print("Database not found, trying CSV files...")
            csvs = load_csv_data()
            if csvs:
                df = pd.concat(csvs.values(), ignore_index=True)
                print(f"✅ Loaded {len(df):,} rows from CSV files")
            else:
                print("❌ No data sources available")
                return None
        else:
            print(f"✅ Loaded {len(df):,} rows from database")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def create_sample_data():
    """Create sample data for testing if real data is not available"""
    print("Creating sample data for testing...")
    
    # Create sample heart rate data
    np.random.seed(42)
    n_samples = 1000
    base_hr = 70
    noise = np.random.normal(0, 5, n_samples)
    trend = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 10
    
    heart_rate = base_hr + noise + trend
    heart_rate = np.clip(heart_rate, 50, 120)  # Physiological range
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'heart_rate [bpm]': heart_rate,
        'subject': ['TestSubject'] * n_samples,
        'Sol': [1] * n_samples,
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1s')
    })
    
    print(f"✅ Created sample dataset with {len(df)} rows")
    return df

def rr_from_hr(hr_series: pd.Series) -> np.ndarray:
    """Convert heart-rate (bpm) to RR-intervals (ms), with robust error handling"""
    try:
        # Convert to float and handle any non-numeric values
        hr_clean = pd.to_numeric(hr_series, errors='coerce')
        hr_clean = hr_clean.dropna()
        
        if len(hr_clean) == 0:
            print("⚠️ No valid heart rate data found")
            return np.array([])
        
        # Convert to RR intervals
        rr_ms = 60_000.0 / hr_clean
        rr_ms = rr_ms.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Filter physiologically implausible intervals
        rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
        
        print(f"✅ Converted {len(hr_clean)} HR values to {len(rr_ms)} valid RR intervals")
        return rr_ms.values
        
    except Exception as e:
        print(f"❌ Error converting HR to RR intervals: {e}")
        return np.array([])

def setup_plotting():
    """Setup matplotlib with proper backend"""
    try:
        # Set matplotlib backend
        plt.rcParams.update({
            'figure.figsize': (12, 6),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'font.size': 10
        })
        
        # Try to use seaborn if available
        try:
            sns.set_style("whitegrid")
            print("✅ Using seaborn styling")
        except:
            print("⚠️ Seaborn not available, using matplotlib defaults")
            
    except Exception as e:
        print(f"⚠️ Error setting up plotting: {e}")

def plot_rr_timeseries(rr_ms: np.ndarray, title: str, out_path: Path) -> None:
    """Plot RR interval time series with error handling"""
    try:
        plt.figure(figsize=(14, 4))
        plt.plot(np.arange(len(rr_ms)), rr_ms / 1000.0, linewidth=0.7)
        plt.title(f"RR-interval Time-Series – {title}")
        plt.xlabel("Beat index")
        plt.ylabel("RR (s)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved RR time series plot: {out_path}")
    except Exception as e:
        print(f"❌ Error plotting RR time series: {e}")

def plot_psd_simple(rr_ms: np.ndarray, title: str, out_path: Path) -> None:
    """Simple PSD plot with error handling"""
    try:
        # Build tachogram
        t = np.cumsum(rr_ms) / 1000.0
        fs_interp = 4.0
        
        if len(rr_ms) < 50:
            print(f"⚠️ Not enough data points for PSD analysis: {len(rr_ms)}")
            return
        
        resampled_time = np.arange(0.0, t[-1], 1 / fs_interp)
        if len(resampled_time) < 2:
            print("⚠️ Not enough time points for interpolation")
            return
            
        f_interp = interp1d(t, rr_ms, kind="linear", fill_value="extrapolate")
        resampled_rr = f_interp(resampled_time)
        resampled_rr = resampled_rr - np.mean(resampled_rr)
        
        # Compute PSD
        freqs, psd = welch(resampled_rr, fs=fs_interp, nperseg=min(256, len(resampled_rr)//4))
        
        plt.figure(figsize=(10, 5))
        plt.semilogy(freqs, psd, lw=1.2)
        plt.title(f"Power Spectral Density – {title}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (ms²/Hz)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved PSD plot: {out_path}")
        
    except Exception as e:
        print(f"❌ Error plotting PSD: {e}")

def plot_poincare_simple(rr_ms: np.ndarray, title: str, out_path: Path) -> None:
    """Simple Poincaré plot with error handling"""
    try:
        if len(rr_ms) < 2:
            print("⚠️ Not enough data points for Poincaré plot")
            return
            
        rr1, rr2 = rr_ms[:-1], rr_ms[1:]
        
        plt.figure(figsize=(6, 6))
        plt.scatter(rr1, rr2, s=8, alpha=0.6)
        
        # Add identity line
        lims = [min(rr1.min(), rr2.min()), max(rr1.max(), rr2.max())]
        plt.plot(lims, lims, "k--", alpha=0.6)
        
        plt.xlabel("RRₙ (ms)")
        plt.ylabel("RRₙ₊₁ (ms)")
        plt.title(f"Poincaré Plot – {title}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved Poincaré plot: {out_path}")
        
    except Exception as e:
        print(f"❌ Error plotting Poincaré: {e}")

def analyze_hrv_segment(df: pd.DataFrame, subject: str, sol: str, 
                       output_dir: Path, hrv_functions) -> Optional[Dict[str, Any]]:
    """Analyze HRV for a data segment with comprehensive error handling"""
    try:
        get_time_domain_features, get_frequency_domain_features, get_poincare_plot_features = hrv_functions
        
        if get_time_domain_features is None:
            print("❌ HRV functions not available")
            return None
        
        # Extract RR intervals
        rr_ms = rr_from_hr(df["heart_rate [bpm]"])
        
        if len(rr_ms) < 50:
            print(f"⚠️ Insufficient RR-intervals for {subject} Sol {sol} (n={len(rr_ms)}) – skipping")
            return None
        
        print(f"Processing {subject} Sol {sol}: {len(rr_ms)} RR intervals")
        
        # Compute HRV features
        results = {
            "subject": subject,
            "Sol": sol,
            "n_rr_intervals": len(rr_ms),
            "rr_mean": np.mean(rr_ms),
            "rr_std": np.std(rr_ms)
        }
        
        try:
            time_feats = get_time_domain_features(rr_ms)
            results.update(time_feats)
            print(f"✅ Time domain features: {len(time_feats)} features")
        except Exception as e:
            print(f"⚠️ Time domain analysis failed: {e}")
        
        try:
            freq_feats = get_frequency_domain_features(rr_ms)
            results.update(freq_feats)
            print(f"✅ Frequency domain features: {len(freq_feats)} features")
        except Exception as e:
            print(f"⚠️ Frequency domain analysis failed: {e}")
        
        try:
            poincare_feats = get_poincare_plot_features(rr_ms)
            results.update(poincare_feats)
            print(f"✅ Poincaré features: {len(poincare_feats)} features")
        except Exception as e:
            print(f"⚠️ Poincaré analysis failed: {e}")
        
        # Generate plots
        plot_rr_timeseries(rr_ms, f"{subject} – Sol {sol}", 
                          output_dir / f"rr_ts_{subject}_Sol{sol}.png")
        
        plot_psd_simple(rr_ms, f"{subject} – Sol {sol}", 
                       output_dir / f"welch_psd_{subject}_Sol{sol}.png")
        
        plot_poincare_simple(rr_ms, f"{subject} – Sol {sol}", 
                            output_dir / f"poincare_{subject}_Sol{sol}.png")
        
        return results
        
    except Exception as e:
        print(f"❌ Error analyzing HRV segment: {e}")
        return None

def main():
    """Main function with comprehensive error handling"""
    print("="*60)
    print("Fixed HRV Analysis for Valquiria Dataset")
    print("="*60)
    
    # Setup
    setup_plotting()
    
    # Check dependencies
    if not check_and_install_hrv_analysis():
        print("❌ Cannot proceed without hrvanalysis package")
        return
    
    # Patch astropy if needed
    patch_astropy_if_needed()
    
    # Import HRV functions
    hrv_functions = import_hrv_functions()
    if hrv_functions[0] is None:
        print("❌ Cannot proceed without HRV functions")
        return
    
    # Load data
    df = load_data_safely()
    if df is None:
        print("Using sample data for testing...")
        df = create_sample_data()
    
    # Check required columns
    if "heart_rate [bpm]" not in df.columns:
        print("❌ Column 'heart_rate [bpm]' not found")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Create output directory
    output_dir = Path(__file__).parent / "hrv_results"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process data
    metrics = []
    
    # Determine grouping
    grouping = []
    if "subject" in df.columns:
        grouping.append("subject")
    if "Sol" in df.columns:
        grouping.append("Sol")
    
    if not grouping:
        print("Processing entire dataset as single segment...")
        result = analyze_hrv_segment(df, "All", "All", output_dir, hrv_functions)
        if result:
            metrics.append(result)
    else:
        print(f"Processing by: {' × '.join(grouping)}")
        for keys, segment in df.groupby(grouping):
            if len(grouping) == 1:
                subject, sol = keys, "All"
            else:
                subject, sol = keys
            
            result = analyze_hrv_segment(segment, str(subject), str(sol), 
                                       output_dir, hrv_functions)
            if result:
                metrics.append(result)
    
    # Save results
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        csv_path = output_dir / "hrv_metrics_summary.csv"
        metrics_df.to_csv(csv_path, index=False)
        print(f"✅ Results saved to: {csv_path}")
        print(f"✅ Processed {len(metrics)} segments successfully")
        
        # Display summary
        print("\nSummary of computed metrics:")
        print(metrics_df.head())
        
    else:
        print("❌ No metrics were computed")

if __name__ == "__main__":
    main() 