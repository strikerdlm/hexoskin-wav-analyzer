# 🔧 Jupyter Notebook Fix Instructions

## ✅ PROBLEM SOLVED! 

All critical issues with your Jupyter notebooks have been fixed:

- ✅ **Missing Dependencies**: All required packages installed
- ✅ **HRV Analysis Errors**: Astropy compatibility patches applied
- ✅ **Import Issues**: Compatibility module created
- ✅ **Data Loading**: Database and CSV loading verified
- ✅ **Environment Setup**: Jupyter kernel configured

---

## 🚀 How to Use Your Fixed Notebooks

### Option 1: Use the New Demo Notebook (Recommended)
1. Open `FIXED_Analysis_Demo.ipynb` 
2. This demonstrates all functionality working correctly
3. Copy code patterns from this notebook to your analysis

### Option 2: Fix Your Existing Notebooks

Add these lines at the top of any notebook cell:

```python
# Fix for all compatibility issues
import sys
sys.path.append('..')  # Adjust path if needed
import valquiria_compat

# Load data safely
df = valquiria_compat.load_data_safely()

# Check what libraries are available
libraries = valquiria_compat.get_available_libraries()
print("Available libraries:", libraries)
```

### Option 3: Use the Startup Scripts

**Windows users:**
- Double-click `start_analysis.bat`

**All users:**
- Run `python start_analysis.py`
- Or run `jupyter notebook` manually

---

## 📊 Data Loading

### Automatic Data Loading
```python
import valquiria_compat
df = valquiria_compat.load_data_safely()
```

This function will:
1. Try loading from `merged_data.db` first
2. Fall back to CSV files if database unavailable
3. Handle errors gracefully
4. Report loading status

### Manual Data Loading
```python
# Load from database
import sqlite3
import pandas as pd

conn = sqlite3.connect('merged_data.db')
df = pd.read_sql_query("SELECT * FROM merged_data", conn)
conn.close()

# Or load from specific CSV
df = pd.read_csv('T01_Mara.csv')
```

---

## 💓 HRV Analysis

### Using HRV-Analysis Library
```python
import valquiria_compat

if valquiria_compat.HAS_HRV_ANALYSIS:
    from hrvanalysis import (
        get_time_domain_features,
        get_frequency_domain_features,
        get_poincare_plot_features
    )
    
    # Convert heart rate to RR intervals
    # Assuming heart_rate is in bpm
    rr_intervals = 60000 / df['heart_rate [bpm]'].dropna()
    rr_intervals = rr_intervals.astype(int).tolist()
    
    # Calculate HRV metrics
    time_domain = get_time_domain_features(rr_intervals)
    freq_domain = get_frequency_domain_features(rr_intervals) 
    nonlinear = get_poincare_plot_features(rr_intervals)
    
else:
    print("Using alternative HRV library...")
    # Use NeuroKit2 or HeartPy as fallback
```

### Alternative HRV Libraries
```python
# NeuroKit2 approach
if valquiria_compat.HAS_NEUROKIT:
    import neurokit2 as nk
    
    # Process ECG or heart rate data
    hrv_metrics = nk.hrv_time(rr_intervals, sampling_rate=1000)

# HeartPy approach  
if valquiria_compat.HAS_HEARTPY:
    import heartpy as hp
    
    # Process heart rate data
    working_data, measures = hp.process(heart_rate_data, sample_rate=1.0)
```

---

## 📈 Statistical Analysis

### Basic Statistical Tests
```python
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway, pearsonr

# T-test between groups
group1 = df[df['subject'] == 'T01']['heart_rate [bpm]'].dropna()
group2 = df[df['subject'] == 'T02']['heart_rate [bpm]'].dropna()
t_stat, p_value = ttest_ind(group1, group2)
print(f"t-test: t={t_stat:.3f}, p={p_value:.3f}")

# ANOVA across multiple groups
subjects = df['subject'].unique()[:3]
groups = [df[df['subject'] == subj]['heart_rate [bpm]'].dropna() for subj in subjects]
f_stat, f_p = f_oneway(*groups)
print(f"ANOVA: F={f_stat:.2f}, p={f_p:.3f}")

# Correlation analysis
var1 = df['heart_rate [bpm]'].dropna()
var2 = df['breathing_rate [rpm]'].dropna()
# Align lengths
min_len = min(len(var1), len(var2))
corr, p_val = pearsonr(var1[:min_len], var2[:min_len])
print(f"Correlation: r={corr:.3f}, p={p_val:.3f}")
```

### Advanced Statistical Reporting
Following the scientific reporting rules:

```python
from scipy.stats import ttest_ind

# Calculate effect size (Cohen's d)
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*x.var() + (ny-1)*y.var()) / dof)
    return (x.mean() - y.mean()) / pooled_std

# Perform t-test with full reporting
group1 = df[df['subject'] == 'T01']['heart_rate [bpm]'].dropna()
group2 = df[df['subject'] == 'T02']['heart_rate [bpm]'].dropna()
t_stat, p_value = ttest_ind(group1, group2)
effect_size = cohen_d(group1, group2)

# Calculate confidence interval (approximate)
se = np.sqrt(group1.var()/len(group1) + group2.var()/len(group2))
dof = len(group1) + len(group2) - 2
ci_lower = (group1.mean() - group2.mean()) - 1.96 * se
ci_upper = (group1.mean() - group2.mean()) + 1.96 * se

# Report in scientific format
print(f"Heart rate differed significantly between subjects")
print(f"t({dof}) = {t_stat:.2f}, p = {p_value:.3f}, d = {effect_size:.2f}")
print(f"95% CI [{ci_lower:.2f}, {ci_upper:.2f}]")
```

---

## 📊 Visualization

### Standard Plotting
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Configure for high-quality plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Example plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Box plot
sns.boxplot(data=df, x='subject', y='heart_rate [bpm]', ax=axes[0,0])
axes[0,0].set_title('Heart Rate by Subject')

# Time series
subject_data = df[df['subject'] == 'T01'].head(1000)
axes[0,1].plot(subject_data['heart_rate [bpm]'])
axes[0,1].set_title('Heart Rate Time Series - T01')

# Correlation plot
axes[1,0].scatter(df['heart_rate [bpm]'], df['breathing_rate [rpm]'], alpha=0.6)
axes[1,0].set_xlabel('Heart Rate (bpm)')
axes[1,0].set_ylabel('Breathing Rate (rpm)')
axes[1,0].set_title('Heart Rate vs Breathing Rate')

# Distribution
df['heart_rate [bpm]'].hist(bins=50, ax=axes[1,1])
axes[1,1].set_title('Heart Rate Distribution')
axes[1,1].set_xlabel('Heart Rate (bpm)')

plt.tight_layout()
plt.show()
```

---

## 🔍 Troubleshooting

### If you still get import errors:
1. Restart your Jupyter kernel: `Kernel → Restart`
2. Make sure you're using the "Valquiria Space Analog Analysis" kernel
3. Re-run the setup: `python comprehensive_jupyter_fix.py`

### If data doesn't load:
1. Check file paths: `ls -la *.csv *.db`
2. Verify database: `sqlite3 merged_data.db ".tables"`
3. Check permissions on data files

### If HRV analysis fails:
1. Verify astropy compatibility: `python -c "from astropy.stats import LombScargle"`
2. Use alternative libraries: NeuroKit2 or HeartPy
3. Check RR interval format (should be in milliseconds)

### For Windows users:
- Use PowerShell instead of Command Prompt
- Ensure Python is in your PATH
- Run as administrator if permissions issues

---

## 📋 Summary of Available Files

- ✅ `comprehensive_jupyter_fix.py` - Main fix script
- ✅ `valquiria_compat.py` - Compatibility module for notebooks  
- ✅ `start_analysis.py` - Python startup script
- ✅ `start_analysis.bat` - Windows batch starter
- ✅ `FIXED_Analysis_Demo.ipynb` - Working demo notebook
- ✅ `notebook_fix_instructions.md` - This file

---

## 🎉 Your Notebooks Are Now Ready!

All the major issues have been resolved:

1. **Dependencies**: All packages installed with compatible versions
2. **HRV Analysis**: Astropy compatibility patches applied
3. **Data Loading**: Robust loading functions created
4. **Environment**: Jupyter kernel configured properly
5. **Documentation**: Complete instructions provided

You can now run your physiological data analysis with confidence! 🚀

---

*For questions or issues, refer to the original fix guide in `scripts/HRV_ANALYSIS_FIX_GUIDE.md`* 