# Jupyter Notebook Analysis Setup
## Valquiria Space Analog Simulation

This directory contains Jupyter notebooks for analyzing physiological data from the Valquiria Space Analog Simulation.

## 📁 Contents

### Notebooks
- **`Results.ipynb`** - Initial data loading and cleaning (1.7MB)
- **`Results_2.ipynb`** - Comprehensive statistical analysis and modeling (2.9MB)

### Data Files
- **`merged_data.db`** - SQLite database with processed data (99MB)
- **Subject CSV files** - Individual participant data files
  - `T01_Mara.csv` (54MB)
  - `T02_Laura.csv` (19MB)
  - `T03_Nancy.csv` (9.4MB)
  - `T04_Michelle.csv` (6.9MB)
  - `T05_Felicitas.csv` (13MB)
  - `T06_Mara_Selena.csv` (12MB)
  - `T07_Geraldinn.csv` (7.4MB)
  - `T08_Karina.csv` (4.3MB)

### Configuration Files
- **`requirements_jupyter.txt`** - Python dependencies for Jupyter analysis
- **`setup_jupyter_environment.py`** - Setup script for Jupyter environment
- **`start_jupyter.bat`** - Windows batch file to start Jupyter
- **`start_jupyter.ps1`** - PowerShell script to start Jupyter

### Helper Scripts
- **`scripts/load_data.py`** - Data loading utilities
- **`scripts/analysis_utils.py`** - Common analysis functions

## 🚀 Quick Start

### Option 1: Automatic Setup (Recommended)

1. **Run the setup script:**
   ```bash
   python setup_jupyter_environment.py
   ```

2. **Start Jupyter:**
   - **Windows:** Double-click `start_jupyter.bat`
   - **PowerShell:** Run `.\start_jupyter.ps1`
   - **Command line:** `jupyter notebook`

### Option 2: Manual Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements_jupyter.txt
   ```

2. **Create Jupyter kernel:**
   ```bash
   python -m ipykernel install --user --name=valquiria-analysis --display-name="Valquiria Space Analog Analysis"
   ```

3. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

## 📊 Data Overview

The dataset contains physiological measurements from 8 participants during a space analog simulation:

### Key Variables
- **Heart Rate** (bpm) - Cardiac activity
- **Breathing Rate** (rpm) - Respiratory frequency
- **Minute Ventilation** (mL/min) - Respiratory volume
- **Activity** (g) - Physical movement
- **SPO2** (%) - Blood oxygen saturation
- **Systolic Pressure** (mmHg) - Blood pressure
- **Temperature** (°C) - Body temperature
- **Cadence** (spm) - Step frequency

### Data Characteristics
- **Total records:** ~1.5 million
- **Time span:** ~14 days
- **Subjects:** 8 participants
- **Sampling:** Variable frequency across sensors

## 🔬 Analysis Workflow

### 1. Data Loading (`Results.ipynb`)
- Load CSV files from multiple subjects
- Perform data quality assessment
- Clean and preprocess data
- Handle missing values and outliers
- Convert time formats

### 2. Statistical Analysis (`Results_2.ipynb`)
- Descriptive statistics and distribution analysis
- Correlation analysis (Pearson and Spearman)
- Inter-subject variability assessment
- Time series analysis across mission days
- Linear Mixed-Effects Models
- ANOVA and post-hoc testing
- Visualization and reporting

## 🛠️ Helper Functions

### Data Loading
```python
from scripts.load_data import load_csv_data, load_database_data

# Load all CSV files
csv_data = load_csv_data()

# Load from database
df = load_database_data("merged_data.db")
```

### Analysis Utilities
```python
from scripts.analysis_utils import *

# Quick analysis of key variables
quick_analysis(df, ['heart_rate', 'breathing_rate', 'activity'])

# Distribution analysis
analyze_variable_distribution(df, 'heart_rate')

# Correlation analysis
correlation_analysis(df, ['heart_rate', 'breathing_rate', 'activity'])

# Time series analysis
time_series_analysis(df, 'heart_rate')

# Statistical comparison
statistical_comparison(df, 'heart_rate', test_type='kruskal')
```

## 📈 Key Findings

### Data Characteristics
- **Non-normal distributions** in key physiological variables
- **Significant inter-subject variability** in baseline measures
- **Zero-inflation** in activity and cadence data
- **Missing data patterns** vary by sensor type

### Statistical Results
- **Moderate correlations** between activity, heart rate, and ventilation
- **Significant differences** between subjects for core physiological measures
- **Time-dependent patterns** across mission days
- **Complex relationships** requiring subject-specific analysis

## 🔧 Troubleshooting

### Common Issues

1. **Jupyter not starting:**
   - Check Python installation: `python --version`
   - Install dependencies: `pip install -r requirements_jupyter.txt`
   - Check port availability (default: 8888)

2. **Kernel not found:**
   - Run setup script: `python setup_jupyter_environment.py`
   - Or manually create kernel: `python -m ipykernel install --user --name=valquiria-analysis`

3. **Memory issues with large datasets:**
   - Use data sampling for initial exploration
   - Consider using Dask for large-scale analysis
   - Monitor system resources

4. **Database connection errors:**
   - Verify `merged_data.db` exists
   - Check file permissions
   - Ensure SQLite3 is available

### Performance Tips

1. **For large datasets:**
   - Use data sampling for initial exploration
   - Load only required columns
   - Consider chunked processing

2. **For visualization:**
   - Use sample data for complex plots
   - Adjust figure sizes for better performance
   - Save plots to files for later viewing

## 📚 Additional Resources

### Documentation
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/)

### Statistical Analysis
- [Statsmodels Documentation](https://www.statsmodels.org/stable/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Scikit-posthocs Documentation](https://scikit-posthocs.readthedocs.io/)

## 🤝 Contributing

When working with the notebooks:

1. **Create backups** before making major changes
2. **Document** any new analysis methods
3. **Test** code with sample data first
4. **Update** this README with new findings

## 📞 Support

For technical issues or questions about the analysis:
- Check the troubleshooting section above
- Review the existing notebook documentation
- Contact the research team for domain-specific questions

---

**Last updated:** March 2025  
**Project:** Valquiria Space Analog Simulation  
**Analysis:** Physiological Data Processing and Statistical Analysis 