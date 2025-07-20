# T03-T08 Physiological Data Analysis

## Overview

This directory contains comprehensive analysis tools for analyzing physiological data from subjects T03-T08 across SOLs (Space Operations Laboratory days). The analysis includes data cleaning, descriptive statistics, non-parametric statistical tests, and time series analysis.

## Files Included

### 1. `comprehensive_analysis_T03_T08.py`
- **Purpose**: Complete Python script for comprehensive analysis
- **Features**: 
  - Data loading and cleaning
  - Descriptive statistics
  - Non-parametric statistical tests (Kruskal-Wallis, Mann-Whitney U, Friedman)
  - Time series analysis with Spearman correlation
  - Comprehensive visualizations
  - Detailed interpretation of findings

### 2. `T03_T08_Analysis.ipynb`
- **Purpose**: Interactive Jupyter notebook for step-by-step analysis
- **Features**:
  - Well-documented cells with explanations
  - Interactive visualizations
  - Detailed interpretations for each finding
  - Export capabilities for results

## Data Requirements

The analysis expects CSV files for the following subjects:
- `T03_Nancy.csv`
- `T04_Michelle.csv`
- `T05_Felicitas.csv`
- `T06_Mara_Selena.csv`
- `T07_Geraldinn.csv`
- `T08_Karina.csv`

### Expected Data Structure

The CSV files should contain:
- **Time column**: Either `time [s/1000]` or `time` for temporal analysis
- **SOL column**: `Sol` for day-by-day analysis
- **Physiological variables**: Columns with units like `[bpm]`, `[rpm]`, `[%]`, `[g]`, `[ml/min]`, `[mmHg]`, `[C]`

## Analysis Components

### 1. Data Loading and Preparation
- Loads all subject CSV files
- Combines data with subject identifiers
- Handles missing values
- Converts time formats
- Identifies physiological variables

### 2. Data Cleaning
- Missing value analysis
- Temporal feature extraction (date, hour, minute)
- Data type conversions
- Outlier detection (optional)

### 3. Descriptive Analysis
- Subject-wise summary statistics
- Variable-wise descriptive statistics
- Effect size calculations (Cohen's d)
- Data distribution analysis

### 4. Non-Parametric Statistical Tests

#### Kruskal-Wallis Test
- **Purpose**: Compare physiological variables between subjects
- **Interpretation**: Identifies variables with significant individual differences
- **Clinical Significance**: Determines if personalized vs. standardized approaches are needed

#### Mann-Whitney U Test
- **Purpose**: Pairwise comparisons between subjects
- **Interpretation**: Identifies specific subject pairs with significant differences
- **Clinical Significance**: Helps understand which subjects need different monitoring protocols

#### Spearman Correlation
- **Purpose**: Detect temporal trends across SOLs
- **Interpretation**: Identifies increasing/decreasing trends over time
- **Clinical Significance**: Detects adaptation, improvement, or decline patterns

#### Friedman Test
- **Purpose**: Within-subject changes across SOLs
- **Interpretation**: Identifies subjects with significant physiological changes over time
- **Clinical Significance**: Detects adaptation patterns and stability

### 5. Time Series Analysis
- Daily (SOL-based) statistics calculation
- Temporal trend analysis
- Correlation analysis with time
- Pattern identification across subjects

### 6. Visualizations
- Box plots for between-subject comparisons
- Time series plots across SOLs
- Correlation heatmaps
- Hourly pattern analysis
- Distribution analysis (violin plots, histograms)
- Statistical summary plots

## Usage Instructions

### Option 1: Python Script
```python
# Run the complete analysis
python comprehensive_analysis_T03_T08.py
```

### Option 2: Jupyter Notebook
```bash
# Start Jupyter Lab/Notebook
jupyter lab T03_T08_Analysis.ipynb
```

Then run cells sequentially for step-by-step analysis.

## Key Features

### 1. Comprehensive Statistical Analysis
- **Non-parametric tests**: Appropriate for physiological data
- **Multiple comparison approaches**: Between-subject, within-subject, temporal
- **Effect size calculations**: Clinical significance assessment
- **Trend analysis**: Temporal pattern detection

### 2. Detailed Interpretations
- **Statistical significance**: What the p-values mean
- **Clinical implications**: What the findings suggest for monitoring
- **Practical recommendations**: Next steps for implementation
- **Visual interpretations**: What the plots show

### 3. Robust Data Handling
- **Missing data management**: Appropriate handling of gaps
- **Flexible data structures**: Adapts to different column formats
- **Error handling**: Graceful handling of data issues
- **Data validation**: Ensures analysis reliability

## Interpretation Guidelines

### Statistical Significance Levels
- **p < 0.001**: Highly significant (marked with ***)
- **p < 0.01**: Very significant (marked with **)
- **p < 0.05**: Significant (marked with *)
- **p ≥ 0.05**: Not significant (marked with ns)

### Effect Size Interpretation
- **Cohen's d < 0.2**: Negligible effect
- **Cohen's d 0.2-0.5**: Small effect
- **Cohen's d 0.5-0.8**: Medium effect
- **Cohen's d > 0.8**: Large effect

### Correlation Strength
- **|r| > 0.7**: Strong correlation
- **|r| 0.5-0.7**: Moderate correlation
- **|r| 0.3-0.5**: Weak correlation
- **|r| < 0.3**: Very weak correlation

## Clinical Implications

### Individual Differences
- **Significant differences**: Personalized monitoring recommended
- **Similar responses**: Standardized protocols feasible
- **Mixed patterns**: Hybrid approach needed

### Temporal Trends
- **Increasing trends**: May indicate adaptation or improvement
- **Decreasing trends**: May indicate fatigue or decline
- **Stable patterns**: Good adaptation to conditions

### Monitoring Recommendations
1. **Continuous monitoring**: For variables showing significant differences
2. **Trend monitoring**: For variables showing temporal patterns
3. **Threshold setting**: Based on individual baseline ranges
4. **Early intervention**: For declining trends
5. **Adaptive protocols**: For changing patterns

## Output Files

The analysis generates:
- **Visualization files**: PNG files with comprehensive plots
- **Statistical results**: CSV files with test results
- **Summary reports**: Detailed interpretation documents
- **Export data**: Processed data for further analysis

## Requirements

### Python Libraries
```python
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.7.0
```

### System Requirements
- Python 3.7+
- 4GB+ RAM (for large datasets)
- Sufficient disk space for visualizations

## Troubleshooting

### Common Issues
1. **File not found**: Check CSV file paths and names
2. **Missing columns**: Verify expected column names
3. **Memory issues**: Reduce data size or increase RAM
4. **Visualization errors**: Check matplotlib backend settings

### Data Quality Issues
1. **Missing values**: Analysis handles gaps automatically
2. **Outliers**: Review data before analysis
3. **Inconsistent formats**: Ensure consistent time formats
4. **Duplicate records**: Check for data duplication

## Support and Documentation

For questions or issues:
1. Check the detailed comments in the code
2. Review the interpretation sections in the notebook
3. Consult the statistical methodology documentation
4. Contact the Valquiria Research Team

## Version History

- **v1.0**: Initial comprehensive analysis implementation
- **Date**: January 2025
- **Author**: Valquiria Research Team

## License

This analysis is part of the Valquiria Research Project. Use in accordance with project guidelines and ethical requirements. 