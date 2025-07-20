# Hexoskin WAV File Analyzer - User Manual

**Version 0.0.3**  
**Created by Diego Malpica MD**  
**Aerospace and Physiology Research**

---

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Getting Started](#getting-started)
5. [User Interface Guide](#user-interface-guide)
6. [Data Loading and Processing](#data-loading-and-processing)
7. [Data Visualization](#data-visualization)
8. [Statistical Analysis](#statistical-analysis)
9. [Data Comparison](#data-comparison)
10. [Export Features](#export-features)
11. [Command Line Interface](#command-line-interface)
12. [Troubleshooting](#troubleshooting)
13. [Best Practices](#best-practices)
14. [Appendices](#appendices)

---

## Introduction

### What is the Hexoskin WAV File Analyzer?

The Hexoskin WAV File Analyzer is a comprehensive Python application designed for researchers and clinicians working with physiological data from Hexoskin smart garments. Hexoskin devices collect various health metrics including:

- **ECG (Electrocardiogram)**: Heart electrical activity
- **Respiration**: Breathing patterns and rate
- **Accelerometry**: Movement and activity data
- **Heart Rate Variability**: RR intervals and HRV metrics

The application provides tools for:
- Loading and validating Hexoskin WAV files
- Processing and filtering physiological data
- Comprehensive statistical analysis
- Data visualization and comparison
- Export capabilities for further analysis

### Key Features

- **Robust Data Loading**: Handles various WAV formats with error checking
- **Advanced Statistical Analysis**: 15+ statistical tests with effect sizes
- **Interactive GUI**: User-friendly interface with tabbed navigation
- **Data Comparison**: Compare up to 15 datasets simultaneously
- **Export Capabilities**: Save data, plots, and statistical results
- **Time Zone Support**: Automatic handling of real timestamps
- **Memory Management**: Efficient processing of large files

---

## System Requirements

### Minimum Requirements

- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.6 or higher
- **Memory**: 4 GB RAM minimum (8 GB recommended for large datasets)
- **Storage**: 1 GB free space for installation and data processing
- **Display**: 1024x768 resolution minimum (1920x1080 recommended)

### Recommended Requirements

- **Python**: Version 3.8 or higher
- **Memory**: 16 GB RAM for optimal performance
- **Storage**: 5 GB free space for large dataset processing
- **Display**: 1920x1080 or higher resolution

---

## Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

The required packages are:
- `numpy>=1.20.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `matplotlib>=3.5.0` - Plotting and visualization
- `scipy>=1.7.0` - Statistical analysis
- `seaborn>=0.11.0` - Advanced plotting
- `openpyxl>=3.0.0` - Excel export support

### Step 2: Install Tkinter (if not already installed)

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-tk
```

**macOS (using Homebrew):**
```bash
brew install python-tk
```

**Windows:**
Tkinter is included with standard Python installations.

### Step 3: Download the Application

Download the following files:
- `hexoskin_wav_loader.py` - Main application
- `hexoskin_wav_example.py` - Example script
- `run.py` - Launcher script
- `requirements.txt` - Dependencies

### Step 4: Verify Installation

```bash
python run.py
```

If the GUI opens successfully, the installation is complete.

---

## Getting Started

### First Launch

1. **Run the Application**:
   ```bash
   python run.py
   ```

2. **Familiarize with the Interface**:
   - The main window contains three tabs: Plot, Statistics, and Comparison
   - The left panel shows file management controls
   - The right panel displays data visualization and analysis

3. **Load Your First File**:
   - Click "Load WAV File" to select a single file
   - Or use "Batch Load WAV Files" for multiple files
   - Supported formats: `.wav` files from Hexoskin devices

### Understanding Your Data

Hexoskin WAV files contain:
- **Timestamps**: Time points from the start of recording
- **Values**: Physiological measurements (varies by sensor type)
- **Metadata**: Information about recording conditions
- **Real Timestamps**: Actual date/time if `info.json` is present

---

## User Interface Guide

### Main Window Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ Menu Bar: File, Edit, View, Help                                │
├─────────────────┬───────────────────────────────────────────────┤
│ File Management │ Main Content Area                             │
│                 │ ┌─────────────────────────────────────────────┤
│ • Load Files    │ │ Plot Tab                                    │
│ • File List     │ │ • Data visualization                        │
│ • Metadata      │ │ • Customization controls                    │
│ • Processing    │ │ • View controls                             │
│                 │ ├─────────────────────────────────────────────┤
│                 │ │ Statistics Tab                              │
│                 │ │ • Descriptive statistics                    │
│                 │ │ • Normality tests                           │
│                 │ │ • Distribution analysis                     │
│                 │ ├─────────────────────────────────────────────┤
│                 │ │ Comparison Tab                              │
│                 │ │ • Multi-file comparison                     │
│                 │ │ • Statistical tests                         │
│                 │ │ • Effect size analysis                      │
└─────────────────┴─────────────────────────────────────────────────┘
```

### Tab Navigation

#### 1. Plot Tab
- **Primary Function**: Data visualization and basic analysis
- **Key Features**:
  - Interactive plots with zoom and pan
  - Time unit conversion (seconds, minutes, hours, days)
  - Customizable colors, line width, and titles
  - Export plots as high-quality PNG images

#### 2. Statistics Tab
- **Primary Function**: Comprehensive statistical analysis
- **Key Features**:
  - Descriptive statistics with 20+ metrics
  - 5 normality tests with interpretations
  - QQ plots and distribution visualization
  - Statistical recommendations

#### 3. Comparison Tab
- **Primary Function**: Multi-dataset comparison
- **Key Features**:
  - Compare up to 15 datasets simultaneously
  - Automatic test selection based on data characteristics
  - Post-hoc analysis with multiple correction methods
  - Effect size calculations and interpretations

---

## Data Loading and Processing

### Loading Single Files

1. **Click "Load WAV File"**
2. **Select your Hexoskin WAV file**
3. **Review the metadata** displayed in the left panel
4. **Check the automatic plot** generated in the Plot tab

### Batch Loading

1. **Click "Batch Load WAV Files"**
2. **Select multiple files** (Ctrl+click or Shift+click)
3. **Files appear in the list** with automatic processing
4. **Select any file** from the list to view its data

### Data Validation

The application automatically validates:
- ✅ **File Format**: Confirms proper WAV file structure
- ✅ **Sample Rate**: Ensures valid sampling frequency
- ✅ **Data Integrity**: Checks for corrupted or missing data
- ✅ **Memory Requirements**: Estimates memory usage for large files
- ✅ **Timestamp Consistency**: Validates temporal data

### Processing Options

#### Resampling
- **Purpose**: Change the sampling frequency of your data
- **Usage**: Enter new frequency in Hz and click "Resample"
- **Recommended**: Use 100 Hz for ECG, 64 Hz for respiration

#### Filtering
- **Lowpass Filter**: Remove high-frequency noise
- **Highpass Filter**: Remove low-frequency drift
- **Bandpass Filter**: Keep only specific frequency range
- **Example**: For ECG, use 0.5-40 Hz bandpass filter

#### Normalization
- **Min-Max**: Scales data to range [0, 1]
- **Z-Score**: Standardizes to mean=0, std=1
- **Robust**: Uses median and IQR for outlier resistance

---

## Data Visualization

### Plot Customization

#### Color Selection
- **Blue**: Default color for single datasets
- **Red**: Alternative color option
- **Green**: For comparison highlights
- **Custom**: Use hex codes for specific colors

#### Line Properties
- **Width**: Adjust from 0.5 to 3.0 pixels
- **Alpha**: Control transparency (0.1 to 1.0)
- **Style**: Solid, dashed, or dotted lines

#### Time Units
- **Seconds**: Default unit, shows raw timestamps
- **Minutes**: Divides timestamps by 60
- **Hours**: Divides timestamps by 3600
- **Days**: Shows calendar dates if info.json available

### View Controls

#### Zoom Operations
- **Zoom In**: Magnify plot by 20% around center
- **Zoom Out**: Expand view by 25% for more context
- **Fit to Window**: Auto-scale to show all data points
- **Reset View**: Return to original zoom level

#### Navigation
- **Mouse Wheel**: Zoom in/out at cursor position
- **Click and Drag**: Pan around the plot
- **Double Click**: Reset to fit window

### Time Zone Handling

When `info.json` is present:
- **Absolute Timestamps**: Shows actual recording time
- **Date Display**: Calendar dates on x-axis
- **Time Zone**: Automatic conversion to local time
- **Duration**: Accurate recording duration

---

## Statistical Analysis

### Descriptive Statistics

#### Basic Metrics
- **Mean**: Average value
- **Median**: Middle value (50th percentile)
- **Standard Deviation**: Measure of spread
- **Min/Max**: Extreme values
- **Count**: Number of data points

#### Advanced Metrics
- **Skewness**: Asymmetry of distribution
- **Kurtosis**: Tail heaviness
- **Coefficient of Variation**: Relative variability
- **Interquartile Range**: 25th to 75th percentile spread
- **Median Absolute Deviation**: Robust measure of spread

#### Percentiles
Complete percentile analysis:
- 1st, 5th, 10th, 20th, 30th, 40th percentiles
- 60th, 70th, 80th, 90th, 95th, 99th percentiles
- Quartiles (25th, 50th, 75th)

### Normality Testing

#### Available Tests
1. **Shapiro-Wilk Test**: Best for sample sizes < 5000
2. **D'Agostino's K² Test**: Tests skewness and kurtosis
3. **Kolmogorov-Smirnov Test**: Compares to normal distribution
4. **Anderson-Darling Test**: Sensitive to distribution tails
5. **Jarque-Bera Test**: Based on skewness and kurtosis

#### Interpretation
- **P-value > 0.05**: Data appears normally distributed
- **P-value ≤ 0.05**: Data deviates from normality
- **Overall Assessment**: Combines all test results
- **Recommendations**: Suggests appropriate analysis methods

### Distribution Analysis

#### QQ Plots
- **Purpose**: Visual assessment of normality
- **Interpretation**: Points on diagonal line indicate normality
- **Deviations**: Curves suggest skewness or kurtosis

#### Histograms
- **Overlaid Normal Curve**: Compare data to normal distribution
- **Bin Selection**: Automatic optimal binning
- **Density Estimation**: Smooth curve fitting

---

## Data Comparison

### Two-Sample Comparisons

#### Parametric Tests
- **Independent t-test**: For normal data with equal variances
- **Welch's t-test**: For normal data with unequal variances
- **Paired t-test**: For matched samples

#### Non-parametric Tests
- **Mann-Whitney U**: For independent samples
- **Wilcoxon Signed-Rank**: For paired samples
- **Kolmogorov-Smirnov**: For distribution comparisons

### Multiple Sample Comparisons

#### Parametric Tests
- **One-way ANOVA**: Normal data, equal variances
- **Welch's ANOVA**: Normal data, unequal variances
- **Repeated Measures ANOVA**: Paired normal data

#### Non-parametric Tests
- **Kruskal-Wallis**: Independent samples
- **Friedman Test**: Repeated measures
- **Aligned Ranks Transform**: Factorial designs

### Post-hoc Analysis

#### Correction Methods
- **Bonferroni**: Conservative, controls family-wise error
- **Benjamini-Hochberg**: Less conservative, controls false discovery rate
- **Holm-Bonferroni**: Step-down procedure

#### Effect Size Calculations
- **Cohen's d**: Standardized mean difference
- **Eta-squared**: Proportion of variance explained
- **Common Language Effect Size**: Probability of superiority

### Test Selection Assistant

The application includes an intelligent test selection system:

1. **Data Assessment**: Analyzes normality, variance, and sample sizes
2. **Automatic Recommendations**: Suggests appropriate tests
3. **Justification**: Explains why specific tests are recommended
4. **Alternative Options**: Provides backup test suggestions

---

## Export Features

### Data Export

#### CSV Format
- **Processed Data**: Timestamps and values after filtering/resampling
- **Original Data**: Raw data as loaded from WAV file
- **Metadata**: Recording information and processing history
- **Usage**: Compatible with Excel, R, Python, and other analysis tools

#### Excel Format
- **Multiple Sheets**: Separate sheets for data, metadata, and statistics
- **Formatting**: Professional appearance with headers
- **Charts**: Embedded plots and graphs
- **Requires**: openpyxl package

### Plot Export

#### PNG Format
- **High Resolution**: 300 DPI for publication quality
- **Customizable Size**: Adjust dimensions for specific needs
- **Transparent Background**: Optional for overlay applications
- **File Naming**: Automatic naming based on sensor and date

#### PDF Format
- **Vector Graphics**: Scalable without quality loss
- **Multi-page**: Multiple plots in single document
- **Annotations**: Metadata and analysis notes included

### Statistical Results Export

#### Text Format
- **Comprehensive Report**: All statistical tests and interpretations
- **Formatted Output**: Professional scientific reporting style
- **Recommendations**: Analysis suggestions and conclusions
- **Timestamped**: Analysis date and software version

#### CSV Format
- **Structured Data**: Tests, statistics, and p-values in columns
- **Import Ready**: Easy import into other statistical software
- **Metadata**: Analysis parameters and settings

---

## Command Line Interface

### Basic Usage

```bash
python hexoskin_wav_example.py <path_to_wav_file>
```

### Example Workflow

```bash
# Process a single ECG file
python hexoskin_wav_example.py ECG_I.wav

# This will:
# 1. Load the WAV file
# 2. Apply a 10 Hz lowpass filter
# 3. Resample to 100 Hz
# 4. Generate a plot
# 5. Save processed data to CSV
# 6. Display the results
```

### Batch Processing

```python
#!/usr/bin/env python3
"""
Batch processing example for multiple Hexoskin files
"""

import os
import glob
from hexoskin_wav_loader import HexoskinWavLoader

def batch_process(directory, output_dir):
    """Process all WAV files in a directory"""
    
    # Find all WAV files
    wav_files = glob.glob(os.path.join(directory, "*.wav"))
    
    for wav_file in wav_files:
        print(f"Processing: {wav_file}")
        
        # Create loader
        loader = HexoskinWavLoader()
        
        # Load file
        if loader.load_wav_file(wav_file):
            # Apply processing
            loader.filter_data(lowcut=0.5, highcut=40)  # ECG filter
            loader.resample_data(100)  # 100 Hz
            
            # Save results
            base_name = os.path.splitext(os.path.basename(wav_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}_processed.csv")
            loader.save_to_csv(output_file)
            
            print(f"Saved: {output_file}")
        else:
            print(f"Failed to load: {wav_file}")

if __name__ == "__main__":
    batch_process("./data", "./output")
```

### Automated Analysis

```python
#!/usr/bin/env python3
"""
Automated statistical analysis example
"""

from hexoskin_wav_loader import HexoskinWavLoader

def analyze_file(file_path):
    """Perform comprehensive analysis on a single file"""
    
    loader = HexoskinWavLoader()
    
    if loader.load_wav_file(file_path):
        # Get descriptive statistics
        stats = loader.get_descriptive_stats()
        
        # Test normality
        normality = loader.test_normality()
        
        # Print results
        print(f"File: {file_path}")
        print(f"Mean: {stats['basic']['mean']:.2f}")
        print(f"Std: {stats['basic']['std']:.2f}")
        print(f"Normal: {normality['overall_assessment']['assessment']}")
        
        # Save statistical report
        with open(f"{file_path}_stats.txt", 'w') as f:
            f.write(f"Statistical Analysis Report\n")
            f.write(f"File: {file_path}\n")
            f.write(f"Analysis Date: {datetime.now()}\n\n")
            
            # Write basic statistics
            for key, value in stats['basic'].items():
                f.write(f"{key}: {value}\n")
    
    else:
        print(f"Failed to load: {file_path}")

if __name__ == "__main__":
    analyze_file("ECG_I.wav")
```

---

## Troubleshooting

### Common Issues

#### 1. File Loading Problems

**Error: "Invalid WAV file format"**
- **Cause**: File is not a valid WAV file or is corrupted
- **Solution**: 
  - Verify file is from Hexoskin device
  - Try re-exporting from Hexoskin platform
  - Check file size (should be > 44 bytes)

**Error: "File does not exist"**
- **Cause**: Incorrect file path or file moved
- **Solution**: 
  - Verify file path is correct
  - Check file permissions
  - Ensure file hasn't been moved or deleted

#### 2. Memory Issues

**Error: "Insufficient memory to load file"**
- **Cause**: File too large for available memory
- **Solution**: 
  - Close other applications
  - Increase virtual memory
  - Use chunked processing for very large files

#### 3. Statistical Analysis Issues

**Error: "Error in normality test"**
- **Cause**: Data contains invalid values or is too small
- **Solution**: 
  - Check for NaN or infinite values
  - Ensure minimum sample size (n ≥ 3)
  - Review data preprocessing steps

#### 4. GUI Issues

**Error: "Tkinter not found"**
- **Cause**: Tkinter not installed or not in PATH
- **Solution**: 
  - Install tkinter for your Python version
  - On Ubuntu: `sudo apt-get install python3-tk`
  - On macOS: `brew install python-tk`

### Performance Optimization

#### For Large Files
1. **Increase Memory Limit**: 
   ```python
   loader = HexoskinWavLoader()
   loader.max_memory_mb = 1000  # Increase to 1GB
   ```

2. **Use Chunked Processing**: Process files in smaller segments

3. **Optimize Filters**: Use lower filter orders for faster processing

#### For Multiple Files
1. **Batch Processing**: Use command-line interface for automation
2. **Parallel Processing**: Process multiple files simultaneously
3. **Output Management**: Clean up temporary files regularly

### Getting Help

#### Log Files
The application generates detailed logs:
- **Location**: Same directory as the application
- **Format**: `hexoskin_analyzer.log`
- **Content**: Timestamps, errors, and processing information

#### Error Reporting
When reporting issues, include:
- **System Information**: OS, Python version, package versions
- **Error Messages**: Complete error text and traceback
- **Sample Files**: Minimal example that reproduces the issue
- **Log Files**: Relevant log entries

#### Contact Information
- **Developer**: Diego Malpica, MD
- **Email**: dlmalpicah@unal.edu.co
- **Project**: Valquiria Space Analog Simulation

---

## Best Practices

### Data Management

#### File Organization
```
Project/
├── raw_data/
│   ├── ECG_I.wav
│   ├── ECG_II.wav
│   ├── Respiration.wav
│   └── info.json
├── processed_data/
│   ├── ECG_I_processed.csv
│   └── ECG_II_processed.csv
├── plots/
│   ├── ECG_I_plot.png
│   └── comparison_plot.png
└── statistics/
    ├── ECG_I_stats.txt
    └── comparison_results.csv
```

#### Naming Conventions
- **Raw Files**: Keep original Hexoskin names
- **Processed Files**: Add `_processed` suffix
- **Plots**: Add `_plot` suffix
- **Statistics**: Add `_stats` suffix

### Analysis Workflow

#### 1. Data Exploration
```python
# Load and explore data
loader = HexoskinWavLoader()
loader.load_wav_file("ECG_I.wav")

# Check metadata
metadata = loader.get_metadata()
print(f"Duration: {metadata['duration']:.2f} seconds")
print(f"Sample Rate: {metadata['sample_rate']} Hz")

# Basic statistics
stats = loader.get_descriptive_stats()
print(f"Mean: {stats['basic']['mean']:.2f}")
print(f"Std: {stats['basic']['std']:.2f}")
```

#### 2. Data Cleaning
```python
# Apply appropriate filters
if "ECG" in filename:
    loader.filter_data(lowcut=0.5, highcut=40)  # ECG filter
elif "Respiration" in filename:
    loader.filter_data(lowcut=0.1, highcut=5)   # Respiration filter

# Resample if needed
loader.resample_data(100)  # Standard 100 Hz
```

#### 3. Statistical Analysis
```python
# Test normality
normality = loader.test_normality()
is_normal = normality['overall_assessment']['is_normal']

# Choose appropriate tests
if is_normal:
    # Use parametric tests
    pass
else:
    # Use non-parametric tests
    pass
```

### Quality Control

#### Data Validation Checklist
- [ ] **File Format**: Valid WAV file with correct header
- [ ] **Sample Rate**: Reasonable for sensor type (64-512 Hz)
- [ ] **Duration**: Matches expected recording time
- [ ] **Data Range**: Values within expected physiological range
- [ ] **Continuity**: No large gaps or discontinuities
- [ ] **Artifacts**: Check for obvious noise or artifacts

#### Statistical Analysis Checklist
- [ ] **Sample Size**: Adequate for chosen statistical tests
- [ ] **Normality**: Appropriate test selection based on distribution
- [ ] **Assumptions**: All test assumptions are met
- [ ] **Effect Sizes**: Calculated and interpreted correctly
- [ ] **Multiple Comparisons**: Appropriate corrections applied

---

## Appendices

### Appendix A: File Format Specifications

#### Hexoskin WAV Format
```
Byte Range | Description
-----------|------------------------------------------
0-3        | "RIFF" header
4-7        | File size (little-endian)
8-11       | "WAVE" format
12-15      | "fmt " subchunk
16-19      | Subchunk size (16 for PCM)
20-21      | Audio format (1 for PCM)
22-23      | Number of channels
24-27      | Sample rate (Hz)
28-31      | Byte rate
32-33      | Block align
34-35      | Bits per sample
36-39      | "data" subchunk
40-43      | Data size
44+        | Audio data (samples)
```

#### Info.json Format
```json
{
  "timestamp": 1640995200,
  "start_date": "2022-01-01 00:00:00",
  "user": "user123",
  "devices": ["hexoskin_device_001"],
  "sensors": ["ECG_I", "ECG_II", "Respiration"],
  "duration": 3600,
  "sample_rates": {
    "ECG_I": 256,
    "ECG_II": 256,
    "Respiration": 128
  }
}
```

### Appendix B: Statistical Test Reference

#### Normality Tests
| Test | Best For | Sample Size | Strengths | Weaknesses |
|------|----------|-------------|-----------|------------|
| Shapiro-Wilk | General use | < 5000 | Most powerful | Limited sample size |
| D'Agostino K² | Skewness/Kurtosis | > 20 | Fast | Less sensitive |
| Kolmogorov-Smirnov | Distribution comparison | Any | Versatile | Conservative |
| Anderson-Darling | Tail sensitivity | > 50 | Sensitive to extremes | Complex interpretation |
| Jarque-Bera | Economic data | > 30 | Simple | Assumes specific distribution |

#### Comparison Tests
| Test | Data Type | Samples | Assumptions | Use Case |
|------|-----------|---------|-------------|----------|
| t-test | Parametric | 2 | Normal, equal variance | Basic comparison |
| Welch's t-test | Parametric | 2 | Normal, unequal variance | Robust comparison |
| Mann-Whitney U | Non-parametric | 2 | Independent samples | Robust alternative |
| Wilcoxon | Non-parametric | 2 | Paired samples | Matched pairs |
| ANOVA | Parametric | 3+ | Normal, equal variance | Multiple groups |
| Kruskal-Wallis | Non-parametric | 3+ | Independent samples | Robust multiple groups |

### Appendix C: Physiological Data Ranges

#### Normal Ranges for Hexoskin Sensors
| Sensor | Normal Range | Units | Sampling Rate |
|--------|--------------|--------|---------------|
| ECG_I | -2000 to 2000 | μV | 256 Hz |
| ECG_II | -2000 to 2000 | μV | 256 Hz |
| Respiration | -500 to 500 | Arbitrary | 128 Hz |
| Heart Rate | 40 to 200 | BPM | 1 Hz |
| Activity | 0 to 4 | G-force | 64 Hz |

#### Filtering Recommendations
| Signal Type | Lowpass (Hz) | Highpass (Hz) | Purpose |
|-------------|--------------|---------------|---------|
| ECG | 0.5 | 40 | Remove baseline drift and noise |
| Respiration | 0.1 | 5 | Isolate breathing signal |
| Heart Rate | N/A | 0.1 | Remove very slow trends |
| Activity | 0.1 | 20 | Remove sensor drift |

### Appendix D: Troubleshooting Matrix

| Symptom | Possible Causes | Solutions |
|---------|-----------------|-----------|
| File won't load | Invalid format, corrupted file | Check file integrity, re-export |
| Memory error | File too large, insufficient RAM | Increase memory limit, use chunked processing |
| Statistical error | Invalid data, too few samples | Clean data, increase sample size |
| GUI freezing | Large dataset, processing intensive | Use progress bars, background processing |
| Export failed | Permissions, disk space | Check write permissions, free up space |
| Plot not updating | Selection issues, cached data | Refresh view, reload data |

---

## Conclusion

The Hexoskin WAV File Analyzer provides a comprehensive solution for physiological data analysis. This manual covers all aspects of the application, from basic usage to advanced statistical analysis. For additional support or to report issues, please contact the development team.

**Remember**: Always validate your data and choose appropriate statistical methods based on your research questions and data characteristics.

---

*This manual is part of the Valquiria Space Analog Simulation project and is continuously updated based on user feedback and software improvements.*