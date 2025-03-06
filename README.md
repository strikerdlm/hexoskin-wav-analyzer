# Hexoskin WAV File Analyzer

**Version 0.0.1**

**Developed by Diego Malpica MD**  
Aerospace and Physiology Research, March 2025  
For the Valquiria Space Analog Simulation

**Special Thanks To:**  
- Centro de Telemedicina de Colombia
- Women AeroSTEAM

## Overview

This project provides tools for loading, analyzing, and visualizing Hexoskin WAV files containing physiological data. Hexoskin smart garments collect various health metrics (ECG, respiration, etc.) which can be exported as WAV files. This application helps researchers and clinicians work with these files.

## Features

- Load and decode Hexoskin WAV files (ECG, breathing, etc.)
- Extract timestamp and value data
- Automatic detection of real timestamps from info.json files
- Apply filters to smooth or process the data
- Resample data to different frequencies
- Plot the data with customizable visualization options
- Time unit conversion (seconds, minutes, hours, days) with accurate timestamps
- View controls (fit to window, zoom in/out)
- Automatic statistical analysis when selecting files
- Save processed data to CSV format
- Export graphs as high-quality PNG images
- Perform statistical analysis including descriptive statistics and normality tests
- Compare datasets with non-parametric statistical tests
- Align and normalize datasets for better comparison
- GUI for interactive data analysis with a tabbed interface
- Command-line interface for batch processing

## Installation

### Requirements

- Python 3.6+
- numpy
- pandas
- matplotlib
- scipy
- tkinter (for GUI)

Install the dependencies using pip:

```bash
pip install numpy pandas matplotlib scipy
```

Tkinter typically comes with Python installations. If it's missing, you can install it with:

```bash
# For Ubuntu/Debian
sudo apt-get install python3-tk

# For macOS (using Homebrew)
brew install python-tk

# For Windows
# Tkinter is included with standard Python installation
```

### Setup

1. Clone this repository or download the files:
   - `hexoskin_wav_loader.py`: Core library for loading WAV files
   - `hexoskin_wav_example.py`: Example script to process WAV files
   - `README.md`: This documentation file

## Usage

### GUI Interface

Run the GUI application:

```bash
python hexoskin_wav_loader.py
```

Using the GUI:
1. Click "Load WAV File" to select a single file or "Batch Load WAV Files" for multiple files
2. Select a file from the list to view its data and metadata (the plot and statistics will update automatically)
3. Use the processing tools to resample or filter the data
4. Customize plot appearance (color, line width, title)
5. Change time units (seconds, minutes, hours, days) for better visualization
6. Use view controls to fit data to window, zoom in/out, or reset view
7. Export the graph as PNG or save the data as CSV
8. Compare multiple files in the Comparison tab using non-parametric statistical tests

### Command Line Interface

For batch processing or scripting, use the example script:

```bash
python hexoskin_wav_example.py path/to/your/file.wav
```

This will:
1. Load the specified WAV file
2. Apply a lowpass filter (10 Hz)
3. Resample to 100 Hz
4. Plot the data
5. Save the processed data to a CSV file

### Using as a Library

You can import `HexoskinWavLoader` in your own Python scripts:

```python
from hexoskin_wav_loader import HexoskinWavLoader

# Create loader instance
loader = HexoskinWavLoader()

# Load a WAV file
loader.load_wav_file("path/to/ECG_I.wav")

# Get data as a pandas DataFrame
data = loader.get_data()

# Process the data
loader.filter_data(lowcut=0.5, highcut=20)  # Apply bandpass filter
loader.resample_data(100)  # Resample to 100 Hz

# Perform statistical analysis
stats = loader.get_descriptive_stats()
normality = loader.test_normality()

# Save to CSV
loader.save_to_csv("processed_data.csv")
```

## Statistical Analysis

The application includes statistical analysis capabilities:

1. **Descriptive Statistics**
   - Mean, median, standard deviation
   - Min/max values
   - 25th and 75th percentiles

2. **Normality Tests**
   - Shapiro-Wilk test
   - D'Agostino's K^2 test
   - Kolmogorov-Smirnov test
   - Skewness and kurtosis calculation
   - Overall assessment of data distribution

3. **Non-parametric Comparison Tests**
   - Mann-Whitney U test for independent samples
   - Wilcoxon signed-rank test for paired samples
   - Kolmogorov-Smirnov test for distribution comparison
   - Effect size calculation and interpretation

## Advanced Features

### Data Alignment

The application can align two datasets to ensure they have the same time points and sampling rate, making comparison more valid:

```python
aligned_data1, aligned_data2 = HexoskinWavLoader.align_datasets(dataset1, dataset2, target_hz=100)
```

### Data Normalization

To compare datasets with different scales, the application provides normalization methods:

```python
normalized_data = HexoskinWavLoader.normalize_dataset(dataset, method='min_max')
```

Supported methods:
- min_max: Scales data to range [0, 1]
- z_score: Standardizes data to mean=0, std=1
- robust: Uses median and IQR for robust scaling

### Time Unit Conversion

The application allows you to visualize time-series data in different time units:

- Seconds (default): Raw timestamps
- Minutes: Timestamps divided by 60
- Hours: Timestamps divided by 3600
- Days: Timestamps divided by 86400 or actual calendar dates when using info.json

When an info.json file is available in the same directory as the WAV file, the application automatically detects and uses real timestamps, converting the relative timestamps in the WAV file to absolute dates and times. This is particularly useful for long recordings or when comparing data from different recording sessions.

### Real Timestamp Detection

The application automatically searches for and uses timestamp information from the info.json file that accompanies Hexoskin recordings:

- Absolute start time of recording (UNIX timestamp)
- Start date in human-readable format
- Device information

This allows for more accurate visualization and comparison of data across multiple recording sessions, with plots showing actual dates and times instead of just relative timestamps.

### View Controls

Several view controls are available to help navigate and visualize the data:

- **Fit to Window**: Automatically adjusts the plot axes to show all data points
- **Zoom In**: Magnifies the plot by 20% around the center
- **Zoom Out**: Expands the view by 25% to show more context
- **Reset View**: Returns to the original view that shows all data

These controls make it easier to explore details in the data while maintaining the ability to see the overall patterns.

## File Format

Hexoskin WAV files follow this format:
- Standard WAV header (44 bytes)
- Binary data in short integer format
- Sampling rates vary by sensor type (typically 128 Hz for ECG, 64 Hz for respiration)

The `HexoskinWavLoader` class automatically:
1. Extracts the sampling rate from the WAV header
2. Calculates timestamps based on frame count and sampling rate
3. Converts the binary data to numeric values

## Troubleshooting

Common issues:

- **"Error loading WAV file"**: Make sure the file is a valid Hexoskin WAV file
- **Tkinter not found**: Install tkinter for your Python version
- **Missing dependencies**: Make sure all required packages are installed

## Further Resources

- [Hexoskin Developer Documentation](https://hexoskin.com)
- [WAV File Format Specification](https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html)

## License

This project is provided as open-source software. Feel free to modify and distribute it according to your needs.

## About the Project

This application was developed for the Valquiria Space Analog Simulation, which studies physiological responses in simulated space environments. The tools provided here help analyze and interpret physiological data collected from participants wearing Hexoskin garments.

For more information about the Valquiria Space Analog Simulation, please contact Diego Malpica MD at dlmalpicah@unal.edu.co

## Contributing

Contributions to the Hexoskin WAV Analyzer are welcome! If you'd like to contribute, please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open-source and available for academic and research use.

## Contact

Diego Malpica, MD - Aerospace Medicine & Physiological Research

## Acknowledgments

- Centro de Telemedicina de Colombia
- Women AeroSTEAM
- Valquiria Space Analog Simulation team