# Hexoskin WAV Analyzer v0.0.3 - Bug Fix and Enhancement Release

**Release Date:** March 2025

## Improvements

- Enhanced consistency in variable naming throughout codebase
- Improved robustness of comparison functionality
- Better handling of multiple dataset comparisons
- Optimized code structure for maintainability

## Bug Fixes

- Fixed critical issue with notebook widget initialization
- Resolved problem with post-hoc analysis not displaying in comparison results
- Fixed variable naming inconsistencies in post-hoc analysis code
- Corrected indentation errors in various functions

---

# Hexoskin WAV Analyzer v0.0.2 - Statistical Enhancement Release

**Release Date:** March 2025

## New Features

- Enhanced statistical analysis with comprehensive descriptive statistics
- Advanced normality tests including Anderson-Darling and Jarque-Bera
- Support for comparing up to 15 datasets simultaneously
- New statistical tests: Welch's ANOVA, RM-ANOVA, and Aligned Ranks Transform
- Improved post-hoc analysis with multiple correction methods (Bonferroni, FDR)
- Interactive statistical visualization with QQ plots and histograms
- Export capabilities for all statistical results

## Improvements

- More detailed descriptive statistics including variance, IQR, and percentiles
- Better visualization of data distributions with QQ plots and histograms
- Enhanced effect size calculations for all statistical tests
- Improved recommendations based on data characteristics
- More robust handling of edge cases in statistical analysis

## Bug Fixes

- Fixed issues with normality test calculations for large datasets
- Corrected effect size calculations for non-parametric tests
- Improved error handling in statistical comparison functions
- Fixed display issues in the statistics results window

---

# Hexoskin WAV Analyzer v0.0.1 - Initial Release

This is the initial release of the Hexoskin WAV Analyzer, a tool for analyzing physiological data from Hexoskin smart garments.

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

See the [INSTALL.md](INSTALL.md) file for detailed installation instructions.

## Usage

See the [README.md](README.md) file for detailed usage instructions.

## Known Issues

- None reported yet. Please submit issues on GitHub.

## Future Plans

- Add support for more Hexoskin data types
- Improve visualization options
- Add more statistical analysis methods
- Create a web-based version

## Acknowledgments

- Centro de Telemedicina de Colombia
- Women AeroSTEAM
- Valquiria Space Analog Simulation team 