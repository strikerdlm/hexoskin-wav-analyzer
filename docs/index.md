---
layout: default
title: Hexoskin WAV Analyzer
---

# Hexoskin WAV Analyzer

![Hexoskin WAV Analyzer](assets/images/header.png)

## About

The Hexoskin WAV Analyzer is an advanced application for analyzing physiological data from Hexoskin smart garments. Developed by Diego Malpica, MD for aerospace medicine research, this tool helps researchers and clinicians work with physiological data collected during space analog simulations and other research scenarios.

## Key Features

- **Data Loading**: Load and decode Hexoskin WAV files (ECG, breathing, etc.)
- **Preprocessing**: Apply filters, resample data, and normalize datasets
- **Visualization**: Plot data with customizable options and time unit conversion
- **Statistical Analysis**: Perform descriptive statistics and normality tests
- **Comparison**: Compare datasets with non-parametric statistical tests
- **Export**: Save processed data to CSV and export graphs as PNG images

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/strikerdlm/hexoskin-wav-analyzer.git

# Navigate to the project directory
cd hexoskin-wav-analyzer

# Install dependencies
pip install -r requirements.txt
```

For detailed installation instructions, see the [Installation Guide](https://github.com/strikerdlm/hexoskin-wav-analyzer/blob/main/INSTALL.md).

### Usage

```bash
# Run the GUI application
python hexoskin_wav_loader.py

# Or use the example script
python hexoskin_wav_example.py path/to/your/file.wav
```

For detailed usage instructions, see the [README](https://github.com/strikerdlm/hexoskin-wav-analyzer/blob/main/README.md).

## Documentation

- [Installation Guide](https://github.com/strikerdlm/hexoskin-wav-analyzer/blob/main/INSTALL.md)
- [User Manual](https://github.com/strikerdlm/hexoskin-wav-analyzer/blob/main/README.md)
- [Contributing Guidelines](https://github.com/strikerdlm/hexoskin-wav-analyzer/blob/main/CONTRIBUTING.md)
- [Release Notes](https://github.com/strikerdlm/hexoskin-wav-analyzer/blob/main/RELEASE_NOTES.md)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/strikerdlm/hexoskin-wav-analyzer/blob/main/LICENSE) file for details.

## Acknowledgments

- Centro de Telemedicina de Colombia
- Women AeroSTEAM
- Valquiria Space Analog Simulation team 