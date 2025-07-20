# Valquiria Space Analog Physiological Data Analysis Suite

**Version 2.0.0**

**Author:** Dr. Diego Malpica MD - Aerospace Medicine Specialist  
**Organization:** Colombian Aerospace Force (FAC) / DIMAE  
**Project:** Valquiria Crew Space Simulation - Physiological Research Platform

---

## 🚨 IMPORTANT DISCLAIMER

**This is an ongoing research project developed for scientific and educational purposes only.**

⚠️ **NOT FOR OPERATIONAL DEPLOYMENT** ⚠️
- This software is **NOT approved** for military operations
- This software is **NOT approved** for clinical diagnosis or treatment
- This software is **NOT approved** for operational crew health monitoring
- Use only for research, training, and educational purposes

**For any operational or clinical applications, please consult with certified medical professionals and use validated, regulatory-approved systems.**

---

## Project Overview

The Valquiria Space Analog Physiological Data Analysis Suite is a comprehensive research platform designed to analyze physiological data collected during space analog simulations. The suite combines two powerful components:

1. **Hexoskin WAV File Analyzer** - Complete physiological data processing and analysis
2. **Enhanced HRV Analysis System** - Advanced heart rate variability analysis with machine learning

This platform was developed to support the Valquiria Space Analog Simulation research program, studying physiological adaptations and crew health monitoring in simulated space environments.

## 🌟 Key Features

### Hexoskin WAV File Analyzer
- **Multi-format Data Loading**: Load and decode Hexoskin WAV files containing ECG, respiration, and other physiological signals
- **Advanced Signal Processing**: Automatic artifact detection, filtering, and signal quality assessment
- **Comprehensive Statistics**: 15+ statistical tests including normality, parametric/non-parametric comparisons
- **Multi-dataset Analysis**: Compare up to 15 datasets simultaneously with post-hoc analysis
- **Interactive Visualization**: Real-time plotting with multiple time units and view controls
- **Export Capabilities**: Save processed data, statistical results, and high-quality plots
- **Dual Interface**: Both GUI and command-line interfaces available

### Enhanced HRV Analysis System
- **Complete HRV Analysis**: Time domain, frequency domain, and nonlinear metrics
- **Autonomic Nervous System Assessment**: Advanced parasympathetic, sympathetic, and ANS balance analysis
- **Machine Learning Integration**: Clustering for autonomic phenotyping and forecasting for adaptation prediction
- **Advanced Statistics**: GAM trend analysis, mixed-effects modeling, bootstrap confidence intervals
- **Interactive Dashboards**: Real-time visualization with Plotly-based interactive plots
- **Enterprise Performance**: Intelligent caching, async processing, and database optimization
- **Research Analytics**: Comprehensive statistical reporting and data quality assessment

## 🏗️ Project Structure

```
Valquiria-Data-Analysis/
├── 📁 docs/                           # Documentation (all markdown files)
├── 📁 working_folder/                 # Main analysis workspace
│   ├── 📁 enhanced_hrv_analysis/      # Advanced HRV Analysis System
│   │   ├── 📁 core/                   # Core processing modules
│   │   ├── 📁 gui/                    # Graphical user interface
│   │   ├── 📁 ml_analysis/           # Machine learning components
│   │   ├── 📁 stats/                 # Advanced statistics
│   │   ├── 📁 visualization/         # Interactive plotting
│   │   └── 📁 tests/                 # Test suite
│   ├── 📁 Jupyter notebooks/         # Analysis notebooks
│   ├── 📁 hrv_results/              # Analysis outputs
│   └── 📁 scripts/                   # Utility scripts
├── 📁 DBs/                           # Database files (Sol data)
├── 📁 csv_joiner/                    # Data merging utilities
├── hexoskin_wav_loader.py            # Main Hexoskin analyzer
├── hexoskin_wav_example.py           # Usage examples
├── analyze_data.py                   # Data analysis scripts
├── requirements.txt                  # Python dependencies
└── setup.py                         # Installation script
```

## 🚀 Quick Start

### System Requirements
- **Python**: 3.8+ (tested up to 3.11)
- **Operating System**: Cross-platform (Linux, macOS, Windows)
- **Memory**: 8GB RAM minimum, 16GB recommended for large datasets
- **Storage**: 2GB free space for installation and cache

### Installation

1. **Clone the Repository**
```bash
git clone <repository-url>
cd Valquiria-Data-Analysis
```

2. **Set Up Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify Installation**
```bash
python test_libraries.py
```

### Quick Usage

#### Hexoskin WAV Analyzer
```bash
# GUI Mode (Recommended)
python hexoskin_wav_loader.py

# Command Line Mode
python hexoskin_wav_example.py path/to/your/file.wav
```

#### Enhanced HRV Analysis
```bash
# Launch Advanced HRV Analysis GUI
cd working_folder/enhanced_hrv_analysis
python launch_hrv_analysis.py
```

## 📊 Analysis Capabilities

### Physiological Data Processing
- **Signal Quality Assessment**: Automatic artifact detection and signal validation
- **Multi-parameter Analysis**: Heart rate, SPO2, temperature, blood pressure, respiratory rate
- **Temporal Analysis**: Time-series analysis with circadian rhythm detection
- **Data Integration**: Merge multiple sessions and subjects for longitudinal studies

### Heart Rate Variability (HRV) Analysis
- **Time Domain**: RMSSD, SDNN, pNN50, triangular index, and 15+ metrics
- **Frequency Domain**: VLF, LF, HF power analysis with Welch and AR methods
- **Nonlinear Analysis**: Poincaré plots, DFA, entropy measures
- **Autonomic Balance**: Advanced sympathetic/parasympathetic assessment

### Advanced Analytics
- **Machine Learning**: Unsupervised clustering for autonomic phenotyping
- **Predictive Modeling**: Time-series forecasting for adaptation prediction
- **Statistical Modeling**: GAM, mixed-effects, bootstrap confidence intervals
- **Multi-subject Analysis**: Population-level analysis with individual profiles

### Visualization & Reporting
- **Interactive Plots**: Poincaré plots, PSD analysis, time-series visualization
- **Statistical Dashboards**: Real-time analytics with performance monitoring
- **Export Options**: HTML reports, CSV data, high-resolution plots
- **Research Reports**: Automated generation of analysis summaries

## 🔬 Scientific Features

### Research-Grade Analysis
- **Artifact Detection**: Multiple algorithms (Malik, Karlsson, Kamath, IQR)
- **Quality Metrics**: Comprehensive signal quality assessment
- **Statistical Validation**: 15+ normality tests and comparison methods
- **Confidence Intervals**: Bootstrap and parametric confidence estimation

### Space Medicine Applications
- **Adaptation Tracking**: Longitudinal analysis of physiological adaptation
- **Stress Assessment**: Autonomic nervous system stress indicators
- **Crew Monitoring**: Individual and group health status analysis
- **Mission Planning**: Predictive modeling for mission duration effects

## 💻 Technical Specifications

### Performance Features
- **Intelligent Caching**: LRU caching with compression (2-10x speed improvement)
- **Async Processing**: Non-blocking analysis with timeout protection
- **Memory Management**: Adaptive memory limits and garbage collection
- **Database Optimization**: Connection pooling and query optimization

### Data Handling
- **Large Datasets**: Chunked processing for millions of records
- **Multiple Formats**: CSV, SQLite, WAV files with auto-detection
- **Data Validation**: Multi-stage quality assessment and cleaning
- **Export Options**: JSON, CSV, HTML with customizable formats

## 📚 Documentation

All project documentation is organized in the `docs/` folder:
- `docs/User_Manual.md` - Complete user guide
- `docs/API_Documentation.md` - Developer reference
- `docs/Scientific_Methods.md` - Analysis methodologies
- `docs/Installation_Guide.md` - Detailed setup instructions

## 🧪 Testing & Validation

The project includes comprehensive test suites:
```bash
# Run all tests
cd working_folder/enhanced_hrv_analysis/tests
python run_all_tests.py

# Run specific component tests
python test_core_functionality.py
python test_advanced_statistics.py
python test_ml_analysis.py
```

## 🤝 Contributors & Acknowledgments

### Special Thanks
- **Centro de Telemedicina de Colombia** - Technical collaboration
- **Women AeroSTEAM** - Educational partnership
- **Valquiria Space Analog Simulation Team** - Research collaboration

### Technical Contributors
- Advanced HRV Analysis Architecture
- Statistical Methods Implementation
- Machine Learning Integration
- Performance Optimization

## 📄 License & Usage

This project is provided as open-source software for **research and educational purposes only**.

### Permitted Uses:
✅ Academic research and publications  
✅ Educational training and demonstrations  
✅ Method development and validation  
✅ Non-commercial scientific collaboration  

### Prohibited Uses:
❌ Military operational deployment  
❌ Clinical diagnosis or treatment  
❌ Commercial health monitoring services  
❌ Safety-critical applications  

## 📞 Contact & Support

**Primary Contact:**  
Dr. Diego Malpica MD  
Aerospace Medicine Specialist  
Colombian Aerospace Force (FAC)  
Email: dlmalpicah@unal.edu.co

**Project Information:**  
For questions about the Valquiria Space Analog Simulation or this software platform, please contact the development team through the official channels.

## 🔄 Version History

### Version 2.0.0 (Current)
- Enhanced HRV Analysis System with ML capabilities
- Advanced statistical methods and GAM analysis
- Interactive visualization dashboard
- Performance optimization with caching
- Comprehensive test suite

### Version 1.0.0
- Hexoskin WAV File Analyzer
- Basic statistical analysis
- GUI and command-line interfaces
- Multi-dataset comparison

---

**⭐ If this software contributes to your research, please cite appropriately and acknowledge the Valquiria Space Analog Simulation project.**

**🔬 Developed for advancing our understanding of human physiological adaptation in extreme environments.**