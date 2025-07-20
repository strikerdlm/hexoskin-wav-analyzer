# Valquiria Data Analysis Suite - Project Structure

This document provides a comprehensive overview of the project structure and organization.

## 📁 Root Directory

```
Valquiria-Data-Analysis/
├── 📄 README.md                      # Main project documentation
├── 📄 requirements.txt               # Python dependencies
├── 📄 setup.py                      # Installation script
├── 📄 LICENSE                       # Project license
├── 📄 test_libraries.py            # Dependency verification
├── 📄 version_check.py              # Version compatibility check
└── 📄 library_versions.txt         # Tested library versions
```

## 🧠 Core Analysis Components

### Hexoskin WAV File Analyzer
```
├── 📄 hexoskin_wav_loader.py        # Main GUI application
├── 📄 hexoskin_wav_example.py       # Command-line usage examples
├── 📄 analyze_data.py               # Data analysis utilities
├── 📄 process_data.py               # Data processing scripts
└── 📄 load_data.py                  # Data loading functions
```

**Key Features:**
- GUI-based physiological data analysis
- Statistical testing suite (15+ tests)
- Multi-dataset comparison capabilities
- Export functionality (CSV, PNG, statistics)

### Enhanced HRV Analysis System
```
working_folder/enhanced_hrv_analysis/
├── 📄 launch_hrv_analysis.py        # Main application launcher
├── 📄 __init__.py                   # Package initialization
├── 📁 core/                         # Core processing modules
├── 📁 gui/                          # Graphical user interface
├── 📁 ml_analysis/                  # Machine learning components
├── 📁 stats/                        # Advanced statistics
├── 📁 visualization/                # Interactive plotting
├── 📁 tests/                        # Test suite
└── 📁 hrv_cache/                    # Intelligent caching system
```

## 🔧 Core Processing Modules (`core/`)

```
core/
├── 📄 __init__.py                   # Package exports
├── 📄 data_loader.py               # Enhanced data loading with validation
├── 📄 hrv_processor.py             # Complete HRV analysis engine
├── 📄 signal_processing.py         # Signal processing and artifact detection
├── 📄 async_processor.py           # Asynchronous processing manager
├── 📄 intelligent_cache.py         # Advanced caching system
└── 📄 optimized_data_loader.py     # High-performance data loading
```

**Capabilities:**
- **Data Loading**: SQLite, CSV, large dataset handling
- **HRV Analysis**: Time/frequency/nonlinear domains
- **Signal Processing**: Artifact detection, filtering, quality assessment
- **Performance**: Async processing, intelligent caching, memory management

## 🖥️ User Interface (`gui/`)

```
gui/
├── 📄 __init__.py                   # GUI package initialization
├── 📄 main_application.py          # Main HRV analysis application
├── 📄 performance_monitor.py       # Real-time performance monitoring
├── 📄 settings_panel.py            # User configuration interface
└── 📄 requirements.txt             # GUI-specific dependencies
```

**Features:**
- Modern tkinter-based interface
- Real-time analysis progress tracking
- Performance monitoring dashboard
- Configurable analysis parameters
- Interactive result visualization

## 🤖 Machine Learning (`ml_analysis/`)

```
ml_analysis/
├── 📄 __init__.py                   # ML package initialization
├── 📄 clustering.py                 # Autonomic phenotype clustering
└── 📄 forecasting.py                # Time-series forecasting
```

**Capabilities:**
- **Clustering**: K-means, hierarchical, HDBSCAN for autonomic phenotyping
- **Forecasting**: ARIMA, Prophet models for adaptation prediction
- **Validation**: Silhouette analysis, cross-validation, model comparison

## 📊 Advanced Statistics (`stats/`)

```
stats/
├── 📄 __init__.py                   # Statistics package initialization
└── 📄 advanced_statistics.py       # GAM, mixed-effects, bootstrap analysis
```

**Methods:**
- **GAM Analysis**: Temporal trend modeling with splines
- **Mixed-Effects Models**: Hierarchical analysis accounting for individual differences
- **Bootstrap Methods**: Confidence intervals and hypothesis testing
- **Power Analysis**: Sample size calculations and sensitivity analysis

## 📈 Visualization (`visualization/`)

```
visualization/
├── 📄 __init__.py                   # Visualization package initialization
└── 📄 interactive_plots.py         # Plotly-based interactive visualizations
```

**Plot Types:**
- **Poincaré Plots**: HRV scatter plots with ellipse fitting
- **PSD Analysis**: Power spectral density with frequency bands
- **Time Series**: Interactive temporal analysis
- **Dashboards**: Comprehensive multi-panel visualizations

## 🧪 Test Suite (`tests/`)

```
tests/
├── 📄 __init__.py                   # Test package initialization
├── 📄 run_all_tests.py             # Master test runner
├── 📄 test_core_functionality.py   # Core module testing
├── 📄 test_advanced_statistics.py  # Statistics testing
├── 📄 test_ans_balance_analysis.py # ANS balance testing
├── 📄 test_ml_analysis.py          # Machine learning testing
└── 📄 test_visualization.py        # Visualization testing
```

## 🗂️ Data Organization

### Database Storage (`DBs/`)
```
DBs/
├── 📁 Sol 2 (completo)/            # Sol 2 mission data
├── 📁 Sol 3 (completo)/            # Sol 3 mission data
├── 📁 Sol 4 (completo)/            # Sol 4 mission data
├── ...                             # Additional Sol missions
└── 📁 Sol 16 (completo)/           # Latest Sol mission data
```

### Working Data (`working_folder/`)
```
working_folder/
├── 📁 Jupyter notebooks/           # Analysis notebooks
├── 📁 hrv_results/                 # Generated analysis results
├── 📁 scripts/                     # Utility and analysis scripts
├── 📄 merged_data.db               # Consolidated database
├── 📄 T01_Mara.csv                # Individual subject data
├── 📄 T02_Laura.csv               # Individual subject data
├── ...                            # Additional subject files
└── 📄 T08_Karina.csv              # Individual subject data
```

## 🔧 Utilities & Scripts

### CSV Processing (`csv_joiner/`)
```
csv_joiner/
├── 📄 __init__.py                   # Package initialization
├── 📄 csv_joiner.py                # Main CSV merging utility
├── 📄 example.py                   # Usage examples
├── 📄 README.md                    # CSV joiner documentation
├── 📄 requirements.txt             # Dependencies
└── 📁 logs/                        # Processing logs
```

### Hexoskin Backup (`hexoskin_backup/`)
```
hexoskin_backup/
├── 📄 hexoskin_wav_loader.py       # Backup WAV loader
├── 📄 README.md                    # Backup documentation
├── 📄 requirements.txt             # Dependencies
└── 📁 data/                        # Sample data files
```

## 📚 Documentation (`docs/`)

```
docs/
├── 📄 README.md                     # General documentation
├── 📄 User_Manual.md               # Complete user guide
├── 📄 Installation_Guide.md        # Setup instructions
├── 📄 Project_Structure.md         # This document
├── 📄 BRANCH_PROTECTION.md         # Git branch policies
├── 📄 IMPROVEMENTS_SUMMARY.md      # Development history
├── 📄 Enhanced_HRV_Analysis.md     # HRV system documentation
├── 📄 HRV_ANALYSIS_COMPREHENSIVE_DOCUMENTATION.md  # Technical reference
├── 📄 Scientific_Discussion_Parasympathetic_Analysis.md  # Research findings
└── 📄 HRV_ANALYSIS_FIX_GUIDE.md   # Troubleshooting guide
```

## 🚀 Entry Points & Launch Scripts

### Primary Applications
1. **Hexoskin WAV Analyzer GUI**:
   ```bash
   python hexoskin_wav_loader.py
   ```

2. **Enhanced HRV Analysis System**:
   ```bash
   cd working_folder/enhanced_hrv_analysis
   python launch_hrv_analysis.py
   ```

3. **Command-line Analysis**:
   ```bash
   python hexoskin_wav_example.py <wav_file>
   ```

### Utility Scripts
- `test_libraries.py`: Verify all dependencies
- `version_check.py`: Check Python/library compatibility
- `simple_test.py`: Basic functionality test
- `library_test.py`: Extended library testing

## 🗄️ Data Flow Architecture

```
Raw Data Sources
    ↓
[Data Loaders] → [Validation] → [Quality Assessment]
    ↓
[Signal Processing] → [Artifact Detection] → [Cleaning]
    ↓
[HRV Analysis] → [Time Domain] → [Frequency Domain] → [Nonlinear]
    ↓
[Advanced Analysis] → [ML Clustering] → [Forecasting] → [Statistics]
    ↓
[Visualization] → [Interactive Plots] → [Reports] → [Export]
```

## ⚡ Performance & Caching

### Cache Structure
```
hrv_cache/
├── 📄 cache_metadata.db            # Cache index and metadata
├── 📁 compressed/                  # LZ4/GZIP compressed results
├── 📁 raw/                         # Uncompressed cache files
└── 📁 temp/                        # Temporary processing files
```

### Memory Management
- **Intelligent Caching**: LRU eviction with compression
- **Chunked Processing**: Large dataset streaming
- **Async Operations**: Non-blocking GUI with progress tracking
- **Memory Limits**: Configurable memory usage boundaries

## 📋 Configuration Files

- `requirements.txt`: Python package dependencies
- `setup.py`: Installation and packaging configuration
- `working_folder/enhanced_hrv_analysis/gui/requirements.txt`: GUI-specific dependencies
- Various `README.md` files: Component-specific documentation
- `.gitignore`: Git exclusion patterns
- `LICENSE`: Project licensing terms

## 🛠️ Development Workflow

1. **Setup**: Virtual environment creation and dependency installation
2. **Testing**: Comprehensive test suite execution
3. **Development**: Feature implementation with test coverage
4. **Documentation**: Markdown documentation updates
5. **Performance**: Profiling and optimization
6. **Validation**: Scientific method verification

---

This structure supports both research workflows and production-quality analysis while maintaining clear separation of concerns and modular architecture. 