# Enhanced HRV Analysis System v2.0.0

A comprehensive, modularized Heart Rate Variability analysis system developed for the Valquiria Space Analog Simulation project. This enhanced version includes advanced statistical methods, machine learning integration, interactive visualizations, and a modern GUI interface.

## 🚀 Features

### Core Improvements
- **Modularized Architecture**: Clean separation of concerns with robust error handling
- **Performance Optimization**: NumPy vectorization and parallel processing with joblib
- **Comprehensive Testing**: Unit tests with pytest framework
- **Documentation**: PEP 257 compliant docstrings throughout

### Advanced Analysis Capabilities
- **Time Domain**: SDNN, RMSSD, pNN50, geometric measures
- **Frequency Domain**: Welch PSD with VLF/LF/HF band analysis
- **Nonlinear Metrics**: Poincaré plots, DFA, entropy measures
- **Parasympathetic Indices**: HF power, RMSSD, respiratory sinus arrhythmia
- **Sympathetic Indices**: LF/HF ratio, stress index, autonomic balance

### Statistical Methods
- **GAMs**: Generalized Additive Models for nonlinear temporal trends
- **Mixed-Effects Models**: Repeated measures with subject random effects
- **Bootstrap Confidence Intervals**: Robust statistical inference
- **Power Analysis**: Effect size calculations and sample size planning

### Machine Learning Integration
- **K-means Clustering**: Autonomic phenotype identification
- **ARIMA Forecasting**: SOL trend prediction and adaptation analysis
- **Dimensionality Reduction**: PCA and UMAP for data exploration
- **Outlier Detection**: Multivariate methods for data quality assessment

### Interactive Visualizations
- **Plotly Integration**: Interactive Poincaré plots, PSD analysis, correlation heatmaps
- **HTML Export**: Dynamic reports with embedded visualizations
- **Real-time Updates**: Live plot updates during analysis
- **Dashboard Views**: Comprehensive multi-panel analysis displays

### GUI Application
- **Modern Tkinter Interface**: Themed, responsive design
- **Data Management**: Multiple data source support (SQLite, CSV, sample data)
- **Analysis Configuration**: Flexible parameter selection
- **Real-time Progress**: Progress bars and status updates
- **Export Capabilities**: Results, plots, and comprehensive reports

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start
```bash
# Clone or download the enhanced system
cd working_folder

# Install dependencies
python run_enhanced_hrv_analysis.py --install-deps

# Launch GUI application
python run_enhanced_hrv_analysis.py --gui

# Or run a quick demo
python run_enhanced_hrv_analysis.py --demo
```

### Manual Installation
```bash
pip install -r requirements.txt
```

### Dependencies

#### Core Requirements
- `numpy>=1.20.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `matplotlib>=3.5.0` - Basic plotting
- `scipy>=1.7.0` - Scientific computing
- `scikit-learn>=1.0.0` - Machine learning
- `seaborn>=0.11.0` - Statistical plotting

#### Enhanced Features
- `plotly>=5.0.0` - Interactive visualizations
- `statsmodels>=0.13.0` - Advanced statistics
- `joblib>=1.0.0` - Parallel processing
- `hrv-analysis>=1.0.4` - HRV computation

#### Optional Advanced Features
- `pmdarima>=2.0.0` - ARIMA modeling
- `prophet>=1.1.0` - Trend forecasting
- `hdbscan>=0.8.0` - Density-based clustering
- `umap-learn>=0.5.0` - Dimensionality reduction
- `pingouin>=0.5.0` - Statistical power analysis

## 🔧 Usage

### GUI Application
```bash
# Launch the comprehensive GUI
python run_enhanced_hrv_analysis.py --gui
```

The GUI provides:
- **Data Loading**: Support for SQLite databases, CSV files, or sample data
- **Subject Selection**: Individual or group analysis
- **Analysis Configuration**: HRV domain selection, advanced options
- **Real-time Processing**: Progress tracking with detailed status updates
- **Interactive Results**: Tabbed interface with results, visualizations, and statistics
- **Export Functions**: Results to JSON/CSV, plots to HTML, comprehensive reports

### Command Line Interface
```bash
# Run analysis with your data
python run_enhanced_hrv_analysis.py --cli --data path/to/your/data.db

# Quick analysis with sample data
python run_enhanced_hrv_analysis.py --cli
```

### Programmatic Usage
```python
from enhanced_hrv_analysis.core.data_loader import DataLoader
from enhanced_hrv_analysis.core.signal_processing import SignalProcessor
from enhanced_hrv_analysis.core.hrv_processor import HRVProcessor, HRVDomain

# Load data
loader = DataLoader()
data = loader.load_database_data("merged_data.db")

# Process signals
processor = SignalProcessor()
rr_intervals, info = processor.compute_rr_intervals(data['heart_rate [bpm]'])

# Compute HRV metrics
hrv_processor = HRVProcessor()
results = hrv_processor.compute_hrv_metrics(
    rr_intervals,
    domains=[HRVDomain.TIME, HRVDomain.FREQUENCY, HRVDomain.NONLINEAR],
    include_confidence_intervals=True
)
```

### Advanced Analysis Examples

#### Clustering Analysis
```python
from enhanced_hrv_analysis.ml_analysis.clustering import HRVClustering

# Prepare HRV metrics DataFrame
hrv_data = pd.DataFrame(...)  # Your HRV metrics

# Perform clustering
clustering = HRVClustering()
cluster_result = clustering.perform_kmeans_clustering(hrv_data)
interpretation = clustering.interpret_clusters(cluster_result)

print(f"Identified {cluster_result.n_clusters} autonomic phenotypes")
print(f"Silhouette score: {cluster_result.silhouette_score:.3f}")
```

#### Time Series Forecasting
```python
from enhanced_hrv_analysis.ml_analysis.forecasting import HRVForecasting

# Prepare time series data
forecasting = HRVForecasting()
time_series_data = forecasting.prepare_time_series(
    hrv_data, value_col='rmssd', time_col='Sol'
)

# Fit models and forecast
for subject, ts in time_series_data.items():
    comparison = forecasting.compare_models(ts)
    print(f"{subject}: Best model = {comparison.best_model_name}")
```

#### Interactive Visualization
```python
from enhanced_hrv_analysis.visualization.interactive_plots import InteractivePlotter

plotter = InteractivePlotter()

# Create interactive Poincaré plot
fig = plotter.create_poincare_plot(
    rr_intervals, 
    title="HRV Poincaré Analysis",
    show_ellipse=True,
    color_by_time=True
)

# Export to HTML
plotter.export_html(fig, "poincare_analysis.html")
```

#### Statistical Analysis
```python
from enhanced_hrv_analysis.stats.advanced_statistics import AdvancedStats

stats_analyzer = AdvancedStats()

# GAM analysis for temporal trends
gam_result = stats_analyzer.fit_gam_temporal_trend(
    hrv_data, outcome_var='rmssd', time_var='Sol'
)

# Bootstrap confidence intervals
ci_lower, ci_upper, bootstrap_samples = stats_analyzer.bootstrap_confidence_interval(
    hrv_metrics['rmssd'], np.mean, confidence_level=0.95
)
```

## 📊 Data Format

### Database Schema (SQLite)
```sql
-- Expected table structure
CREATE TABLE merged_data (
    subject TEXT,
    Sol INTEGER,
    "heart_rate [bpm]" REAL,
    "SPO2 [%]" REAL,
    "temperature_celcius [C]" REAL,
    "time [s/1000]" REAL,
    -- Additional physiological parameters...
);
```

### CSV Format
```csv
subject,Sol,heart_rate [bpm],SPO2 [%],temperature_celcius [C],time [s/1000]
T01_Subject1,2,72.5,97.2,36.8,1000
T01_Subject1,2,74.1,97.0,36.9,2000
...
```

### Supported Data Sources
- **SQLite Database**: Primary format for the Valquiria dataset
- **CSV Files**: Individual or multiple files in a directory
- **Sample Data**: Automatically generated realistic physiological data

## 🧪 Testing

```bash
# Run complete test suite
python run_enhanced_hrv_analysis.py --test

# Run specific test categories
pytest enhanced_hrv_analysis/tests/test_core_functionality.py -v
pytest enhanced_hrv_analysis/tests/test_ml_integration.py -v
```

### Test Coverage
- **Core Functionality**: Data loading, signal processing, HRV computation
- **Statistical Methods**: GAMs, mixed-effects models, power analysis
- **Machine Learning**: Clustering, forecasting, dimensionality reduction  
- **Visualization**: Plot generation and export
- **Integration**: Complete analysis workflows
- **Error Handling**: Graceful failure and recovery

## 📈 Output Files

### Results Export
- **JSON Format**: Complete analysis results with metadata
- **CSV Format**: Tabular HRV metrics for statistical analysis
- **HTML Reports**: Comprehensive analysis reports with embedded plots

### Visualizations
- **Interactive HTML Plots**: Poincaré plots, PSD analysis, correlation heatmaps
- **Dashboard Views**: Multi-panel analysis summaries
- **Statistical Plots**: Trend analysis, cluster visualizations

### Analysis Reports
- **Individual Subject Reports**: Detailed HRV analysis per subject/session
- **Group Analysis**: Population-level statistics and comparisons
- **Clustering Reports**: Autonomic phenotype identification and interpretation
- **Forecasting Reports**: Temporal trend analysis and adaptation predictions

## 🔬 Scientific Applications

### Space Analog Research
- **Autonomic Adaptation**: Track changes in autonomic function during isolation
- **Stress Response**: Quantify sympathetic activation and recovery patterns  
- **Individual Differences**: Identify autonomic phenotypes and adaptation strategies
- **Temporal Dynamics**: Model adaptation trajectories and predict future states

### Clinical Applications  
- **Cardiovascular Health**: Assess autonomic cardiac control
- **Stress Assessment**: Quantify sympathovagal balance
- **Recovery Monitoring**: Track autonomic recovery after interventions
- **Risk Stratification**: Identify individuals at risk for autonomic dysfunction

### Research Features
- **Reproducible Analysis**: Standardized processing with version control
- **Statistical Rigor**: Confidence intervals, power analysis, effect sizes
- **Publication Ready**: Formatted results and publication-quality figures
- **Extensible Framework**: Easy integration of new HRV methods and metrics

## 🛠️ Development

### Project Structure
```
enhanced_hrv_analysis/
├── core/                  # Core functionality
│   ├── data_loader.py     # Data loading and validation
│   ├── signal_processing.py  # RR interval processing
│   └── hrv_processor.py   # HRV metrics computation
├── visualization/         # Interactive plotting
│   └── interactive_plots.py
├── stats/                 # Advanced statistics
│   └── advanced_statistics.py
├── ml_analysis/          # Machine learning
│   ├── clustering.py
│   └── forecasting.py
├── gui/                  # GUI application
│   └── main_application.py
└── tests/                # Test suite
    └── test_core_functionality.py
```

### Contributing Guidelines
1. **Code Style**: Follow PEP 8 guidelines
2. **Documentation**: Add docstrings per PEP 257
3. **Testing**: Include unit tests for new functionality
4. **Error Handling**: Implement comprehensive error handling
5. **Performance**: Consider vectorization and parallel processing

### Extending the System
```python
# Add new HRV metrics
class CustomHRVProcessor(HRVProcessor):
    def _compute_custom_metrics(self, rr_intervals):
        # Implement custom HRV metrics
        return custom_metrics

# Add new visualization types
class CustomPlotter(InteractivePlotter):
    def create_custom_plot(self, data):
        # Implement custom visualization
        return plotly_figure
```

## 📚 References

### HRV Guidelines
- Task Force of the European Society of Cardiology. (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use.
- Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. Frontiers in public health, 5, 258.

### Statistical Methods  
- Hastie, T., & Tibshirani, R. (1990). Generalized Additive Models. Chapman and Hall.
- Pinheiro, J., & Bates, D. (2000). Mixed-effects models in S and S-PLUS. Springer.

### Machine Learning Applications
- Brennan, M., Palaniswami, M., & Kamen, P. (2001). Do existing measures of Poincaré plot geometry reflect nonlinear features of heart rate variability? IEEE transactions on biomedical engineering, 48(11), 1342-1347.

## 📞 Support

### Issue Reporting
- Check existing issues in the project repository
- Provide detailed error messages and system information
- Include sample data when possible

### Feature Requests
- Describe the scientific rationale for new features
- Provide examples of expected input/output
- Consider backward compatibility

### Community
- Join discussions about HRV analysis methodology
- Share analysis results and interpretations
- Contribute to the open-source HRV analysis ecosystem

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Valquiria Space Analog Simulation Team** - For providing the motivation and dataset
- **HRV Analysis Community** - For scientific foundations and methodological guidance
- **Open Source Contributors** - For the excellent libraries that make this project possible

---

**Enhanced HRV Analysis System v2.0.0** - Comprehensive autonomic nervous system analysis for space analog research and beyond. 