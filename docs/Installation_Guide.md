# Valquiria Data Analysis Suite - Installation Guide

## Prerequisites

### System Requirements
- **Python**: 3.8+ (tested up to Python 3.11)
- **Operating System**: Cross-platform (Linux, macOS, Windows)
- **Memory**: 8GB RAM minimum, 16GB recommended for large datasets
- **Storage**: 2GB free space for installation and cache files
- **Git**: For cloning the repository

### Recommended Environment
- **Virtual Environment**: Always use a Python virtual environment
- **IDE**: VSCode, PyCharm, or Jupyter Lab for development
- **Terminal**: Modern terminal with UTF-8 support

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Valquiria-Data-Analysis
```

### 2. Create Virtual Environment

**Using venv (Recommended):**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**Using conda:**
```bash
# Create conda environment
conda create -n valquiria python=3.10
conda activate valquiria
```

### 3. Install Dependencies

**Install core dependencies:**
```bash
pip install -r requirements.txt
```

**Install optional dependencies for full functionality:**
```bash
# For enhanced performance (recommended)
pip install numba

# For machine learning analysis
pip install scikit-learn

# For interactive visualization
pip install plotly kaleido

# For advanced statistical modeling
pip install statsmodels
```

### 4. Verify Installation

Run the verification script to check all dependencies:
```bash
python test_libraries.py
```

Expected output:
```
✅ numpy - OK
✅ pandas - OK
✅ scipy - OK
✅ tkinter - OK
✅ matplotlib - OK
⚠️  numba - OPTIONAL (for performance)
⚠️  plotly - OPTIONAL (for interactive plots)
⚠️  sklearn - OPTIONAL (for ML analysis)

Installation verification completed successfully!
```

## Quick Start Verification

### Test Hexoskin WAV Analyzer
```bash
python hexoskin_wav_loader.py
```
This should launch the GUI interface.

### Test Enhanced HRV Analysis
```bash
cd working_folder/enhanced_hrv_analysis
python launch_hrv_analysis.py
```
This should launch the advanced HRV analysis GUI.

## Configuration

### Environment Variables (Optional)
```bash
# Set cache directory (default: ./hrv_cache)
export HRV_CACHE_DIR=/path/to/cache

# Set log level (default: INFO)
export HRV_LOG_LEVEL=DEBUG

# Set maximum memory usage (default: 8GB)
export HRV_MAX_MEMORY_GB=16
```

### Cache Configuration
The system uses intelligent caching. Default settings:
- Cache directory: `./hrv_cache`
- Maximum cache size: 500MB
- Cache TTL: 24 hours

## Troubleshooting

### Common Issues

**1. Python Version Issues**
```bash
# Check Python version
python --version

# If using older Python, install compatible version
pyenv install 3.10.0
pyenv global 3.10.0
```

**2. Tkinter Missing (Linux)**
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# CentOS/RHEL
sudo yum install tkinter
```

**3. Memory Issues with Large Datasets**
- Reduce cache size in settings
- Use chunked processing options
- Increase system virtual memory

**4. Performance Issues**
```bash
# Install performance packages
pip install numba fastparquet
```

### Dependency Conflicts

**Create clean environment:**
```bash
# Remove existing environment
rm -rf venv

# Create fresh environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

### Platform-Specific Notes

**Linux:**
- Ensure `python3-dev` is installed for compilation
- May need `build-essential` for certain packages

**macOS:**
- Use Homebrew Python for best compatibility
- May need Xcode command line tools

**Windows:**
- Use UTF-8 console encoding: `chcp 65001`
- Consider Windows Terminal for better Unicode support

## Development Setup

### Additional Development Dependencies
```bash
pip install pytest pytest-cov black flake8 mypy
```

### Run Tests
```bash
cd working_folder/enhanced_hrv_analysis/tests
python run_all_tests.py
```

### Code Formatting
```bash
black . --line-length 100
flake8 . --max-line-length 100
```

## Performance Optimization

### For Large Datasets
1. **Enable Numba JIT compilation:**
   ```bash
   pip install numba
   ```

2. **Configure memory limits:**
   ```python
   # In GUI settings or config file
   MAX_MEMORY_MB = 2000  # 2GB limit
   CHUNK_SIZE = 10000    # Process in chunks
   ```

3. **Use SSD storage for cache:**
   ```bash
   export HRV_CACHE_DIR=/path/to/ssd/cache
   ```

## Docker Setup (Advanced)

**Create Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["python", "hexoskin_wav_loader.py"]
```

**Build and run:**
```bash
docker build -t valquiria-analysis .
docker run -p 8080:8080 valquiria-analysis
```

## Support

For installation issues:
1. Check this guide first
2. Verify Python and pip versions
3. Check system requirements
4. Review error logs in `./logs/`
5. Contact development team

## Updating

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Update Application
```bash
git pull origin main
pip install -r requirements.txt
```

---

**Note:** This installation guide replaces all Windows-specific batch files. The application is designed to run in a proper Python virtual environment for better dependency management and reproducibility. 