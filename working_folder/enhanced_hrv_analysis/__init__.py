"""
Enhanced HRV Analysis System for the Valquiria Dataset

This package provides a comprehensive, modularized HRV analysis system with:
- Robust error handling and validation
- Interactive visualizations with Plotly
- Advanced statistical methods (GAMs, mixed-effects models)
- Machine learning integration (clustering, forecasting)
- Performance optimizations with parallel processing
- Comprehensive testing with pytest

Author: Enhanced by AI Assistant
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Enhanced HRV Analysis Team"

from .core.hrv_processor import HRVProcessor
from .core.data_loader import DataLoader
from .core.signal_processing import SignalProcessor
from .visualization.interactive_plots import InteractivePlotter
from .stats.advanced_statistics import AdvancedStats
from .ml_analysis.clustering import HRVClustering
from .ml_analysis.forecasting import HRVForecasting

__all__ = [
    'HRVProcessor',
    'DataLoader', 
    'SignalProcessor',
    'InteractivePlotter',
    'AdvancedStats',
    'HRVClustering',
    'HRVForecasting'
] 