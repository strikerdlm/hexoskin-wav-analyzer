"""
Scripts package for Valquiria Jupyter Analysis
Helper functions and utilities for data analysis
"""

__version__ = "1.0.0"
__author__ = "Valquiria Research Team"
__description__ = "Helper scripts for Jupyter notebook analysis"

from .load_data import load_csv_data, load_database_data, get_data_summary
from .analysis_utils import (
    setup_plotting_style,
    analyze_variable_distribution,
    correlation_analysis,
    time_series_analysis,
    statistical_comparison,
    quick_analysis
)

__all__ = [
    'load_csv_data',
    'load_database_data', 
    'get_data_summary',
    'setup_plotting_style',
    'analyze_variable_distribution',
    'correlation_analysis',
    'time_series_analysis',
    'statistical_comparison',
    'quick_analysis'
] 