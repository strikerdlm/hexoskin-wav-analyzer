"""
CSV Joiner for Valquiria Research Data

This package provides functionality for joining CSV files for the same subjects
across different Sol folders in the Valquiria research data.
"""

from .csv_joiner import main, join_csv_files, setup_logging, list_all_subject_folders

__version__ = "1.0.0"
__author__ = "Valquiria Research Team" 