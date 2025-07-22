"""
Configuration module for Enhanced HRV Analysis System

This module provides centralized configuration for export paths and settings
to ensure all plots and exports go to the unified location.
"""

import os
from pathlib import Path


def get_export_directory() -> Path:
    """
    Get the unified export directory for all plots and analysis exports.
    
    Returns:
        Path: Unified export directory path
    """
    # Check if the export directory was set by the launcher
    if 'HRV_EXPORT_DIR' in os.environ:
        return Path(os.environ['HRV_EXPORT_DIR'])
    
    # Fallback: use plots_output in the enhanced_hrv_analysis directory
    current_dir = Path(__file__).parent
    export_dir = current_dir / "plots_output"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    return export_dir


def get_plots_output_path(filename: str = None) -> Path:
    """
    Get the full path for saving plots and exports.
    
    Args:
        filename: Optional filename to append to the export directory
        
    Returns:
        Path: Full path for the export file or directory
    """
    export_dir = get_export_directory()
    
    if filename:
        return export_dir / filename
    else:
        return export_dir


# Export directory constant for backward compatibility
PLOTS_OUTPUT_DIR = get_export_directory() 