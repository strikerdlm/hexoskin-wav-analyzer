"""
Hexoskin WAV File Analyzer Package

Advanced physiological data analysis for Hexoskin WAV files with comprehensive
statistical analysis and multi-dataset comparison capabilities.

Key Features:
- WAV file loading and signal processing
- Statistical analysis with 15+ tests
- Multi-dataset comparison
- Interactive visualization
- Export capabilities

Author: Dr. Diego Malpica MD - Aerospace Medicine Specialist
"""

__version__ = "1.0.3"

# Import main components
try:
    from .hexoskin_wav_loader import HexoskinWavLoader
except ImportError:
    # Handle import error gracefully
    HexoskinWavLoader = None

__all__ = ['HexoskinWavLoader'] 