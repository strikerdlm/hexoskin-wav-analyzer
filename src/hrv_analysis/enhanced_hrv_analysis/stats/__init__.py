"""
Advanced statistical analysis modules for HRV analysis.

This subpackage provides sophisticated statistical methods including:
- Generalized Additive Models (GAMs) for nonlinear trend analysis
- Mixed-effects models for repeated measures
- Power analysis and effect size calculations
- Bootstrap confidence intervals and permutation tests
"""

from .advanced_statistics import AdvancedStats as AdvancedHRVStatistics
from .advanced_statistics import AdvancedStats

__all__ = ['AdvancedStats', 'AdvancedHRVStatistics'] 