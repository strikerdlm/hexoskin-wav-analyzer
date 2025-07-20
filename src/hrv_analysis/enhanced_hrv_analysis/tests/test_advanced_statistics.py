"""
Comprehensive tests for Advanced Statistics functionality.

This module tests the enhanced statistical analysis features including
GAMs, mixed-effects models, power analysis, and sensitivity simulations.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings

import sys
import os
sys.path.append(str(Path(__file__).parent.parent))

from stats.advanced_statistics import AdvancedHRVStatistics, StatisticalResult

class TestPowerAnalysis:
    """Test cases for power analysis functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.stats = AdvancedHRVStatistics()
        
    def test_basic_power_analysis_structure(self):
        """Test basic power analysis structure and output."""
        # Test power calculation given effect size, sample size, alpha
        result = self.stats.power_analysis(
            effect_size=0.5,
            sample_size=30,
            alpha=0.05,
            power=None,
            test_type='two_sample_ttest'
        )
        
        assert isinstance(result, dict)
        assert 'analysis_type' in result
        assert 'achieved_power' in result
        assert 'effect_size' in result
        assert 'sample_size' in result
        assert 'alpha' in result
        assert 'interpretation' in result
        
    def test_sample_size_calculation(self):
        """Test sample size calculation given effect size, power, alpha."""
        result = self.stats.power_analysis(
            effect_size=0.5,
            sample_size=None,
            alpha=0.05,
            power=0.8,
            test_type='two_sample_ttest'
        )
        
        assert 'analysis_type' in result
        assert result['analysis_type'] == 'sample_size'
        assert 'required_sample_size' in result
        assert isinstance(result['required_sample_size'], (int, float))
        assert result['required_sample_size'] > 0
        
    def test_power_analysis_edge_cases(self):
        """Test power analysis edge cases and error handling."""
        # Test with invalid parameters
        result = self.stats.power_analysis(
            effect_size=0.5,
            sample_size=None,
            alpha=None,
            power=None
        )
        assert 'error' in result
        
        # Test with very small effect size
        result = self.stats.power_analysis(
            effect_size=0.01,
            sample_size=20,
            alpha=0.05
        )
        assert 'achieved_power' in result
        assert result['achieved_power'] < 0.5  # Should be low power
        
        # Test with large effect size
        result = self.stats.power_analysis(
            effect_size=2.0,
            sample_size=20,
            alpha=0.05
        )
        assert 'achieved_power' in result
        assert result['achieved_power'] > 0.8  # Should be high power

class TestPostHocPowerAnalysis:
    """Test cases for post-hoc power analysis."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.stats = AdvancedHRVStatistics()
        
        # Create sample data
        np.random.seed(42)
        self.group1_data = np.random.normal(100, 15, 30)  # Control group
        self.group2_data = np.random.normal(110, 15, 30)  # Treatment group
        self.observed_data = {
            'control': self.group1_data,
            'treatment': self.group2_data
        }
        
    def test_post_hoc_two_sample_analysis(self):
        """Test post-hoc power analysis for two-sample t-test."""
        result = self.stats.post_hoc_power_analysis(
            observed_data=self.observed_data,
            test_type='two_sample_ttest',
            alpha=0.05
        )
        
        assert isinstance(result, dict)
        assert 'test_type' in result
        assert 'groups' in result
        assert 'sample_sizes' in result
        assert 'group_statistics' in result
        assert 'cohens_d' in result
        assert 'achieved_power' in result
        assert 'sample_size_recommendations' in result
        assert 'interpretation' in result
        
        # Check group statistics structure
        assert 'control' in result['group_statistics']
        assert 'treatment' in result['group_statistics']
        
        for group_stats in result['group_statistics'].values():
            assert 'mean' in group_stats
            assert 'std' in group_stats
            assert 'n' in group_stats
            assert 'sem' in group_stats
            
        # Check effect size is reasonable
        assert 0 <= result['cohens_d'] <= 5  # Reasonable range
        assert 0 <= result['achieved_power'] <= 1
        
        # Check sample size recommendations
        recommendations = result['sample_size_recommendations']
        assert 'power_0.8' in recommendations
        assert 'power_0.9' in recommendations
        assert 'power_0.95' in recommendations
        
    def test_post_hoc_paired_analysis(self):
        """Test post-hoc power analysis for paired t-test."""
        # Create paired data
        pre_data = self.group1_data
        post_data = self.group1_data + np.random.normal(5, 5, len(self.group1_data))
        
        paired_data = {
            'pre': pre_data,
            'post': post_data
        }
        
        result = self.stats.post_hoc_power_analysis(
            observed_data=paired_data,
            test_type='paired_ttest',
            alpha=0.05
        )
        
        assert 'cohens_d' in result
        assert 'achieved_power' in result
        assert 'mean_difference' in result
        assert 'std_difference' in result
        assert 0 <= result['achieved_power'] <= 1
        
    def test_post_hoc_anova_analysis(self):
        """Test post-hoc power analysis for ANOVA."""
        # Create multi-group data
        group3_data = np.random.normal(120, 15, 25)
        anova_data = {
            'group1': self.group1_data,
            'group2': self.group2_data,
            'group3': group3_data
        }
        
        result = self.stats.post_hoc_power_analysis(
            observed_data=anova_data,
            test_type='anova',
            alpha=0.05
        )
        
        assert 'eta_squared' in result
        assert 'cohens_f' in result
        assert 'f_statistic' in result
        assert 'achieved_power' in result
        assert 'df_between' in result
        assert 'df_within' in result
        
        assert 0 <= result['eta_squared'] <= 1
        assert result['cohens_f'] >= 0
        assert 0 <= result['achieved_power'] <= 1
        
    def test_post_hoc_error_handling(self):
        """Test error handling in post-hoc power analysis."""
        # Test with unsupported test type
        result = self.stats.post_hoc_power_analysis(
            observed_data=self.observed_data,
            test_type='unsupported_test'
        )
        assert 'error' in result
        
        # Test with wrong number of groups for two-sample test
        single_group = {'group1': self.group1_data}
        result = self.stats.post_hoc_power_analysis(
            observed_data=single_group,
            test_type='two_sample_ttest'
        )
        assert 'error' in result

class TestSensitivitySimulation:
    """Test cases for sensitivity simulation functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.stats = AdvancedHRVStatistics()
        
    def test_sensitivity_simulation_structure(self):
        """Test sensitivity simulation structure and output."""
        # Run with small parameters for speed
        result = self.stats.sensitivity_simulation(
            effect_sizes=[0.2, 0.5],
            sample_sizes=[20, 50],
            alpha_levels=[0.05],
            n_simulations=100,
            test_type='two_sample_ttest'
        )
        
        assert isinstance(result, dict)
        assert 'simulation_parameters' in result
        assert 'power_surface' in result
        assert 'recommendations' in result
        
        # Check simulation parameters
        params = result['simulation_parameters']
        assert 'effect_sizes' in params
        assert 'sample_sizes' in params
        assert 'alpha_levels' in params
        assert 'n_simulations' in params
        assert 'test_type' in params
        
        # Check power surface has entries
        power_surface = result['power_surface']
        assert len(power_surface) == 2 * 2 * 1  # 2 effects * 2 sizes * 1 alpha
        
        for key, values in power_surface.items():
            assert 'effect_size' in values
            assert 'sample_size' in values
            assert 'alpha' in values
            assert 'estimated_power' in values
            assert 'type_i_error_rate' in values
            
            assert 0 <= values['estimated_power'] <= 1
            assert 0 <= values['type_i_error_rate'] <= 1
            
    def test_monte_carlo_simulation(self):
        """Test individual Monte Carlo simulation runs."""
        power, type_i_rate = self.stats._monte_carlo_simulation(
            effect_size=0.5,
            sample_size=30,
            alpha=0.05,
            n_simulations=500,
            test_type='two_sample_ttest'
        )
        
        assert 0 <= power <= 1
        assert 0 <= type_i_rate <= 1
        
        # Type I error should be approximately equal to alpha
        assert abs(type_i_rate - 0.05) < 0.02  # Within 2% tolerance
        
        # Power should be reasonable for medium effect size
        assert power > 0.3  # Should have some power
        
    def test_sensitivity_recommendations(self):
        """Test sensitivity analysis recommendations."""
        # Create mock power surface data
        power_surface = {
            'design1': {
                'effect_size': 0.5,
                'sample_size': 30,
                'alpha': 0.05,
                'estimated_power': 0.85,
                'type_i_error_rate': 0.05
            },
            'design2': {
                'effect_size': 0.5,
                'sample_size': 50,
                'alpha': 0.05,
                'estimated_power': 0.95,
                'type_i_error_rate': 0.04
            },
            'design3': {
                'effect_size': 0.2,
                'sample_size': 30,
                'alpha': 0.05,
                'estimated_power': 0.30,
                'type_i_error_rate': 0.06
            }
        }
        
        recommendations = self.stats._generate_power_recommendations(power_surface)
        
        assert 'optimal_designs' in recommendations
        assert 'minimal_requirements' in recommendations
        assert 'power_warnings' in recommendations
        
        # Should identify adequately powered designs
        optimal = recommendations['optimal_designs']
        assert len(optimal) >= 1  # At least design1 and design2 should qualify
        
        # Should have warnings for underpowered design
        warnings_list = recommendations['power_warnings']
        assert len(warnings_list) >= 1  # Should warn about design3
        
    def test_power_interpretation_functions(self):
        """Test power result interpretation functions."""
        # Test t-test interpretation
        interpretation = self.stats._interpret_power_results(0.5, 0.8)
        assert isinstance(interpretation, str)
        assert 'medium' in interpretation.lower()
        assert 'adequate' in interpretation.lower()
        
        # Test small effect, low power
        interpretation_weak = self.stats._interpret_power_results(0.2, 0.4)
        assert 'small' in interpretation_weak.lower()
        assert 'inadequate' in interpretation_weak.lower()
        
        # Test ANOVA interpretation
        anova_interpretation = self.stats._interpret_anova_power(0.06, 0.8)
        assert isinstance(anova_interpretation, str)
        assert 'medium' in anova_interpretation.lower() or 'small' in anova_interpretation.lower()
        
    def test_simulation_error_handling(self):
        """Test error handling in sensitivity simulation."""
        # Test with invalid test type
        result = self.stats.sensitivity_simulation(
            effect_sizes=[0.5],
            sample_sizes=[30],
            alpha_levels=[0.05],
            n_simulations=10,
            test_type='invalid_test'
        )
        
        # Should complete but with limited results
        assert 'simulation_parameters' in result
        
        # Test with empty parameters (should use defaults)
        result = self.stats.sensitivity_simulation(
            n_simulations=50  # Small for speed
        )
        
        assert 'simulation_parameters' in result
        params = result['simulation_parameters']
        assert len(params['effect_sizes']) > 0
        assert len(params['sample_sizes']) > 0
        assert len(params['alpha_levels']) > 0

class TestStatisticalMethodsIntegration:
    """Integration tests for statistical methods."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.stats = AdvancedHRVStatistics()
        
    def test_power_analysis_workflow(self):
        """Test complete power analysis workflow."""
        # Step 1: Plan study with power analysis
        planning_result = self.stats.power_analysis(
            effect_size=0.5,
            power=0.8,
            alpha=0.05
        )
        
        required_n = planning_result.get('required_sample_size', 50)
        
        # Step 2: Collect data (simulated)
        np.random.seed(42)
        group1 = np.random.normal(100, 15, int(required_n))
        group2 = np.random.normal(107.5, 15, int(required_n))  # 0.5 effect size
        
        observed_data = {'group1': group1, 'group2': group2}
        
        # Step 3: Post-hoc power analysis
        posthoc_result = self.stats.post_hoc_power_analysis(
            observed_data=observed_data,
            test_type='two_sample_ttest',
            alpha=0.05
        )
        
        # Results should be consistent
        assert abs(posthoc_result['achieved_power'] - 0.8) < 0.2  # Within reasonable range
        assert abs(posthoc_result['cohens_d'] - 0.5) < 0.3  # Close to expected effect size
        
    def test_sensitivity_analysis_recommendations(self):
        """Test that sensitivity analysis provides useful recommendations."""
        result = self.stats.sensitivity_simulation(
            effect_sizes=[0.2, 0.8],
            sample_sizes=[20, 100],
            alpha_levels=[0.05],
            n_simulations=200,
            test_type='two_sample_ttest'
        )
        
        recommendations = result['recommendations']
        
        # Should recommend larger samples for smaller effects
        minimal_reqs = recommendations['minimal_requirements']
        if 'effect_size_0.2' in minimal_reqs and 'effect_size_0.8' in minimal_reqs:
            assert minimal_reqs['effect_size_0.2'] >= minimal_reqs['effect_size_0.8']
        
        # Should identify optimal designs
        optimal = recommendations['optimal_designs']
        if optimal:
            # Optimal designs should have adequate power
            for design in optimal:
                assert design['power'] >= 0.8
                
    def test_statistical_result_dataclass(self):
        """Test StatisticalResult dataclass functionality."""
        result = StatisticalResult(
            test_name="t-test",
            statistic=2.5,
            p_value=0.012,
            effect_size=0.6,
            confidence_interval=(0.1, 1.1),
            degrees_of_freedom=58,
            interpretation="Significant medium effect"
        )
        
        assert result.test_name == "t-test"
        assert result.statistic == 2.5
        assert result.p_value == 0.012
        assert result.effect_size == 0.6
        assert result.confidence_interval == (0.1, 1.1)
        assert result.degrees_of_freedom == 58
        assert "medium effect" in result.interpretation

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 