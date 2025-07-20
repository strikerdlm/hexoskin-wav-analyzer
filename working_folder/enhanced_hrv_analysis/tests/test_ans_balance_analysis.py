"""
Comprehensive tests for ANS Balance Analysis functionality.

This module tests the enhanced autonomic nervous system balance analysis
including parasympathetic, sympathetic metrics, and bootstrap confidence intervals.
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

from core.hrv_processor import (
    HRVProcessor, HRVDomain, 
    ParasympatheticMetrics, SympatheticMetrics, ANSBalanceMetrics
)

class TestEnhancedParasympatheticAnalysis:
    """Test cases for enhanced parasympathetic analysis."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.processor = HRVProcessor()
        # Generate realistic RR intervals (600-1200ms, ~60-100 BPM)
        np.random.seed(42)
        self.rr_intervals = np.random.normal(800, 50, 300)  # 300 intervals, ~4 min recording
        
    def test_parasympathetic_metrics_structure(self):
        """Test that parasympathetic metrics are computed with correct structure."""
        metrics = self.processor._compute_parasympathetic_indices(self.rr_intervals)
        
        assert isinstance(metrics, ParasympatheticMetrics)
        
        # Check all fields are present
        assert hasattr(metrics, 'hf_power')
        assert hasattr(metrics, 'rmssd')
        assert hasattr(metrics, 'pnn50')
        assert hasattr(metrics, 'sd1')
        assert hasattr(metrics, 'hf_nu')
        assert hasattr(metrics, 'parasympathetic_index')
        assert hasattr(metrics, 'rsa_amplitude')
        assert hasattr(metrics, 'respiratory_frequency')
        
        # Enhanced metrics
        assert hasattr(metrics, 'hf_rmssd_ratio')
        assert hasattr(metrics, 'rsa_coupling_index')
        assert hasattr(metrics, 'vagal_tone_index')
        assert hasattr(metrics, 'respiratory_coherence')
        
    def test_parasympathetic_values_realistic(self):
        """Test that parasympathetic values are in realistic ranges."""
        metrics = self.processor._compute_parasympathetic_indices(self.rr_intervals)
        
        # Basic checks for reasonable ranges
        assert metrics.hf_power >= 0
        assert metrics.rmssd >= 0
        assert 0 <= metrics.pnn50 <= 100
        assert metrics.sd1 >= 0
        assert 0 <= metrics.hf_nu <= 100
        assert 0 <= metrics.parasympathetic_index <= 1
        
        # Enhanced metrics ranges
        assert metrics.hf_rmssd_ratio >= 0
        assert -1 <= metrics.rsa_coupling_index <= 1  # Correlation coefficient
        assert 0 <= metrics.vagal_tone_index <= 1
        assert 0 <= metrics.respiratory_coherence <= 1
        
    def test_rsa_coupling_computation(self):
        """Test RSA coupling index computation."""
        # Test with very short data (should return 0)
        short_rr = np.array([800, 820, 810, 830])
        coupling = self.processor._compute_rsa_coupling(short_rr)
        assert coupling == 0.0
        
        # Test with normal length data
        coupling = self.processor._compute_rsa_coupling(self.rr_intervals)
        assert -1 <= coupling <= 1  # Should be a correlation coefficient
        
    def test_vagal_tone_computation(self):
        """Test vagal tone index computation."""
        # Test with realistic values
        vagal_tone = self.processor._compute_vagal_tone(
            hf_nu=30.0, rmssd=40.0, pnn50=15.0, sd1=25.0
        )
        assert 0 <= vagal_tone <= 1
        
        # Test with extreme values
        vagal_tone_high = self.processor._compute_vagal_tone(
            hf_nu=80.0, rmssd=100.0, pnn50=50.0, sd1=50.0
        )
        assert vagal_tone_high > vagal_tone  # Should be higher
        
    def test_respiratory_coherence_computation(self):
        """Test respiratory coherence computation."""
        # Test with zero HF peak
        coherence = self.processor._compute_respiratory_coherence(self.rr_intervals, 0.0)
        assert coherence == 0.0
        
        # Test with realistic HF peak
        coherence = self.processor._compute_respiratory_coherence(self.rr_intervals, 0.25)
        assert 0 <= coherence <= 1

class TestEnhancedSympatheticAnalysis:
    """Test cases for enhanced sympathetic analysis."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.processor = HRVProcessor()
        np.random.seed(42)
        self.rr_intervals = np.random.normal(700, 80, 300)  # More variable, sympathetic-like
        
    def test_sympathetic_metrics_structure(self):
        """Test that sympathetic metrics are computed with correct structure."""
        metrics = self.processor._compute_sympathetic_indices(self.rr_intervals)
        
        assert isinstance(metrics, SympatheticMetrics)
        
        # Check all fields are present
        assert hasattr(metrics, 'lf_power')
        assert hasattr(metrics, 'lf_nu')
        assert hasattr(metrics, 'lf_hf_ratio')
        assert hasattr(metrics, 'stress_index')
        assert hasattr(metrics, 'sympathetic_index')
        assert hasattr(metrics, 'autonomic_balance')
        assert hasattr(metrics, 'sympathovagal_balance')
        
        # Enhanced metrics
        assert hasattr(metrics, 'cardiac_sympathetic_index')
        assert hasattr(metrics, 'sympathetic_modulation')
        assert hasattr(metrics, 'beta_adrenergic_sensitivity')
        
    def test_sympathetic_values_realistic(self):
        """Test that sympathetic values are in realistic ranges."""
        metrics = self.processor._compute_sympathetic_indices(self.rr_intervals)
        
        # Basic checks
        assert metrics.lf_power >= 0
        assert 0 <= metrics.lf_nu <= 100
        assert metrics.lf_hf_ratio >= 0
        assert metrics.stress_index >= 0
        assert 0 <= metrics.sympathetic_index <= 1
        assert -1 <= metrics.autonomic_balance <= 1
        assert metrics.sympathovagal_balance >= 0
        
        # Enhanced metrics
        assert 0 <= metrics.cardiac_sympathetic_index <= 1
        assert 0 <= metrics.sympathetic_modulation <= 1
        assert 0 <= metrics.beta_adrenergic_sensitivity <= 1
        
    def test_cardiac_sympathetic_index_computation(self):
        """Test cardiac sympathetic index computation."""
        # Test with low sympathetic activity
        csi_low = self.processor._compute_cardiac_sympathetic_index(
            lf_nu=20.0, stress_index=10.0, mean_hr=65.0
        )
        
        # Test with high sympathetic activity
        csi_high = self.processor._compute_cardiac_sympathetic_index(
            lf_nu=70.0, stress_index=50.0, mean_hr=90.0
        )
        
        assert 0 <= csi_low <= 1
        assert 0 <= csi_high <= 1
        assert csi_high > csi_low  # Should be higher with higher input values
        
    def test_sympathetic_modulation_computation(self):
        """Test sympathetic modulation computation."""
        # Test with zero SDNN (should return 0)
        modulation = self.processor._compute_sympathetic_modulation(100.0, 0.0)
        assert modulation == 0.0
        
        # Test with normal values
        modulation = self.processor._compute_sympathetic_modulation(500.0, 50.0)
        assert 0 <= modulation <= 1
        
    def test_beta_adrenergic_sensitivity_computation(self):
        """Test beta-adrenergic sensitivity computation."""
        # Test with very short data
        short_rr = np.array([700, 720, 710, 730])
        sensitivity = self.processor._compute_beta_adrenergic_sensitivity(short_rr)
        assert sensitivity == 0.0
        
        # Test with normal data
        sensitivity = self.processor._compute_beta_adrenergic_sensitivity(self.rr_intervals)
        assert 0 <= sensitivity <= 1

class TestANSBalanceAnalysis:
    """Test cases for comprehensive ANS balance analysis."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.processor = HRVProcessor()
        np.random.seed(42)
        self.rr_intervals = np.random.normal(750, 60, 400)  # Balanced ANS state
        
    def test_ans_balance_metrics_structure(self):
        """Test that ANS balance metrics are computed with correct structure."""
        metrics = self.processor._compute_ans_balance_metrics(self.rr_intervals)
        
        assert isinstance(metrics, ANSBalanceMetrics)
        
        # Check all fields are present
        assert hasattr(metrics, 'lf_hf_ratio')
        assert hasattr(metrics, 'autonomic_balance')
        assert hasattr(metrics, 'sympathovagal_index')
        assert hasattr(metrics, 'ans_complexity')
        assert hasattr(metrics, 'cardiac_autonomic_balance')
        assert hasattr(metrics, 'autonomic_reactivity')
        assert hasattr(metrics, 'baroreflex_sensitivity')
        
        # Confidence intervals
        assert hasattr(metrics, 'lf_hf_ratio_ci')
        assert hasattr(metrics, 'sympathovagal_index_ci')
        assert hasattr(metrics, 'ans_complexity_ci')
        
    def test_ans_balance_values_realistic(self):
        """Test that ANS balance values are in realistic ranges."""
        metrics = self.processor._compute_ans_balance_metrics(self.rr_intervals)
        
        assert metrics.lf_hf_ratio >= 0
        assert -1 <= metrics.autonomic_balance <= 1
        assert 0 <= metrics.sympathovagal_index <= 1
        assert metrics.ans_complexity >= 0
        assert 0 <= metrics.cardiac_autonomic_balance <= 1
        assert 0 <= metrics.autonomic_reactivity <= 1
        assert 0 <= metrics.baroreflex_sensitivity <= 20  # Typical BRS range
        
        # Check CI structure
        assert isinstance(metrics.lf_hf_ratio_ci, tuple)
        assert len(metrics.lf_hf_ratio_ci) == 2
        assert isinstance(metrics.sympathovagal_index_ci, tuple)
        assert len(metrics.sympathovagal_index_ci) == 2
        assert isinstance(metrics.ans_complexity_ci, tuple)
        assert len(metrics.ans_complexity_ci) == 2
        
    def test_sympathovagal_index_computation(self):
        """Test sympathovagal index computation."""
        # Test with balanced state
        index_balanced = self.processor._compute_sympathovagal_index(
            lf_nu=50.0, hf_nu=50.0, stress_index=20.0, parasympathetic_index=0.5
        )
        
        # Test with sympathetic dominance
        index_sympathetic = self.processor._compute_sympathovagal_index(
            lf_nu=80.0, hf_nu=20.0, stress_index=60.0, parasympathetic_index=0.2
        )
        
        assert 0 <= index_balanced <= 1
        assert 0 <= index_sympathetic <= 1
        assert index_sympathetic > index_balanced  # Should be higher
        
    def test_ans_complexity_computation(self):
        """Test ANS complexity computation."""
        # Test with very short data
        short_rr = np.random.normal(750, 30, 50)
        complexity = self.processor._compute_ans_complexity(short_rr)
        assert complexity == 0.0
        
        # Test with normal length data
        complexity = self.processor._compute_ans_complexity(self.rr_intervals)
        assert complexity >= 0
        
    def test_cardiac_autonomic_balance_computation(self):
        """Test cardiac autonomic balance computation."""
        # Test with equal indices
        balance = self.processor._compute_cardiac_autonomic_balance(0.5, 0.5)
        assert balance == 0.5
        
        # Test with parasympathetic dominance
        balance_para = self.processor._compute_cardiac_autonomic_balance(0.8, 0.2)
        assert balance_para == 0.8
        
        # Test with sympathetic dominance
        balance_sympa = self.processor._compute_cardiac_autonomic_balance(0.2, 0.8)
        assert balance_sympa == 0.2
        
        # Test with zero activity
        balance_zero = self.processor._compute_cardiac_autonomic_balance(0.0, 0.0)
        assert balance_zero == 0.5  # Should default to neutral
        
    def test_autonomic_reactivity_computation(self):
        """Test autonomic reactivity computation."""
        # Test with very short data
        short_rr = np.random.normal(750, 30, 50)
        reactivity = self.processor._compute_autonomic_reactivity(short_rr)
        assert reactivity == 0.0
        
        # Test with normal data
        reactivity = self.processor._compute_autonomic_reactivity(self.rr_intervals)
        assert 0 <= reactivity <= 1
        
    def test_baroreflex_sensitivity_estimation(self):
        """Test baroreflex sensitivity estimation."""
        # Test with very short data
        short_rr = np.random.normal(750, 30, 50)
        brs = self.processor._estimate_baroreflex_sensitivity(short_rr)
        assert brs == 0.0
        
        # Test with normal data
        brs = self.processor._estimate_baroreflex_sensitivity(self.rr_intervals)
        assert 0 <= brs <= 20  # Typical BRS range
        
    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence intervals computation."""
        # Set small n_bootstrap for testing
        with patch.object(self.processor, 'confidence_level', 0.95):
            lf_hf_ci, sympathovagal_ci, complexity_ci = self.processor._bootstrap_ans_confidence_intervals(
                self.rr_intervals
            )
            
            # Check CI structure
            assert isinstance(lf_hf_ci, tuple)
            assert len(lf_hf_ci) == 2
            assert lf_hf_ci[0] <= lf_hf_ci[1]  # Lower <= Upper
            
            assert isinstance(sympathovagal_ci, tuple)
            assert len(sympathovagal_ci) == 2
            assert sympathovagal_ci[0] <= sympathovagal_ci[1]
            
            assert isinstance(complexity_ci, tuple)
            assert len(complexity_ci) == 2
            assert complexity_ci[0] <= complexity_ci[1]
        
        # Test with insufficient data
        short_rr = np.random.normal(750, 30, 30)
        lf_hf_ci, sympathovagal_ci, complexity_ci = self.processor._bootstrap_ans_confidence_intervals(
            short_rr
        )
        assert lf_hf_ci == (0.0, 0.0)
        assert sympathovagal_ci == (0.0, 0.0)
        assert complexity_ci == (0.0, 0.0)

class TestANSBalanceIntegration:
    """Integration tests for ANS balance analysis."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.processor = HRVProcessor()
        np.random.seed(42)
        
    def test_ans_balance_domain_processing(self):
        """Test ANS balance analysis through main processing pipeline."""
        rr_intervals = np.random.normal(800, 50, 300)
        
        # Test ANS balance domain specifically
        result = self.processor.compute_hrv_metrics(
            rr_intervals, 
            domains=[HRVDomain.ANS_BALANCE],
            include_confidence_intervals=False
        )
        
        assert 'ans_balance' in result
        ans_metrics = result['ans_balance']
        
        # Check key metrics are present
        assert 'lf_hf_ratio' in ans_metrics
        assert 'sympathovagal_index' in ans_metrics
        assert 'ans_complexity' in ans_metrics
        assert 'lf_hf_ratio_ci' in ans_metrics
        
    def test_comprehensive_analysis_with_ans_balance(self):
        """Test comprehensive analysis including ANS balance."""
        rr_intervals = np.random.normal(750, 60, 400)
        
        result = self.processor.compute_hrv_metrics(
            rr_intervals,
            domains=[HRVDomain.ALL],
            include_confidence_intervals=True
        )
        
        # Check all domains are present
        expected_domains = ['time_domain', 'frequency_domain', 'nonlinear', 
                          'parasympathetic', 'sympathetic', 'ans_balance']
        
        for domain in expected_domains:
            assert domain in result, f"Missing domain: {domain}"
            
        # Check enhanced metrics in parasympathetic
        parasympathetic = result['parasympathetic']
        assert 'hf_rmssd_ratio' in parasympathetic
        assert 'vagal_tone_index' in parasympathetic
        assert 'rsa_coupling_index' in parasympathetic
        
        # Check enhanced metrics in sympathetic
        sympathetic = result['sympathetic']
        assert 'cardiac_sympathetic_index' in sympathetic
        assert 'sympathetic_modulation' in sympathetic
        assert 'beta_adrenergic_sensitivity' in sympathetic
        
        # Check ANS balance metrics
        ans_balance = result['ans_balance']
        assert 'cardiac_autonomic_balance' in ans_balance
        assert 'autonomic_reactivity' in ans_balance
        assert 'baroreflex_sensitivity' in ans_balance
        
    def test_parallel_processing_with_ans_balance(self):
        """Test parallel processing includes ANS balance analysis."""
        rr_intervals = np.random.normal(800, 50, 300)
        
        # Enable parallel processing
        self.processor.parallel_processing = True
        self.processor.n_jobs = 2
        
        result = self.processor.compute_hrv_metrics(
            rr_intervals,
            domains=[HRVDomain.TIME, HRVDomain.FREQUENCY, HRVDomain.ANS_BALANCE]
        )
        
        assert 'ans_balance' in result
        assert 'time_domain' in result
        assert 'frequency_domain' in result
        
    def test_error_handling_in_ans_balance(self):
        """Test error handling in ANS balance analysis."""
        # Test with empty array
        empty_rr = np.array([])
        with pytest.raises(Exception):
            self.processor._compute_ans_balance_metrics(empty_rr)
        
        # Test with very short array (should not raise exception but return zeros)
        short_rr = np.array([800, 820])
        try:
            metrics = self.processor._compute_ans_balance_metrics(short_rr)
            # Should complete without error, but values should be mostly zero/default
            assert isinstance(metrics, ANSBalanceMetrics)
        except Exception as e:
            pytest.fail(f"Should handle short arrays gracefully, but got: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 