"""
Comprehensive tests for core HRV analysis functionality.

This module tests the fundamental data loading, signal processing,
and HRV computation components of the enhanced analysis system.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sqlite3
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(str(Path(__file__).parent.parent))

from core.data_loader import DataLoader
from core.signal_processing import SignalProcessor, ArtifactMethod, InterpolationMethod
from core.hrv_processor import HRVProcessor, HRVDomain

class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.data_loader = DataLoader()
        
    def test_create_sample_data(self):
        """Test sample data generation."""
        sample_data = DataLoader.create_sample_data(n_subjects=3, n_sols=4, samples_per_session=100)
        
        assert isinstance(sample_data, pd.DataFrame)
        assert len(sample_data) == 3 * 4 * 100
        assert 'heart_rate [bpm]' in sample_data.columns
        assert 'subject' in sample_data.columns
        assert 'Sol' in sample_data.columns
        
        # Check heart rate values are reasonable
        hr_values = sample_data['heart_rate [bpm]'].dropna()
        assert hr_values.min() >= 45
        assert hr_values.max() <= 120
        
    def test_data_validation(self):
        """Test data validation and cleaning."""
        # Create test data with issues
        test_data = pd.DataFrame({
            'heart_rate [bpm]': [70, 80, 300, np.nan, -10, 90, 85],  # Invalid values
            'subject': ['T01'] * 7,
            'Sol': [1] * 7
        })
        
        self.data_loader.validate_data = True
        cleaned_data = self.data_loader._validate_and_clean_data(test_data)
        
        # Check that invalid HR values were handled
        valid_hr = cleaned_data['heart_rate [bpm]'].dropna()
        assert all(hr >= 30 and hr <= 220 for hr in valid_hr)
        
    def test_quality_metrics_calculation(self):
        """Test data quality metrics calculation.""" 
        sample_data = DataLoader.create_sample_data(n_subjects=2, n_sols=3)
        
        # Load and validate data to trigger quality calculation
        self.data_loader._validate_and_clean_data(sample_data)
        
        assert hasattr(self.data_loader, 'data_quality_metrics')
        quality = self.data_loader.data_quality_metrics
        
        assert hasattr(quality, 'total_samples')
        assert hasattr(quality, 'valid_hr_samples')
        assert hasattr(quality, 'hr_quality_ratio')
        assert quality.hr_quality_ratio > 0.8  # Should be high quality for sample data
        
    def test_load_csv_nonexistent_directory(self):
        """Test loading from non-existent directory."""
        result = self.data_loader.load_csv_data(data_dir="/nonexistent/directory")
        assert result is None
        
    def test_quality_recommendations(self):
        """Test quality recommendation generation.""" 
        # Create low quality data
        low_quality_data = pd.DataFrame({
            'heart_rate [bpm]': [70, np.nan, np.nan, 80, np.nan] * 20
        })
        
        self.data_loader._validate_and_clean_data(low_quality_data) 
        report = self.data_loader.get_quality_report()
        
        assert 'recommendations' in report
        assert len(report['recommendations']) > 0

class TestSignalProcessor:
    """Test cases for SignalProcessor class."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.signal_processor = SignalProcessor()
        
    def test_rr_interval_conversion(self):
        """Test HR to RR interval conversion."""
        # Create realistic heart rate data
        hr_data = pd.Series([60, 70, 80, 65, 75])  # BPM
        
        rr_intervals, info = self.signal_processor.compute_rr_intervals(hr_data)
        
        assert len(rr_intervals) == 5
        assert all(300 <= rr <= 2000 for rr in rr_intervals)  # Physiological range
        
        # Check conversion accuracy (60 BPM = 1000 ms)
        expected_first = 60000 / 60
        assert abs(rr_intervals[0] - expected_first) < 1
        
    def test_artifact_detection_malik(self):
        """Test Malik artifact detection method."""
        # Create RR intervals with artifacts
        clean_rr = np.array([800, 820, 810, 815, 805])
        artifact_rr = np.array([800, 820, 1200, 815, 805])  # Artifact at index 2
        
        self.signal_processor.artifact_method = ArtifactMethod.MALIK
        
        clean_result, _ = self.signal_processor._detect_and_clean_artifacts(clean_rr)
        artifact_result, artifact_info = self.signal_processor._detect_and_clean_artifacts(artifact_rr)
        
        # Clean data should remain unchanged
        assert len(clean_result) == len(clean_rr)
        
        # Artifact should be detected and removed
        assert len(artifact_result) < len(artifact_rr)
        assert artifact_info['artifacts_detected'] > 0
        
    def test_interpolation_methods(self):
        """Test different interpolation methods."""
        rr_intervals = np.array([800, 820, 810, 815, 805, 825, 800])
        
        for method in [InterpolationMethod.LINEAR, InterpolationMethod.CUBIC]:
            self.signal_processor.interp_method = method
            
            time_vector, interpolated = self.signal_processor.interpolate_rr_series(
                rr_intervals, target_fs=4.0
            )
            
            assert len(time_vector) > 0
            assert len(interpolated) == len(time_vector)
            assert not np.any(np.isnan(interpolated))
            
    def test_signal_quality_assessment(self):
        """Test signal quality assessment."""
        # High quality signal
        good_rr = np.random.normal(800, 50, 200)  # Good variability
        good_rr = good_rr[(good_rr > 300) & (good_rr < 2000)]
        
        # Low quality signal  
        bad_rr = np.full(50, 800)  # No variability
        
        good_quality = self.signal_processor._assess_signal_quality(good_rr)
        bad_quality = self.signal_processor._assess_signal_quality(bad_rr)
        
        assert good_quality.quality_score > bad_quality.quality_score
        assert good_quality.rmssd > 0
        assert bad_quality.rmssd == 0
        
    def test_filter_application(self):
        """Test signal filtering methods."""
        # Create noisy RR intervals
        clean_signal = np.sin(np.linspace(0, 4*np.pi, 100)) * 50 + 800
        noise = np.random.normal(0, 10, 100)
        noisy_signal = clean_signal + noise
        
        # Apply filters
        filtered_savgol = self.signal_processor.apply_filter(
            noisy_signal, filter_type='savgol', window_length=9, polyorder=2
        )
        
        filtered_median = self.signal_processor.apply_filter(
            noisy_signal, filter_type='median', kernel_size=5
        )
        
        # Filtered signal should be smoother
        assert np.std(filtered_savgol) < np.std(noisy_signal)
        assert np.std(filtered_median) < np.std(noisy_signal)

class TestHRVProcessor:
    """Test cases for HRVProcessor class."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.hrv_processor = HRVProcessor()
        
    def test_time_domain_computation(self):
        """Test time domain HRV metrics computation."""
        # Create realistic RR intervals
        np.random.seed(42)
        rr_intervals = np.random.normal(800, 50, 200)
        rr_intervals = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
        
        time_domain = self.hrv_processor._compute_time_domain(rr_intervals)
        
        # Check that all expected metrics are computed
        assert hasattr(time_domain, 'mean_nni')
        assert hasattr(time_domain, 'sdnn')
        assert hasattr(time_domain, 'rmssd')
        assert hasattr(time_domain, 'pnn50')
        assert hasattr(time_domain, 'mean_hr')
        
        # Check reasonable values
        assert 300 < time_domain.mean_nni < 2000
        assert time_domain.sdnn > 0
        assert 30 < time_domain.mean_hr < 200
        
    def test_frequency_domain_computation(self):
        """Test frequency domain HRV metrics computation."""
        # Create RR intervals with known spectral content
        np.random.seed(42)
        time_points = np.linspace(0, 300, 600)  # 5 minutes at 2Hz
        
        # Add different frequency components
        lf_component = 20 * np.sin(2 * np.pi * 0.1 * time_points)  # 0.1 Hz (LF)
        hf_component = 10 * np.sin(2 * np.pi * 0.25 * time_points)  # 0.25 Hz (HF) 
        rr_intervals = 800 + lf_component + hf_component
        
        freq_domain = self.hrv_processor._compute_frequency_domain(rr_intervals)
        
        # Check that metrics are computed
        assert hasattr(freq_domain, 'lf_power')
        assert hasattr(freq_domain, 'hf_power')
        assert hasattr(freq_domain, 'lf_hf_ratio')
        assert hasattr(freq_domain, 'total_power')
        
        # Check that power values are reasonable
        assert freq_domain.lf_power > 0
        assert freq_domain.hf_power > 0
        assert freq_domain.total_power > 0
        
    def test_nonlinear_metrics_computation(self):
        """Test nonlinear HRV metrics computation."""
        # Create RR intervals with known nonlinear properties
        np.random.seed(42)
        rr_intervals = np.random.normal(800, 50, 200)
        
        nonlinear = self.hrv_processor._compute_nonlinear(rr_intervals)
        
        # Check Poincaré metrics
        assert hasattr(nonlinear, 'sd1')
        assert hasattr(nonlinear, 'sd2')
        assert hasattr(nonlinear, 'sd1_sd2_ratio')
        assert hasattr(nonlinear, 'ellipse_area')
        
        # Check that values are reasonable
        assert nonlinear.sd1 > 0
        assert nonlinear.sd2 > 0
        assert nonlinear.ellipse_area > 0
        
    def test_comprehensive_analysis(self):
        """Test comprehensive HRV analysis."""
        # Generate sample data
        np.random.seed(42)
        rr_intervals = np.random.normal(800, 50, 300)
        rr_intervals = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
        
        # Test all domains
        results = self.hrv_processor.compute_hrv_metrics(
            rr_intervals, 
            domains=[HRVDomain.TIME, HRVDomain.FREQUENCY, HRVDomain.NONLINEAR],
            include_confidence_intervals=True
        )
        
        # Check that all domains are present
        assert 'time_domain' in results
        assert 'frequency_domain' in results 
        assert 'nonlinear' in results
        assert 'confidence_intervals' in results
        assert 'quality_assessment' in results
        
        # Check quality assessment
        quality = results['quality_assessment']
        assert 'data_quality' in quality
        assert 'analysis_reliability' in quality
        
    def test_parasympathetic_indices(self):
        """Test parasympathetic indices computation."""
        np.random.seed(42)
        rr_intervals = np.random.normal(800, 50, 200)
        
        parasympathetic = self.hrv_processor._compute_parasympathetic_indices(rr_intervals)
        
        assert hasattr(parasympathetic, 'hf_power')
        assert hasattr(parasympathetic, 'rmssd')
        assert hasattr(parasympathetic, 'parasympathetic_index')
        
        # Index should be between 0 and 1
        assert 0 <= parasympathetic.parasympathetic_index <= 1
        
    def test_sympathetic_indices(self):
        """Test sympathetic indices computation."""
        np.random.seed(42)
        rr_intervals = np.random.normal(800, 30, 200)  # Lower variability
        
        sympathetic = self.hrv_processor._compute_sympathetic_indices(rr_intervals)
        
        assert hasattr(sympathetic, 'lf_power')
        assert hasattr(sympathetic, 'lf_hf_ratio')
        assert hasattr(sympathetic, 'sympathetic_index')
        assert hasattr(sympathetic, 'autonomic_balance')
        
        # Balance should be between -1 and 1
        assert -1 <= sympathetic.autonomic_balance <= 1
        
    def test_batch_processing(self):
        """Test batch processing of multiple datasets."""
        np.random.seed(42)
        
        # Create multiple datasets
        datasets = {}
        for i in range(3):
            rr_intervals = np.random.normal(800, 50, 200)
            datasets[f'subject_{i}'] = rr_intervals
            
        # Process in batch
        batch_results = self.hrv_processor.batch_process(
            datasets, domains=[HRVDomain.TIME]
        )
        
        assert len(batch_results) == 3
        for subject_id, results in batch_results.items():
            assert 'time_domain' in results
            
    def test_input_validation(self):
        """Test input validation and error handling."""
        # Empty array
        results_empty = self.hrv_processor.compute_hrv_metrics(np.array([]))
        assert 'error' in results_empty
        
        # Too few intervals
        results_few = self.hrv_processor.compute_hrv_metrics(np.array([800, 850]))
        assert 'error' in results_few or len(results_few.get('time_domain', {})) == 0
        
        # Invalid values
        invalid_rr = np.array([800, np.inf, -100, 5000])
        results_invalid = self.hrv_processor.compute_hrv_metrics(invalid_rr)
        # Should handle invalid values gracefully
        assert isinstance(results_invalid, dict)

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def setup_method(self):
        """Setup for integration tests."""
        self.data_loader = DataLoader()
        self.signal_processor = SignalProcessor()
        self.hrv_processor = HRVProcessor()
        
    def test_complete_analysis_workflow(self):
        """Test complete analysis from data loading to HRV computation."""
        # Generate sample data
        sample_data = DataLoader.create_sample_data(n_subjects=2, n_sols=3, samples_per_session=200)
        
        # Process first subject
        subject_data = sample_data[sample_data['subject'] == 'T01_Subject1']
        sol_data = subject_data[subject_data['Sol'] == 2]
        
        # Signal processing
        rr_intervals, processing_info = self.signal_processor.compute_rr_intervals(
            sol_data['heart_rate [bpm]']
        )
        
        assert len(rr_intervals) > 50
        assert processing_info['valid_hr_samples'] > 0
        
        # HRV analysis
        hrv_results = self.hrv_processor.compute_hrv_metrics(
            rr_intervals,
            domains=[HRVDomain.TIME, HRVDomain.FREQUENCY]
        )
        
        assert 'time_domain' in hrv_results
        assert 'frequency_domain' in hrv_results
        assert 'quality_assessment' in hrv_results
        
        # Check that analysis succeeded
        quality = hrv_results['quality_assessment']
        assert quality['data_quality'] in ['fair', 'good', 'excellent']
        
    def test_error_handling_pipeline(self):
        """Test error handling throughout the analysis pipeline."""
        # Create problematic data
        bad_data = pd.DataFrame({
            'heart_rate [bpm]': [np.nan] * 100,  # All NaN values
            'subject': ['T01'] * 100,
            'Sol': [1] * 100
        })
        
        # Process through pipeline
        rr_intervals, _ = self.signal_processor.compute_rr_intervals(bad_data['heart_rate [bpm]'])
        assert len(rr_intervals) == 0
        
        # HRV analysis should handle empty input gracefully
        hrv_results = self.hrv_processor.compute_hrv_metrics(rr_intervals)
        assert 'error' in hrv_results or len(hrv_results) == 0

# Fixtures for test data
@pytest.fixture
def sample_rr_intervals():
    """Generate sample RR intervals for testing."""
    np.random.seed(42)
    return np.random.normal(800, 50, 300)

@pytest.fixture  
def sample_hr_data():
    """Generate sample heart rate data for testing."""
    return DataLoader.create_sample_data(n_subjects=1, n_sols=2, samples_per_session=100)

if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"]) 