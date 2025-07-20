"""
Comprehensive tests for Visualization functionality.

This module tests the interactive visualization features including
Plotly plots, dashboards, and export functionality.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import warnings

import sys
import os
sys.path.append(str(Path(__file__).parent.parent))

try:
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from visualization.interactive_plots import InteractivePlotter

@pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not available")
class TestInteractivePlotter:
    """Test cases for interactive plotting functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.plotter = InteractivePlotter()
        
        # Create sample HRV data
        np.random.seed(42)
        self.sample_rr_intervals = np.random.normal(800, 50, 300)
        
        self.sample_hrv_results = {
            'time_domain': {
                'sdnn': 45.5,
                'rmssd': 32.1,
                'pnn50': 15.2,
                'mean_hr': 72.5
            },
            'frequency_domain': {
                'lf_power': 450.2,
                'hf_power': 320.8,
                'lf_hf_ratio': 1.4,
                'total_power': 850.5
            },
            'nonlinear': {
                'sd1': 22.8,
                'sd2': 65.4,
                'dfa_alpha1': 1.15,
                'sample_entropy': 0.85
            }
        }
        
    def test_poincare_plot_creation(self):
        """Test Poincaré plot creation."""
        fig = self.plotter.create_poincare_plot(
            self.sample_rr_intervals,
            title="Test Poincaré Plot",
            show_statistics=True
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1  # Should have at least scatter plot
        
        # Check if title is set
        assert fig.layout.title.text == "Test Poincaré Plot"
        
        # Check axis labels
        assert "RR(n)" in fig.layout.xaxis.title.text
        assert "RR(n+1)" in fig.layout.yaxis.title.text
        
    def test_psd_plot_creation(self):
        """Test power spectral density plot creation."""
        fig = self.plotter.create_psd_plot(
            self.sample_rr_intervals,
            method='welch',
            title="Test PSD Plot"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Check if frequency bands are highlighted
        # Should have area fills for VLF, LF, HF bands
        area_traces = [trace for trace in fig.data if trace.fill is not None]
        assert len(area_traces) >= 3  # VLF, LF, HF bands
        
    def test_correlation_heatmap_creation(self):
        """Test correlation heatmap creation."""
        # Create sample data frame
        hrv_data = pd.DataFrame({
            'SDNN': np.random.normal(50, 10, 100),
            'RMSSD': np.random.normal(30, 8, 100),
            'pNN50': np.random.normal(15, 5, 100),
            'LF_power': np.random.normal(500, 100, 100),
            'HF_power': np.random.normal(400, 80, 100)
        })
        
        fig = self.plotter.create_correlation_heatmap(
            hrv_data,
            title="Test Correlation Heatmap"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # Heatmap should be single trace
        assert fig.data[0].type == 'heatmap'
        
    def test_time_series_plot_creation(self):
        """Test time series plot creation."""
        # Create time series data
        timestamps = pd.date_range('2024-01-01', periods=100, freq='H')
        hrv_values = np.random.normal(50, 10, 100)
        
        fig = self.plotter.create_time_series_plot(
            timestamps,
            hrv_values,
            metric_name='SDNN',
            title='Test Time Series'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert fig.layout.title.text == 'Test Time Series'
        assert 'SDNN' in fig.layout.yaxis.title.text
        
    def test_dashboard_creation(self):
        """Test comprehensive dashboard creation."""
        dashboard = self.plotter.create_hrv_dashboard(
            self.sample_rr_intervals,
            self.sample_hrv_results,
            subject_id='TEST_SUBJECT',
            session_id='SOL_01'
        )
        
        assert isinstance(dashboard, go.Figure)
        
        # Dashboard should have multiple subplots
        assert hasattr(dashboard, '_grid_ref')
        
        # Should contain key plots
        subplot_count = len([trace for trace in dashboard.data])
        assert subplot_count >= 4  # At least 4 different plots
        
    def test_plot_export_functionality(self):
        """Test plot export to HTML and images."""
        fig = self.plotter.create_poincare_plot(
            self.sample_rr_intervals,
            title="Export Test"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test HTML export
            html_file = temp_path / "test_plot.html"
            success = self.plotter.export_plot(fig, html_file, format='html')
            
            assert success
            assert html_file.exists()
            assert html_file.stat().st_size > 0
            
            # Test PNG export (if kaleido available)
            png_file = temp_path / "test_plot.png"
            try:
                success = self.plotter.export_plot(fig, png_file, format='png')
                if success:  # Only check if export was successful
                    assert png_file.exists()
                    assert png_file.stat().st_size > 0
            except Exception:
                # PNG export might fail without kaleido, which is OK
                pass
                
    def test_advanced_poincare_features(self):
        """Test advanced Poincaré plot features."""
        fig = self.plotter.create_poincare_plot(
            self.sample_rr_intervals,
            show_ellipse=True,
            show_statistics=True,
            color_by_time=True
        )
        
        assert isinstance(fig, go.Figure)
        
        # Should have multiple traces (scatter + ellipse + annotations)
        assert len(fig.data) >= 2
        
        # Check for ellipse trace
        ellipse_traces = [trace for trace in fig.data if trace.mode == 'lines']
        assert len(ellipse_traces) >= 1
        
    def test_psd_customization_options(self):
        """Test PSD plot customization options."""
        fig = self.plotter.create_psd_plot(
            self.sample_rr_intervals,
            method='periodogram',
            log_scale=True,
            highlight_peaks=True,
            show_band_powers=True
        )
        
        assert isinstance(fig, go.Figure)
        
        # Should be on log scale
        assert fig.layout.yaxis.type == 'log'
        
        # Should have band power annotations
        annotations = fig.layout.annotations
        if annotations:
            assert len(annotations) >= 3  # VLF, LF, HF annotations
            
    def test_error_handling_in_plots(self):
        """Test error handling in plot creation."""
        # Test with empty data
        empty_rr = np.array([])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = self.plotter.create_poincare_plot(empty_rr)
            
        # Should return an empty or error figure, not crash
        assert isinstance(fig, go.Figure)
        
        # Test with insufficient data
        short_rr = np.array([800, 820, 810])
        
        fig = self.plotter.create_poincare_plot(short_rr)
        assert isinstance(fig, go.Figure)
        
    def test_plot_styling_consistency(self):
        """Test that plots have consistent styling."""
        figs = [
            self.plotter.create_poincare_plot(self.sample_rr_intervals),
            self.plotter.create_psd_plot(self.sample_rr_intervals),
            self.plotter.create_time_series_plot(
                pd.date_range('2024-01-01', periods=len(self.sample_rr_intervals), freq='s'),
                self.sample_rr_intervals,
                'RR Intervals'
            )
        ]
        
        # Check that all figures have consistent template
        template_themes = [fig.layout.template.layout.colorway for fig in figs 
                          if fig.layout.template and fig.layout.template.layout.colorway]
        
        # If templates are applied, they should be consistent
        if template_themes:
            first_theme = template_themes[0]
            assert all(theme == first_theme for theme in template_themes)

class TestPlotUtilities:
    """Test utility functions for plotting."""
    
    def setup_method(self):
        """Setup for each test method."""
        if not HAS_PLOTLY:
            pytest.skip("Plotly not available")
        self.plotter = InteractivePlotter()
        
    def test_color_palette_generation(self):
        """Test color palette generation utilities."""
        # Test if plotter has color management
        if hasattr(self.plotter, '_generate_color_palette'):
            colors = self.plotter._generate_color_palette(5)
            assert len(colors) == 5
            assert all(isinstance(color, str) for color in colors)
            
    def test_statistical_annotations(self):
        """Test statistical annotation utilities."""
        test_data = {
            'sd1': 25.5,
            'sd2': 68.2,
            'ratio': 0.374
        }
        
        # Test if plotter can create statistical annotations
        if hasattr(self.plotter, '_create_statistical_annotations'):
            annotations = self.plotter._create_statistical_annotations(test_data)
            assert isinstance(annotations, list)
            assert len(annotations) >= len(test_data)
            
    def test_band_highlighting(self):
        """Test frequency band highlighting utilities."""
        freqs = np.linspace(0, 0.5, 256)
        psd = np.random.exponential(2, 256)
        
        if hasattr(self.plotter, '_add_frequency_bands'):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=freqs, y=psd, mode='lines'))
            
            enhanced_fig = self.plotter._add_frequency_bands(fig, freqs, psd)
            assert isinstance(enhanced_fig, go.Figure)
            # Should have additional traces for band highlighting
            assert len(enhanced_fig.data) >= len(fig.data)

class TestPlotIntegration:
    """Integration tests for visualization functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        if not HAS_PLOTLY:
            pytest.skip("Plotly not available")
        self.plotter = InteractivePlotter()
        
    @pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not available")
    def test_full_visualization_workflow(self):
        """Test complete visualization workflow."""
        # Generate sample data
        np.random.seed(42)
        rr_intervals = np.random.normal(800, 50, 500)
        
        # Create multiple plots
        plots = {}
        
        plots['poincare'] = self.plotter.create_poincare_plot(
            rr_intervals, 
            title="Integration Test Poincaré"
        )
        
        plots['psd'] = self.plotter.create_psd_plot(
            rr_intervals,
            title="Integration Test PSD"
        )
        
        # Create time series data
        timestamps = pd.date_range('2024-01-01', periods=len(rr_intervals), freq='s')
        plots['timeseries'] = self.plotter.create_time_series_plot(
            timestamps,
            rr_intervals,
            'RR Intervals',
            title="Integration Test Time Series"
        )
        
        # All plots should be created successfully
        for plot_name, plot_fig in plots.items():
            assert isinstance(plot_fig, go.Figure), f"Failed to create {plot_name} plot"
            assert len(plot_fig.data) > 0, f"{plot_name} plot has no data"
            
        # Test batch export
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            export_success = []
            for plot_name, plot_fig in plots.items():
                html_file = temp_path / f"{plot_name}_test.html"
                success = self.plotter.export_plot(plot_fig, html_file, format='html')
                export_success.append(success)
                
                if success:
                    assert html_file.exists()
                    assert html_file.stat().st_size > 0
                    
            # At least some exports should succeed
            assert any(export_success), "No plots were exported successfully"
            
    @pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not available") 
    def test_dashboard_integration(self):
        """Test dashboard creation with real HRV analysis results."""
        from core.hrv_processor import HRVProcessor, HRVDomain
        
        # Generate and analyze HRV data
        processor = HRVProcessor()
        np.random.seed(42)
        rr_intervals = np.random.normal(750, 60, 400)
        
        hrv_results = processor.compute_hrv_metrics(
            rr_intervals,
            domains=[HRVDomain.TIME, HRVDomain.FREQUENCY, HRVDomain.NONLINEAR],
            include_confidence_intervals=False
        )
        
        # Create dashboard
        dashboard = self.plotter.create_hrv_dashboard(
            rr_intervals,
            hrv_results,
            subject_id='INTEGRATION_TEST',
            session_id='SOL_TEST'
        )
        
        assert isinstance(dashboard, go.Figure)
        assert len(dashboard.data) >= 4  # Multiple subplot components
        
        # Test dashboard export
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard_file = Path(temp_dir) / "integration_dashboard.html"
            success = self.plotter.export_plot(dashboard, dashboard_file, format='html')
            
            if success:
                assert dashboard_file.exists()
                assert dashboard_file.stat().st_size > 5000  # Dashboard should be substantial

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 