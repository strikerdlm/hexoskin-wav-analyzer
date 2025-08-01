"""
Interactive HRV Visualization with Plotly

This module provides interactive plotting capabilities for HRV analysis including:
- Interactive Poincaré plots with ellipse fitting
- Power spectral density plots with frequency band highlighting
- Correlation heatmaps with statistical significance
- Time series plots with trend analysis
- Multi-dimensional analysis dashboards
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from scipy import signal, stats
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import json

# Import HRV reference ranges
try:
    from visualization.hrv_reference_ranges import hrv_reference_ranges, assess_hrv_value
except ImportError:
    try:
        from enhanced_hrv_analysis.visualization.hrv_reference_ranges import hrv_reference_ranges, assess_hrv_value
    except ImportError:
        hrv_reference_ranges = None
        assess_hrv_value = None

# Enhanced statistical modeling for GAM analysis
try:
    from statsmodels.gam.api import GLMGam, BSplines
    from statsmodels.genmod.families import Gaussian
    import statsmodels.api as sm
    GAM_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("GAM analysis capabilities enabled with statsmodels")
except ImportError:
    GAM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("GAM analysis not available - install statsmodels for advanced trend analysis")

logger = logging.getLogger(__name__)

@dataclass
class PlotConfig:
    """Configuration for plot styling and behavior."""
    color_palette: List[str]
    template: str
    width: int
    height: int
    font_size: int
    show_toolbar: bool
    export_format: str

class InteractivePlotter:
    """Enhanced interactive plotter using Plotly for HRV analysis."""
    
    def __init__(self, 
                 template: str = "plotly_white",
                 color_palette: List[str] = None,
                 default_width: str = "100%",
                 default_height: str = "100vh",
                 font_size: int = 12,
                 responsive: bool = True):
        """
        Initialize interactive plotter.
        
        Args:
            template: Plotly template name
            color_palette: Custom color palette
            default_width: Default plot width (responsive: "100%")
            default_height: Default plot height (responsive: "100vh")
            font_size: Default font size
            responsive: Enable responsive full-screen layout
        """
        self.template = template
        self.color_palette = color_palette or [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        self.default_width = default_width
        self.default_height = default_height
        self.font_size = font_size
        self.responsive = responsive
        
        # Set up plotly configuration for full-screen responsive plots
        pyo.init_notebook_mode(connected=True)
    
    def _get_responsive_layout(self, width: Union[str, int] = None, height: Union[str, int] = None) -> Dict[str, Any]:
        """Get responsive layout configuration for full-screen plots."""
        layout_config = {
            'template': self.template,
            'font': dict(size=self.font_size),
            'margin': dict(l=50, r=50, t=80, b=50),
            'showlegend': True,
            'autosize': True if self.responsive else False,
        }
        
        if self.responsive:
            # Full-screen responsive configuration - let HTML/CSS handle sizing
            layout_config.update({
                'autosize': True,
                'margin': dict(l=40, r=40, t=60, b=40),  # Tighter margins for full screen
                # Don't set width/height - let the responsive HTML template handle it
            })
        else:
            # Fixed size configuration with numeric values only
            fixed_width = 800
            fixed_height = 600
            
            if width is not None and isinstance(width, (int, float)):
                fixed_width = width
            if height is not None and isinstance(height, (int, float)):
                fixed_height = height
                
            layout_config.update({
                'width': fixed_width,
                'height': fixed_height,
                'autosize': False,
            })
        
        return layout_config
    
    def create_poincare_plot(self,
                           rr_intervals: np.ndarray,
                           title: str = "",
                           show_ellipse: bool = True,
                           show_grid: bool = True,
                           color_by_time: bool = False) -> go.Figure:
        """
        Create interactive Poincaré plot with ellipse fitting.
        
        Args:
            rr_intervals: RR intervals in milliseconds
            title: Plot title
            show_ellipse: Whether to show fitted ellipse
            show_grid: Whether to show grid lines
            color_by_time: Whether to color points by time progression
            
        Returns:
            Plotly figure object
        """
        try:
            if len(rr_intervals) < 2:
                logger.error("Insufficient data for Poincaré plot")
                return self._create_error_figure("Insufficient data for Poincaré plot")
                
            # Create Poincaré vectors
            rr1 = rr_intervals[:-1]
            rr2 = rr_intervals[1:]
            
            # Create hover text with indices
            hover_text = [f"Beat {i}<br>RR(n): {rr1[i]:.1f} ms<br>RR(n+1): {rr2[i]:.1f} ms" 
                         for i in range(len(rr1))]
            
            # Create figure
            fig = go.Figure()
            
            # Color mapping
            if color_by_time:
                colors = np.arange(len(rr1))
                colorscale = 'Viridis'
                colorbar_title = "Time Progression"
            else:
                colors = self.color_palette[0]
                colorscale = None
                colorbar_title = None
                
            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=rr1,
                y=rr2,
                mode='markers',
                marker=dict(
                    color=colors,
                    size=6,
                    opacity=0.7,
                    colorscale=colorscale,
                    colorbar=dict(title=colorbar_title) if color_by_time else None,
                    line=dict(width=0.5, color='white')
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='RR Intervals'
            ))
            
            # Add identity line (y = x)
            min_rr = min(np.min(rr1), np.min(rr2))
            max_rr = max(np.max(rr1), np.max(rr2))
            
            fig.add_trace(go.Scatter(
                x=[min_rr, max_rr],
                y=[min_rr, max_rr],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                name='Identity Line',
                hovertemplate='Identity Line<extra></extra>'
            ))
            
            # Add fitted ellipse if requested
            if show_ellipse:
                ellipse_x, ellipse_y, sd1, sd2 = self._calculate_poincare_ellipse(rr1, rr2)
                
                fig.add_trace(go.Scatter(
                    x=ellipse_x,
                    y=ellipse_y,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name=f'95% Ellipse (SD1: {sd1:.1f}, SD2: {sd2:.1f})',
                    hovertemplate=f'95% Confidence Ellipse<br>SD1: {sd1:.1f} ms<br>SD2: {sd2:.1f} ms<extra></extra>'
                ))
                
                # Add center point
                center_x = np.mean(rr1)
                center_y = np.mean(rr2)
                fig.add_trace(go.Scatter(
                    x=[center_x],
                    y=[center_y],
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='x'),
                    name='Center',
                    hovertemplate=f'Center<br>({center_x:.1f}, {center_y:.1f})<extra></extra>'
                ))
                
            # Update layout with responsive full-screen configuration
            layout_config = self._get_responsive_layout()
            layout_config.update({
                'title': dict(
                    text="Poincaré Plot",
                    x=0.02,
                    y=0.98,
                    xanchor='left',
                    yanchor='top',
                    font=dict(size=12, color="#666666")
                ),
                'xaxis_title': "RR(n) [ms]",
                'yaxis_title': "RR(n+1) [ms]",
                'hovermode': 'closest',
            })
            
            fig.update_layout(**layout_config)
            
            # Equal aspect ratio
            fig.update_xaxes(scaleanchor="y", scaleratio=1)
            
            if show_grid:
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Poincaré plot: {e}")
            return self._create_error_figure(f"Error creating Poincaré plot: {e}")
            
    def create_psd_plot(self,
                       rr_intervals: np.ndarray,
                       method: str = 'welch',
                       sampling_rate: float = 4.0,
                       title: str = "",
                       show_bands: bool = True,
                       log_scale: bool = True) -> go.Figure:
        """
        Create interactive power spectral density plot.
        
        Args:
            rr_intervals: RR intervals in milliseconds
            method: PSD estimation method ('welch', 'periodogram')
            sampling_rate: Sampling rate for interpolation
            title: Plot title
            show_bands: Whether to highlight frequency bands
            log_scale: Whether to use logarithmic y-axis
            
        Returns:
            Plotly figure object
        """
        try:
            if len(rr_intervals) < 50:
                return self._create_error_figure("Insufficient data for PSD analysis (minimum 50 intervals required)")
                
            # Interpolate RR intervals to regular sampling
            # Create time stamps at RR interval midpoints for proper interpolation
            time_original = np.cumsum(rr_intervals) / 1000.0  # Convert to seconds
            time_original = np.insert(time_original, 0, 0)    # Add initial time = 0
            
            # Use midpoints between consecutive time stamps
            time_midpoints = (time_original[:-1] + time_original[1:]) / 2
            rr_for_interp = rr_intervals[:-1]  # Remove last RR to match midpoint length
            
            # Ensure arrays have same length
            if len(time_midpoints) != len(rr_for_interp):
                min_len = min(len(time_midpoints), len(rr_for_interp))
                time_midpoints = time_midpoints[:min_len]
                rr_for_interp = rr_for_interp[:min_len]
            
            time_regular = np.arange(0, time_midpoints[-1], 1/sampling_rate)
            
            if len(time_regular) < 10:
                return self._create_error_figure("Insufficient temporal coverage for PSD analysis")
                
            f_interp = interp1d(time_midpoints, rr_for_interp, 
                              kind='cubic' if len(rr_for_interp) >= 4 else 'linear',
                              bounds_error=False, fill_value='extrapolate')
            rr_interpolated = f_interp(time_regular)
            
            # Detrend
            rr_detrended = signal.detrend(rr_interpolated)
            
            # Compute PSD
            if method == 'welch':
                freqs, psd = signal.welch(rr_detrended, fs=sampling_rate,
                                        nperseg=min(len(rr_detrended)//4, 256),
                                        window='hann')
            else:
                freqs, psd = signal.periodogram(rr_detrended, fs=sampling_rate, window='hann')
                
            # Create figure
            fig = go.Figure()
            
            # Add PSD line
            fig.add_trace(go.Scatter(
                x=freqs,
                y=psd,
                mode='lines',
                line=dict(color=self.color_palette[0], width=2),
                name='Power Spectral Density',
                hovertemplate='Frequency: %{x:.4f} Hz<br>Power: %{y:.2f} ms²/Hz<extra></extra>'
            ))
            
            # Add frequency bands if requested
            if show_bands:
                bands = {
                    'VLF': (0.0033, 0.04, 'rgba(255, 0, 0, 0.2)'),
                    'LF': (0.04, 0.15, 'rgba(0, 255, 0, 0.2)'),
                    'HF': (0.15, 0.4, 'rgba(0, 0, 255, 0.2)')
                }
                
                for band_name, (f_low, f_high, color) in bands.items():
                    band_mask = (freqs >= f_low) & (freqs <= f_high)
                    if np.any(band_mask):
                        band_power = np.trapz(psd[band_mask], freqs[band_mask])
                        peak_freq = freqs[band_mask][np.argmax(psd[band_mask])]
                        
                        # Add shaded area
                        fig.add_shape(
                            type="rect",
                            x0=f_low, x1=f_high,
                            y0=0, y1=np.max(psd),
                            fillcolor=color,
                            opacity=0.3,
                            line_width=0,
                        )
                        
                        # Add band label
                        fig.add_annotation(
                            x=(f_low + f_high) / 2,
                            y=np.max(psd) * 0.9,
                            text=f"{band_name}<br>Power: {band_power:.0f} ms²<br>Peak: {peak_freq:.3f} Hz",
                            showarrow=False,
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=1,
                            font=dict(size=10)
                        )
                        
            # Update layout with responsive full-screen configuration
            yaxis_type = 'log' if log_scale else 'linear'
            
            layout_config = self._get_responsive_layout()
            layout_config.update({
                'title': dict(
                    text="Power Spectral Density",
                    x=0.02,
                    y=0.98,
                    xanchor='left',
                    yanchor='top',
                    font=dict(size=12, color="#666666")
                ),
                'xaxis_title': "Frequency [Hz]",
                'yaxis_title': "Power [ms²/Hz]",
                'yaxis_type': yaxis_type,
                'hovermode': 'x unified',
            })
            
            fig.update_layout(**layout_config)
            
            # Set frequency range
            fig.update_xaxes(range=[0, 0.5])
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating PSD plot: {e}")
            return self._create_error_figure(f"Error creating PSD plot: {e}")
    
    def create_combined_time_series_analysis(self,
                                           analysis_results: Dict[str, Any],
                                           metrics_to_plot: List[str] = None,
                                           subjects_to_include: List[str] = None) -> go.Figure:
        """
        Create comprehensive time series analysis for all subjects and HRV metrics.
        
        Args:
            analysis_results: Complete analysis results from HRV analysis
            metrics_to_plot: List of specific metrics to plot (if None, plots key metrics)
            subjects_to_include: List of subjects to include (if None, includes all)
            
        Returns:
            Plotly figure with combined time series analysis
        """
        try:
            # Use comprehensive metrics if none specified
            if metrics_to_plot is None:
                metrics_to_plot = self._get_comprehensive_gam_metrics()
            
            # Extract and organize time series data
            time_series_data = self._extract_time_series_data(analysis_results, subjects_to_include)
            
            if not time_series_data:
                return self._create_error_figure("No time series data available for analysis")
            
            # Calculate number of subplots needed
            n_metrics = len([m for m in metrics_to_plot if self._metric_has_data(m, time_series_data)])
            if n_metrics == 0:
                return self._create_error_figure("No valid metrics found in analysis results")
            
            # Create subplots - arrange in 3 columns for better layout
            cols = 3
            rows = (n_metrics + cols - 1) // cols
            
            subplot_titles = []
            valid_metrics = []
            
            for metric in metrics_to_plot:
                if self._metric_has_data(metric, time_series_data):
                    # Create nice titles from metric names
                    domain, metric_name = metric.split('_', 1)
                    title = self._format_metric_title(metric_name, domain)
                    subplot_titles.append(title)
                    valid_metrics.append(metric)
            
            fig = make_subplots(
                rows=rows, cols=cols,
                vertical_spacing=0.08,
                horizontal_spacing=0.08
            )
            
            # Color palette for subjects
            subject_colors = {}
            available_subjects = list(time_series_data.keys())
            for i, subject in enumerate(available_subjects):
                subject_colors[subject] = self.color_palette[i % len(self.color_palette)]
            
            # Plot each metric
            for idx, metric in enumerate(valid_metrics):
                row = (idx // cols) + 1
                col = (idx % cols) + 1
                
                # Add traces for each subject
                for subject in available_subjects:
                    if subject in time_series_data and metric in time_series_data[subject]:
                        subject_data = time_series_data[subject][metric]
                        if len(subject_data['sols']) > 0:
                            
                            # Add main line
                            fig.add_trace(
                                go.Scatter(
                                    x=subject_data['sols'],
                                    y=subject_data['values'],
                                    mode='lines+markers',
                                    line=dict(color=subject_colors[subject], width=2),
                                    marker=dict(size=6),
                                    name=f"{subject}",
                                    showlegend=(idx == 0),  # Only show legend on first subplot
                                    hovertemplate=f'<b>{subject}</b><br>SOL: %{{x}}<br>{subplot_titles[idx]}: %{{y:.2f}}<extra></extra>'
                                ),
                                row=row, col=col
                            )
                            
                            # Add trend line if data has enough points
                            if len(subject_data['sols']) >= 3:
                                trend_line = self._calculate_trend_line(subject_data['sols'], subject_data['values'])
                                if trend_line is not None:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=subject_data['sols'],
                                            y=trend_line,
                                            mode='lines',
                                            line=dict(color=subject_colors[subject], width=1, dash='dash'),
                                            showlegend=False,
                                            hoverinfo='skip',
                                            opacity=0.7
                                        ),
                                        row=row, col=col
                                    )
            
            # Update layout with responsive full-screen configuration
            # Calculate appropriate height based on number of rows
            calculated_height = 300 * rows if not self.responsive else None
            
            layout_config = self._get_responsive_layout(height=calculated_height)
            layout_config.update({
                'title': dict(
                    text="HRV Time Series",
                    x=0.02,
                    y=0.98,
                    xanchor='left',
                    yanchor='top',
                    font=dict(size=12, color="#666666")
                ),
                'showlegend': True,
                'legend': dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                'font': dict(size=10)
            })
            
            # Override responsive height with calculated value for multi-row plots
            if self.responsive:
                layout_config['height'] = None  # Let browser handle height
                layout_config['autosize'] = True
            
            fig.update_layout(**layout_config)
            
            # Update axes labels
            for i in range(1, rows + 1):
                for j in range(1, cols + 1):
                    fig.update_xaxes(title_text="SOL Session", row=i, col=j)
            
            # Add annotations with statistical insights
            self._add_statistical_annotations(fig, time_series_data, valid_metrics)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating combined time series analysis: {e}")
            return self._create_error_figure(f"Error creating time series analysis: {e}")
    
    def create_gam_crew_analysis(self,
                               analysis_results: Dict[str, Any],
                               metrics_to_plot: List[str] = None,
                               subjects_to_include: List[str] = None,
                               show_individual_points: bool = True,
                               show_crew_median: bool = True,
                               confidence_level: float = 0.95) -> go.Figure:
        """
        Create GAM (Generalized Additive Model) analysis for crew-wide HRV trends.
        
        This creates professional time series plots with:
        - Individual subject data points (colored by subject)
        - GAM trend line for the entire crew
        - Confidence intervals around the trend
        - Crew median calculation over time
        - Professional styling matching aerospace research standards
        
        Args:
            analysis_results: Complete analysis results from HRV analysis
            metrics_to_plot: List of specific metrics to plot
            subjects_to_include: List of subjects to include
            show_individual_points: Whether to show individual subject data points
            show_crew_median: Whether to show crew median trend
            confidence_level: Confidence level for GAM intervals (default 95%)
            
        Returns:
            Plotly figure with GAM crew analysis
        """
        try:
            # Comprehensive default metrics covering all HRV domains
            if metrics_to_plot is None:
                metrics_to_plot = self._get_comprehensive_gam_metrics()
            
            # Extract time series data
            time_series_data = self._extract_time_series_data(analysis_results, subjects_to_include)
            
            if not time_series_data:
                return self._create_error_figure("No time series data available for GAM analysis")
            
            # Filter to metrics with data
            valid_metrics = [m for m in metrics_to_plot if self._metric_has_data(m, time_series_data)]
            
            if not valid_metrics:
                return self._create_error_figure("No valid metrics found for GAM analysis")
            
            # Calculate layout: prefer 2-3 columns for better readability
            n_metrics = len(valid_metrics)
            cols = min(3, n_metrics)
            rows = (n_metrics + cols - 1) // cols
            
            # Create subplot titles
            subplot_titles = [self._format_metric_title_professional(metric) for metric in valid_metrics]
            
            fig = make_subplots(
                rows=rows, cols=cols,
                vertical_spacing=0.12,
                horizontal_spacing=0.10
            )
            
            # Define professional color palette for subjects
            subject_colors = self._get_subject_color_palette(list(time_series_data.keys()))
            
            # Process each metric
            for idx, metric in enumerate(valid_metrics):
                row = (idx // cols) + 1
                col = (idx % cols) + 1
                
                # Collect all data points for this metric across all subjects
                all_sols = []
                all_values = []
                subject_labels = []
                
                # Individual subject data
                for subject in time_series_data:
                    if metric in time_series_data[subject]:
                        subject_data = time_series_data[subject][metric]
                        if len(subject_data['sols']) > 0:
                            all_sols.extend(subject_data['sols'])
                            all_values.extend(subject_data['values'])
                            subject_labels.extend([subject] * len(subject_data['sols']))
                            
                            # Add individual subject points if requested
                            if show_individual_points:
                                fig.add_trace(
                                    go.Scatter(
                                        x=subject_data['sols'],
                                        y=subject_data['values'],
                                        mode='markers',
                                        marker=dict(
                                            color=subject_colors[subject],
                                            size=8,
                                            opacity=0.7,
                                            line=dict(width=1, color='rgba(0,0,0,0.3)')
                                        ),
                                        name=subject,
                                        showlegend=(idx == 0),  # Only show legend on first subplot
                                        hovertemplate=f'<b>{subject}</b><br>SOL: %{{x}}<br>{subplot_titles[idx]}: %{{y:.2f}}<extra></extra>'
                                    ),
                                    row=row, col=col
                                )
                
                if len(all_sols) >= 3:  # Need minimum data points for GAM
                    # Convert to arrays for analysis
                    sols_array = np.array(all_sols)
                    values_array = np.array(all_values)
                    
                    # Calculate crew median at each SOL if requested
                    if show_crew_median:
                        median_data = self._calculate_crew_median_by_sol(
                            time_series_data, metric
                        )
                        
                        if median_data['sols'] and len(median_data['sols']) > 0:
                            fig.add_trace(
                                go.Scatter(
                                    x=median_data['sols'],
                                    y=median_data['medians'],
                                    mode='markers',
                                    marker=dict(
                                        color='black',
                                        size=10,
                                        symbol='diamond',
                                        line=dict(width=2, color='white')
                                    ),
                                    name='Crew Median',
                                    showlegend=(idx == 0),
                                    hovertemplate=f'<b>Crew Median</b><br>SOL: %{{x}}<br>{subplot_titles[idx]}: %{{y:.2f}}<extra></extra>'
                                ),
                                row=row, col=col
                            )
                    
                    # Fit GAM model and add trend line with confidence intervals
                    gam_data = self._fit_gam_model(sols_array, values_array, confidence_level)
                    
                    if gam_data is not None:
                        # Determine the X-axis range for reference bands
                        x_min = min(sols_array.min(), gam_data['x_smooth'].min())
                        x_max = max(sols_array.max(), gam_data['x_smooth'].max())
                        # Add a small buffer to ensure full coverage
                        x_buffer = (x_max - x_min) * 0.05  # 5% buffer
                        x_range = (x_min - x_buffer, x_max + x_buffer)
                        
                        # Add normal reference ranges as background bands FIRST
                        # This ensures they appear behind all other data
                        self._add_reference_bands_to_gam_plot(fig, metric, row, col, idx == 0, x_range)
                        
                        # Add confidence interval (appears above reference bands, below data)
                        fig.add_trace(
                            go.Scatter(
                                x=np.concatenate([gam_data['x_smooth'], gam_data['x_smooth'][::-1]]),
                                y=np.concatenate([gam_data['upper_ci'], gam_data['lower_ci'][::-1]]),
                                fill='toself',
                                fillcolor='rgba(65, 105, 225, 0.2)',  # Light blue
                                line=dict(color='rgba(255,255,255,0)'),
                                showlegend=(idx == 0),
                                name=f'{int(confidence_level*100)}% Confidence Interval',
                                hoverinfo='skip'
                            ),
                            row=row, col=col
                        )
                        
                        # Add GAM trend line (appears above everything)
                        fig.add_trace(
                            go.Scatter(
                                x=gam_data['x_smooth'],
                                y=gam_data['y_smooth'],
                                mode='lines',
                                line=dict(
                                    color='rgb(65, 105, 225)',  # Professional blue
                                    width=3
                                ),
                                name='GAM Trend',
                                showlegend=(idx == 0),
                                hovertemplate=f'<b>GAM Trend</b><br>SOL: %{{x}}<br>{subplot_titles[idx]}: %{{y:.2f}}<extra></extra>'
                            ),
                            row=row, col=col
                        )
                    else:
                        # If GAM fitting failed, still add reference bands with data range
                        x_range = (sols_array.min() - 0.5, sols_array.max() + 0.5)
                        self._add_reference_bands_to_gam_plot(fig, metric, row, col, idx == 0, x_range)
            
            # Professional layout configuration
            layout_config = self._get_responsive_layout(height=300 * rows)
            layout_config.update({
                'title': dict(
                    text="Crew HRV Trends",
                    x=0.02,
                    y=0.98,
                    xanchor='left',
                    yanchor='top',
                    font=dict(size=12, color="#666666")
                ),
                'showlegend': True,
                'legend': dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=11)
                ),
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'font': dict(
                    family="Arial, sans-serif",
                    size=11,
                    color="#2c3e50"
                )
            })
            
            fig.update_layout(**layout_config)
            
            # Update axes with professional styling
            for i in range(1, rows + 1):
                for j in range(1, cols + 1):
                    fig.update_xaxes(
                        title_text="Sol (Mission Day)",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128,128,128,0.2)',
                        row=i, col=j
                    )
                    fig.update_yaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128,128,128,0.2)',
                        row=i, col=j
                    )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating GAM crew analysis: {e}")
            return self._create_error_figure(f"Error creating GAM analysis: {e}")
    
    def _get_subject_color_palette(self, subjects: List[str]) -> Dict[str, str]:
        """Get a professional color palette for subjects."""
        # Professional color palette suitable for scientific publications
        professional_colors = [
            '#FF6B8A',  # Pink/Red - T01_Mara
            '#FF8C42',  # Orange - T02_Laura  
            '#8FBC8F',  # Sage Green - T03_Nancy
            '#20B2AA',  # Light Sea Green - T04_Michelle
            '#4682B4',  # Steel Blue - T05_Felicitas
            '#9370DB',  # Medium Purple - T06_Mara_Selena
            '#32CD32',  # Lime Green - T07_Geraldinn
            '#FF69B4'   # Hot Pink - T08_Karina
        ]
        
        subject_colors = {}
        for i, subject in enumerate(subjects):
            subject_colors[subject] = professional_colors[i % len(professional_colors)]
        
        return subject_colors
    
    def _format_metric_title_professional(self, metric: str) -> str:
        """Format metric names for professional presentation."""
        # Extract domain and metric name
        if '_' in metric:
            domain, metric_name = metric.split('_', 1)
        else:
            domain, metric_name = '', metric
        
        title_mapping = {
            'rmssd': 'RMSSD Value',
            'sdnn': 'SDNN Value', 
            'pnn50': 'pNN50 Value',
            'mean_hr': 'Mean HR (bpm)',
            'lf_power': 'LF Power',
            'hf_power': 'HF Power',
            'lf_hf_ratio': 'LF/HF Ratio',
            'lf_nu': 'LF (n.u.)',
            'hf_nu': 'HF (n.u.)'
        }
        
        return title_mapping.get(metric_name, metric_name.replace('_', ' ').title())
    
    def _calculate_crew_median_by_sol(self, time_series_data: Dict, metric: str) -> Dict[str, List]:
        """Calculate crew median values at each SOL."""
        sol_values = {}
        
        # Collect all values by SOL
        for subject in time_series_data:
            if metric in time_series_data[subject]:
                subject_data = time_series_data[subject][metric]
                for sol, value in zip(subject_data['sols'], subject_data['values']):
                    if sol not in sol_values:
                        sol_values[sol] = []
                    sol_values[sol].append(value)
        
        # Calculate median for each SOL
        sols = sorted(sol_values.keys())
        medians = [np.median(sol_values[sol]) for sol in sols]
        
        return {'sols': sols, 'medians': medians}
    
    def _fit_gam_model(self, x_data: np.ndarray, y_data: np.ndarray, confidence_level: float = 0.95) -> Optional[Dict[str, np.ndarray]]:
        """Fit GAM model and return smoothed trend with confidence intervals."""
        if not GAM_AVAILABLE:
            logger.warning("GAM not available, using polynomial trend instead")
            return self._fit_polynomial_trend(x_data, y_data, confidence_level)
        
        try:
            # Validate input data
            if len(x_data) != len(y_data):
                raise ValueError(f"X and Y data must have same length: {len(x_data)} != {len(y_data)}")
            
            if len(x_data) < 5:
                logger.debug(f"Insufficient data points for GAM ({len(x_data)}), using polynomial fallback")
                return self._fit_polynomial_trend(x_data, y_data, confidence_level)
            
            # Remove any NaN values
            valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
            if not np.any(valid_mask):
                raise ValueError("No valid data points after removing NaNs")
            
            x_clean = x_data[valid_mask]
            y_clean = y_data[valid_mask]
            
            if len(x_clean) < 5:
                logger.debug(f"Insufficient valid data points for GAM ({len(x_clean)}), using polynomial fallback")
                return self._fit_polynomial_trend(x_clean, y_clean, confidence_level)
            
            # Prepare data and sort by x values
            df = pd.DataFrame({'x': x_clean, 'y': y_clean}).sort_values('x').reset_index(drop=True)
            
            # Determine appropriate degrees of freedom based on data size
            # Rule of thumb: df should be much smaller than number of data points
            n_points = len(df)
            if n_points >= 20:
                df_spline = min(6, max(3, n_points // 4))  # Between 3 and 6, based on data size
            elif n_points >= 10:
                df_spline = 3
            else:
                df_spline = 2
            
            logger.debug(f"GAM fitting with {n_points} points, using df={df_spline}")
            
            # Create smoothing spline for GAM with appropriate degrees of freedom
            x_spline = BSplines(df[['x']], df=[df_spline], degree=[3])
            
            # Fit GAM model
            gam_model = GLMGam(df['y'], smoother=x_spline, family=Gaussian())
            gam_results = gam_model.fit()
            
            # Create smooth x values for prediction
            x_min, x_max = df['x'].min(), df['x'].max()
            x_range = np.linspace(x_min, x_max, 50)
            x_smooth_df = pd.DataFrame({'x': x_range})
            
            # Create matching spline basis for prediction with same parameters
            x_smooth_spline = BSplines(x_smooth_df[['x']], df=[df_spline], degree=[3])
            
            # Predict with confidence intervals
            predictions = gam_results.get_prediction(x_smooth_spline)
            y_smooth = predictions.predicted_mean
            conf_int = predictions.conf_int(alpha=1-confidence_level)
            
            # Validate prediction outputs
            if len(y_smooth) != len(x_range):
                raise ValueError(f"Prediction length mismatch: {len(y_smooth)} != {len(x_range)}")
            
            logger.debug(f"GAM fitting successful with df={df_spline}, predictions shape: {y_smooth.shape}")
            
            return {
                'x_smooth': x_range,
                'y_smooth': y_smooth,
                'lower_ci': conf_int[:, 0],
                'upper_ci': conf_int[:, 1]
            }
            
        except Exception as e:
            logger.warning(f"GAM fitting failed: {e}, using polynomial fallback")
            logger.debug(f"GAM error details - Data shapes: x_data={x_data.shape}, y_data={y_data.shape}")
            return self._fit_polynomial_trend(x_data, y_data, confidence_level)
    
    def _fit_polynomial_trend(self, x_data: np.ndarray, y_data: np.ndarray, confidence_level: float = 0.95) -> Optional[Dict[str, np.ndarray]]:
        """Fallback polynomial trend fitting with confidence intervals."""
        try:
            # Validate input data
            if len(x_data) != len(y_data):
                raise ValueError(f"X and Y data must have same length: {len(x_data)} != {len(y_data)}")
            
            if len(x_data) < 2:
                logger.warning(f"Insufficient data points for trend fitting ({len(x_data)})")
                return None
            
            # Remove any NaN values
            valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
            if not np.any(valid_mask):
                logger.warning("No valid data points after removing NaNs")
                return None
            
            x_clean = x_data[valid_mask]
            y_clean = y_data[valid_mask]
            
            if len(x_clean) < 2:
                logger.warning(f"Insufficient valid data points for trend fitting ({len(x_clean)})")
                return None
            
            # Sort data
            sorted_indices = np.argsort(x_clean)
            x_sorted = x_clean[sorted_indices]
            y_sorted = y_clean[sorted_indices]
            
            # Choose polynomial degree based on data amount
            if len(x_sorted) >= 4:
                degree = 2  # Quadratic for smooth curves
            else:
                degree = 1  # Linear for limited data
            
            logger.debug(f"Polynomial fitting with {len(x_sorted)} points, degree={degree}")
            
            # Fit polynomial
            poly_coeffs = np.polyfit(x_sorted, y_sorted, degree)
            
            # Create smooth x values
            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 50)
            y_smooth = np.polyval(poly_coeffs, x_smooth)
            
            # Calculate confidence intervals using bootstrap if we have enough data
            if len(x_sorted) >= 5:
                n_bootstrap = min(100, len(x_sorted) * 10)  # Adaptive bootstrap samples
                predictions = []
                
                for _ in range(n_bootstrap):
                    # Bootstrap sample
                    indices = np.random.choice(len(x_sorted), len(x_sorted), replace=True)
                    x_boot = x_sorted[indices]
                    y_boot = y_sorted[indices]
                    
                    try:
                        # Fit polynomial
                        boot_coeffs = np.polyfit(x_boot, y_boot, degree)
                        boot_pred = np.polyval(boot_coeffs, x_smooth)
                        predictions.append(boot_pred)
                    except np.linalg.LinAlgError:
                        # Skip this bootstrap sample if singular matrix
                        continue
                
                if predictions:
                    predictions = np.array(predictions)
                    alpha = 1 - confidence_level
                    lower_ci = np.percentile(predictions, 100 * alpha/2, axis=0)
                    upper_ci = np.percentile(predictions, 100 * (1-alpha/2), axis=0)
                else:
                    # Fallback to simple error estimation
                    residuals = y_sorted - np.polyval(poly_coeffs, x_sorted)
                    std_error = np.std(residuals)
                    lower_ci = y_smooth - 1.96 * std_error
                    upper_ci = y_smooth + 1.96 * std_error
            else:
                # For very small datasets, use simple error estimation
                residuals = y_sorted - np.polyval(poly_coeffs, x_sorted)
                std_error = np.std(residuals) if len(residuals) > 1 else 0.1 * np.std(y_sorted)
                lower_ci = y_smooth - 1.96 * std_error
                upper_ci = y_smooth + 1.96 * std_error
            
            logger.debug(f"Polynomial fitting successful, predictions shape: {y_smooth.shape}")
            
            return {
                'x_smooth': x_smooth,
                'y_smooth': y_smooth,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci
            }
            
        except Exception as e:
            logger.error(f"Polynomial trend fitting failed: {e}")
            logger.debug(f"Polynomial error details - Data shapes: x_data={x_data.shape}, y_data={y_data.shape}")
            return None

    def _extract_time_series_data(self, analysis_results: Dict[str, Any], subjects_filter: List[str] = None) -> Dict[str, Dict[str, Dict[str, list]]]:
        """Extract and organize time series data from analysis results."""
        time_series_data = {}
        
        for key, result in analysis_results.items():
            # Skip non-subject results
            if key in ['clustering', 'forecasting'] or 'hrv_results' not in result:
                continue
            
            # Parse subject and SOL from key (e.g., 'T01_Mara_Sol10' -> subject='T01_Mara', sol=10)
            if '_Sol' in key:
                try:
                    subject, sol_str = key.rsplit('_Sol', 1)
                    sol = int(sol_str)
                except:
                    continue  # Skip if parsing fails
            else:
                continue  # Skip if no SOL info
            
            # Apply subject filter if provided
            if subjects_filter and subject not in subjects_filter:
                continue
            
            # Initialize subject data if not exists
            if subject not in time_series_data:
                time_series_data[subject] = {}
            
            hrv_results = result['hrv_results']
            
            # Extract all metrics
            for domain, metrics in hrv_results.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            full_metric_name = f"{domain}_{metric_name}"
                            
                            if full_metric_name not in time_series_data[subject]:
                                time_series_data[subject][full_metric_name] = {
                                    'sols': [],
                                    'values': []
                                }
                            
                            time_series_data[subject][full_metric_name]['sols'].append(sol)
                            time_series_data[subject][full_metric_name]['values'].append(float(value))
        
        # Sort data by SOL for each subject and metric
        for subject in time_series_data:
            for metric in time_series_data[subject]:
                if time_series_data[subject][metric]['sols']:
                    # Sort by SOL
                    sorted_indices = np.argsort(time_series_data[subject][metric]['sols'])
                    time_series_data[subject][metric]['sols'] = [
                        time_series_data[subject][metric]['sols'][i] for i in sorted_indices
                    ]
                    time_series_data[subject][metric]['values'] = [
                        time_series_data[subject][metric]['values'][i] for i in sorted_indices
                    ]
        
        return time_series_data
    
    def _metric_has_data(self, metric: str, time_series_data: Dict) -> bool:
        """Check if any subject has data for the given metric."""
        for subject in time_series_data:
            if metric in time_series_data[subject] and len(time_series_data[subject][metric]['sols']) > 0:
                return True
        return False
    
    def _format_metric_title(self, metric_name: str, domain: str) -> str:
        """Format metric names for display titles."""
        # Create nice titles
        title_mapping = {
            # Time domain
            'sdnn': 'SDNN (ms)',
            'rmssd': 'RMSSD (ms)', 
            'pnn50': 'pNN50 (%)',
            'mean_hr': 'Mean HR (BPM)',
            'cvnn': 'CVNN (%)',
            
            # Frequency domain
            'lf_power': 'LF Power (ms²)',
            'hf_power': 'HF Power (ms²)',
            'lf_hf_ratio': 'LF/HF Ratio',
            'lf_nu': 'LF Normalized Units (%)',
            'hf_nu': 'HF Normalized Units (%)',
            'total_power': 'Total Power (ms²)',
            
            # Nonlinear
            'sd1': 'Poincaré SD1 (ms)',
            'sd2': 'Poincaré SD2 (ms)',
            'dfa_alpha1': 'DFA α1',
            'sample_entropy': 'Sample Entropy'
        }
        
        return title_mapping.get(metric_name, metric_name.replace('_', ' ').title())
    
    def _calculate_trend_line(self, x_values: List, y_values: List) -> Optional[np.ndarray]:
        """Calculate linear trend line for data points."""
        try:
            if len(x_values) < 2:
                return None
            
            x = np.array(x_values)
            y = np.array(y_values)
            
            # Simple linear regression
            coeffs = np.polyfit(x, y, 1)
            trend = np.polyval(coeffs, x)
            return trend
        except:
            return None
    
    def _add_statistical_annotations(self, fig: go.Figure, time_series_data: Dict, metrics: List[str]):
        """Add statistical insights as annotations."""
        try:
            # Calculate some basic statistics for annotations
            total_subjects = len(time_series_data)
            total_sessions = sum(len(time_series_data[subj][metrics[0]]['sols']) 
                               for subj in time_series_data if metrics[0] in time_series_data[subj])
            
            # Add summary annotation
            fig.add_annotation(
                text=f"📊 Analysis: {total_subjects} subjects, {total_sessions} total sessions<br>"
                     f"📈 Trends and patterns across space simulation SOL sessions<br>"
                     f"💡 Dashed lines show individual subject trends",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1
            )
            
        except Exception as e:
            logger.error(f"Error adding statistical annotations: {e}")
            
    def create_correlation_heatmap(self,
                                 hrv_metrics: pd.DataFrame,
                                 title: str = "",
                                 show_significance: bool = True,
                                 cluster_metrics: bool = True) -> go.Figure:
        """
        Create interactive correlation heatmap with significance testing.
        
        Args:
            hrv_metrics: DataFrame with HRV metrics
            title: Plot title  
            show_significance: Whether to show statistical significance
            cluster_metrics: Whether to cluster similar metrics
            
        Returns:
            Plotly figure object
        """
        try:
            if hrv_metrics.empty:
                return self._create_error_figure("No data provided for correlation analysis")
                
            # Select only numeric columns
            numeric_cols = hrv_metrics.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return self._create_error_figure("Insufficient numeric variables for correlation analysis")
                
            data = hrv_metrics[numeric_cols]
            
            # Remove columns with no variation
            data = data.loc[:, data.std() > 0]
            
            if data.empty:
                return self._create_error_figure("No variables with sufficient variation")
                
            # Compute correlation matrix
            corr_matrix = data.corr()
            
            # Compute p-values if significance testing is requested
            p_values = None
            if show_significance and len(data) > 3:
                p_values = np.zeros_like(corr_matrix.values)
                for i in range(len(corr_matrix.columns)):
                    for j in range(len(corr_matrix.columns)):
                        if i != j:
                            try:
                                _, p_val = stats.pearsonr(data.iloc[:, i].dropna(), 
                                                        data.iloc[:, j].dropna())
                                p_values[i, j] = p_val
                            except:
                                p_values[i, j] = 1.0
                                
            # Cluster metrics if requested
            if cluster_metrics:
                try:
                    from scipy.cluster.hierarchy import dendrogram, linkage
                    from scipy.spatial.distance import squareform
                    
                    # Convert correlation to distance
                    distance_matrix = 1 - np.abs(corr_matrix.values)
                    np.fill_diagonal(distance_matrix, 0)
                    
                    # Hierarchical clustering
                    condensed_distances = squareform(distance_matrix)
                    linkage_matrix = linkage(condensed_distances, method='ward')
                    
                    # Get cluster order
                    from scipy.cluster.hierarchy import leaves_list
                    cluster_order = leaves_list(linkage_matrix)
                    
                    # Reorder correlation matrix
                    ordered_cols = corr_matrix.columns[cluster_order]
                    corr_matrix = corr_matrix.loc[ordered_cols, ordered_cols]
                    
                    if p_values is not None:
                        p_values = p_values[cluster_order][:, cluster_order]
                        
                except ImportError:
                    logger.warning("Scipy clustering not available, skipping metric clustering")
                except Exception as e:
                    logger.warning(f"Clustering failed: {e}")
                    
            # Create hover text
            hover_text = []
            for i in range(len(corr_matrix.index)):
                hover_row = []
                for j in range(len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    text = f"{corr_matrix.index[i]} vs {corr_matrix.columns[j]}<br>"
                    text += f"Correlation: {corr_val:.3f}<br>"
                    
                    if p_values is not None:
                        p_val = p_values[i, j]
                        sig_level = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                        text += f"p-value: {p_val:.3f} {sig_level}"
                    
                    hover_row.append(text)
                hover_text.append(hover_row)
                
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                zmin=-1,
                zmax=1,
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                colorbar=dict(
                    title="Correlation Coefficient",
                    titleside="right"
                )
            ))
            
            # Add significance markers if requested
            if show_significance and p_values is not None:
                for i in range(len(corr_matrix.index)):
                    for j in range(len(corr_matrix.columns)):
                        p_val = p_values[i, j]
                        if p_val < 0.05:
                            marker_symbol = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                            fig.add_annotation(
                                x=j, y=i,
                                text=marker_symbol,
                                showarrow=False,
                                font=dict(color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white',
                                         size=12)
                            )
                            
            # Update layout
            fig.update_layout(
                template=self.template,
                width=max(self.default_width, len(corr_matrix.columns) * 40 + 200),
                height=max(self.default_height, len(corr_matrix.index) * 40 + 200),
                font=dict(size=self.font_size)
            )
            
            # Equal aspect ratio
            fig.update_xaxes(side="bottom")
            fig.update_yaxes(autorange="reversed")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return self._create_error_figure(f"Error creating correlation heatmap: {e}")
            
    def create_time_series_plot(self,
                              rr_intervals: np.ndarray,
                              timestamps: np.ndarray = None,
                              title: str = "",
                              show_trend: bool = True,
                              show_variability_bands: bool = True) -> go.Figure:
        """
        Create interactive RR interval time series plot with trend analysis.
        
        Args:
            rr_intervals: RR intervals in milliseconds
            timestamps: Optional timestamps for x-axis
            title: Plot title
            show_trend: Whether to show trend line
            show_variability_bands: Whether to show variability bands
            
        Returns:
            Plotly figure object
        """
        try:
            if len(rr_intervals) == 0:
                return self._create_error_figure("No RR interval data provided")
                
            # Create time vector if not provided
            if timestamps is None:
                timestamps = np.arange(len(rr_intervals))
                x_title = "Beat Index"
            else:
                x_title = "Time"
                
            # Create figure
            fig = go.Figure()
            
            # Add RR intervals
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=rr_intervals,
                mode='lines+markers',
                line=dict(color=self.color_palette[0], width=1),
                marker=dict(size=3, opacity=0.7),
                name='RR Intervals',
                hovertemplate='%{x}<br>RR: %{y:.1f} ms<extra></extra>'
            ))
            
            # Add trend line if requested
            if show_trend and len(rr_intervals) > 10:
                try:
                    # Polynomial trend (degree 1 for linear)
                    trend_coeffs = np.polyfit(timestamps, rr_intervals, 1)
                    trend_line = np.polyval(trend_coeffs, timestamps)
                    
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=trend_line,
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name=f'Trend (slope: {trend_coeffs[0]:.2f} ms/unit)',
                        hovertemplate='Linear Trend<br>%{x}<br>%{y:.1f} ms<extra></extra>'
                    ))
                except Exception as e:
                    logger.warning(f"Could not compute trend line: {e}")
                    
            # Add variability bands if requested
            if show_variability_bands:
                # Calculate rolling statistics
                window_size = max(10, len(rr_intervals) // 20)
                rr_series = pd.Series(rr_intervals)
                rolling_mean = rr_series.rolling(window=window_size, center=True).mean()
                rolling_std = rr_series.rolling(window=window_size, center=True).std()
                
                upper_band = rolling_mean + rolling_std
                lower_band = rolling_mean - rolling_std
                
                # Add bands
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=upper_band,
                    mode='lines',
                    line=dict(color='rgba(128,128,128,0)', width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=lower_band,
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.2)',
                    line=dict(color='rgba(128,128,128,0)', width=0),
                    name='±1 SD Band',
                    hovertemplate='Variability Band<br>%{x}<br>%{y:.1f} ms<extra></extra>'
                ))
                
            # Update layout with responsive full-screen configuration
            layout_config = self._get_responsive_layout()
            layout_config.update({
                'title': dict(
                    text="RR Interval Time Series",
                    x=0.02,
                    y=0.98,
                    xanchor='left',
                    yanchor='top',
                    font=dict(size=12, color="#666666")
                ),
                'xaxis_title': x_title,
                'yaxis_title': "RR Interval [ms]",
                'hovermode': 'x unified',
            })
            
            fig.update_layout(**layout_config)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating time series plot: {e}")
            return self._create_error_figure(f"Error creating time series plot: {e}")
            
    def create_hrv_dashboard(self,
                           rr_intervals: np.ndarray,
                           hrv_results: Dict[str, Any],
                           subject_id: str = "Subject",
                           session_id: str = "Session") -> go.Figure:
        """
        Create comprehensive HRV analysis dashboard.
        
        Args:
            rr_intervals: RR intervals in milliseconds
            hrv_results: HRV analysis results dictionary
            subject_id: Subject identifier
            session_id: Session identifier
            
        Returns:
            Plotly figure with subplots dashboard
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"type": "table"}],
                    [{"type": "domain"}, {"secondary_y": False}]
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # 1. RR Interval Time Series (top left)
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(rr_intervals)),
                    y=rr_intervals,
                    mode='lines',
                    line=dict(color=self.color_palette[0], width=1),
                    name='RR Intervals'
                ),
                row=1, col=1
            )
            
            # 2. Poincaré Plot (top right)  
            if len(rr_intervals) > 1:
                rr1 = rr_intervals[:-1]
                rr2 = rr_intervals[1:]
                
                fig.add_trace(
                    go.Scatter(
                        x=rr1,
                        y=rr2,
                        mode='markers',
                        marker=dict(size=4, color=self.color_palette[1], opacity=0.6),
                        name='Poincaré Points'
                    ),
                    row=1, col=2
                )
                
                # Add identity line
                min_rr = min(np.min(rr1), np.min(rr2))
                max_rr = max(np.max(rr1), np.max(rr2))
                fig.add_trace(
                    go.Scatter(
                        x=[min_rr, max_rr],
                        y=[min_rr, max_rr],
                        mode='lines',
                        line=dict(color='gray', width=1, dash='dash'),
                        name='Identity Line'
                    ),
                    row=1, col=2
                )
                
            # 3. Power Spectral Density (middle left)
            if len(rr_intervals) > 50:
                try:
                    # Quick PSD computation
                    time_original = np.cumsum(rr_intervals) / 1000.0
                    time_original = np.insert(time_original[:-1], 0, 0)
                    time_regular = np.arange(0, time_original[-1], 0.25)
                    
                    f_interp = interp1d(time_original, rr_intervals[:-1], 
                                      kind='linear', bounds_error=False, fill_value='extrapolate')
                    rr_interpolated = f_interp(time_regular)
                    rr_detrended = signal.detrend(rr_interpolated)
                    
                    freqs, psd = signal.welch(rr_detrended, fs=4.0, nperseg=min(len(rr_detrended)//4, 128))
                    
                    fig.add_trace(
                        go.Scatter(
                            x=freqs,
                            y=psd,
                            mode='lines',
                            line=dict(color=self.color_palette[2], width=2),
                            name='PSD'
                        ),
                        row=2, col=1
                    )
                except Exception as e:
                    logger.warning(f"Could not add PSD to dashboard: {e}")
                    
            # 4. HRV Metrics Table (middle right)
            if 'time_domain' in hrv_results and 'frequency_domain' in hrv_results:
                time_metrics = hrv_results['time_domain']
                freq_metrics = hrv_results['frequency_domain']
                
                # Enhanced metrics table with reference ranges
                metrics_data = self._create_metrics_table_with_ranges(
                    time_metrics, freq_metrics
                )
                
                # Enhanced table with reference ranges
                header_values = ['Metric', 'Value', 'Normal Range'] if hrv_reference_ranges else ['Metric', 'Value']
                
                fig.add_trace(
                    go.Table(
                        header=dict(values=header_values,
                                  fill_color=self.color_palette[0],
                                  font=dict(color='white', size=11)),
                        cells=dict(values=list(zip(*metrics_data)),
                                 fill_color='white',
                                 font=dict(size=10))
                    ),
                    row=2, col=2
                )
                
            # 5. Frequency Band Pie Chart (bottom left)
            if 'frequency_domain' in hrv_results:
                freq_metrics = hrv_results['frequency_domain']
                
                labels = ['VLF', 'LF', 'HF']
                values = [
                    freq_metrics.get('vlf_power', 0),
                    freq_metrics.get('lf_power', 0),
                    freq_metrics.get('hf_power', 0)
                ]
                
                fig.add_trace(
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.3,
                        marker_colors=self.color_palette[:3]
                    ),
                    row=3, col=1
                )
                
            # 6. Time vs. Frequency Domain Scatter (bottom right)
            if 'time_domain' in hrv_results and 'frequency_domain' in hrv_results:
                time_metrics = hrv_results['time_domain']
                freq_metrics = hrv_results['frequency_domain']
                
                fig.add_trace(
                    go.Scatter(
                        x=[time_metrics.get('rmssd', 0)],
                        y=[freq_metrics.get('hf_power', 0)],
                        mode='markers',
                        marker=dict(size=15, color=self.color_palette[0]),
                        name='Current Analysis',
                        text=[f"{subject_id} - {session_id}"],
                        textposition="middle right"
                    ),
                    row=3, col=2
                )
                
            # Update layout with responsive full-screen configuration
            layout_config = self._get_responsive_layout()
            layout_config.update({
                'title': dict(
                    text=f"HRV Analysis Dashboard - {subject_id} ({session_id})",
                    x=0.5,
                    font=dict(size=16)
                ),
                'showlegend': False,
                'font': dict(size=10)
            })
            
            fig.update_layout(**layout_config)
            
            # Update axis labels
            fig.update_xaxes(title_text="Beat Index", row=1, col=1)
            fig.update_yaxes(title_text="RR Interval [ms]", row=1, col=1)
            
            fig.update_xaxes(title_text="RR(n) [ms]", row=1, col=2)
            fig.update_yaxes(title_text="RR(n+1) [ms]", row=1, col=2)
            
            fig.update_xaxes(title_text="Frequency [Hz]", row=2, col=1)
            fig.update_yaxes(title_text="Power [ms²/Hz]", row=2, col=1, type="log")
            
            fig.update_xaxes(title_text="RMSSD [ms]", row=3, col=2)
            fig.update_yaxes(title_text="HF Power [ms²]", row=3, col=2)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating HRV dashboard: {e}")
            return self._create_error_figure(f"Error creating HRV dashboard: {e}")
            
    def _calculate_poincare_ellipse(self, rr1: np.ndarray, rr2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Calculate 95% confidence ellipse for Poincaré plot."""
        # Calculate SD1 and SD2
        diff_rr = rr2 - rr1
        sum_rr = rr2 + rr1
        
        sd1 = np.std(diff_rr) / np.sqrt(2)
        sd2 = np.std(sum_rr) / np.sqrt(2)
        
        # Calculate ellipse parameters
        center_x = np.mean(rr1)
        center_y = np.mean(rr2)
        
        # 95% confidence ellipse (chi-square 2 DOF at 0.05 = 5.991)
        chi2_95 = 5.991
        a = sd1 * np.sqrt(chi2_95)  # Semi-major axis
        b = sd2 * np.sqrt(chi2_95)  # Semi-minor axis
        
        # Rotation angle (45 degrees for Poincaré plot)
        angle = np.pi / 4
        
        # Generate ellipse points
        t = np.linspace(0, 2*np.pi, 100)
        ellipse_x_rot = a * np.cos(t)
        ellipse_y_rot = b * np.sin(t)
        
        # Apply rotation
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        ellipse_x = center_x + ellipse_x_rot * cos_angle - ellipse_y_rot * sin_angle
        ellipse_y = center_y + ellipse_x_rot * sin_angle + ellipse_y_rot * cos_angle
        
        return ellipse_x, ellipse_y, sd1, sd2
        
    def _create_error_figure(self, error_message: str) -> go.Figure:
        """Create error figure for failed plots."""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text=f"Error: {error_message}",
            showarrow=False,
            font=dict(size=16, color="red"),
            align="center"
        )
        
        # Use responsive layout configuration for error figures too
        layout_config = self._get_responsive_layout()
        layout_config.update({
            'showlegend': False,
            'xaxis': dict(showgrid=False, showticklabels=False),
            'yaxis': dict(showgrid=False, showticklabels=False)
        })
        
        fig.update_layout(**layout_config)
        
        return fig
        
    def export_html(self, 
                   fig: go.Figure, 
                   filename: str,
                   include_plotlyjs: str = "cdn") -> bool:
        """
        Export plotly figure to HTML file with full-screen responsive configuration.
        
        Args:
            fig: Plotly figure to export
            filename: Output filename
            include_plotlyjs: How to include plotly.js ('cdn', 'inline', 'directory')
            
        Returns:
            Success status
        """
        try:
            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Standard export for non-responsive mode
            pyo.plot(fig, 
                    filename=str(output_path),
                    auto_open=False,
                    include_plotlyjs=include_plotlyjs)
            
            logger.info(f"Exported standard plot to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting HTML: {e}")
            return False
    
    def _create_metrics_table_with_ranges(self, time_metrics: Dict[str, float], 
                                        freq_metrics: Dict[str, float]) -> List[List[str]]:
        """
        Create enhanced metrics table with reference ranges and status indicators.
        
        Args:
            time_metrics: Time domain HRV metrics
            freq_metrics: Frequency domain HRV metrics
            
        Returns:
            List of metric rows for table display
        """
        if not hrv_reference_ranges or not assess_hrv_value:
            # Fallback to simple table if reference ranges not available
            return [
                ['SDNN (ms)', f"{time_metrics.get('sdnn', 0):.1f}"],
                ['RMSSD (ms)', f"{time_metrics.get('rmssd', 0):.1f}"],
                ['pNN50 (%)', f"{time_metrics.get('pnn50', 0):.1f}"],
                ['Mean HR (BPM)', f"{time_metrics.get('mean_hr', 0):.1f}"],
                ['LF Power (ms²)', f"{freq_metrics.get('lf_power', 0):.0f}"],
                ['HF Power (ms²)', f"{freq_metrics.get('hf_power', 0):.0f}"],
                ['LF/HF Ratio', f"{freq_metrics.get('lf_hf_ratio', 0):.2f}"]
            ]
        
        # Enhanced table with reference ranges
        metrics_data = []
        
        # Define metrics to display
        metric_configs = [
            ('SDNN (ms)', 'sdnn', time_metrics.get('sdnn', 0), 1),
            ('RMSSD (ms)', 'rmssd', time_metrics.get('rmssd', 0), 1),
            ('pNN50 (%)', 'pnn50', time_metrics.get('pnn50', 0), 1),
            ('HF Power (ms²)', 'hf_power', freq_metrics.get('hf_power', 0), 0),
            ('LF Power (ms²)', 'lf_power', freq_metrics.get('lf_power', 0), 0),
            ('LF/HF Ratio', 'lf_hf_ratio', freq_metrics.get('lf_hf_ratio', 0), 2)
        ]
        
        for display_name, metric_key, value, decimals in metric_configs:
            # Get assessment
            assessment = assess_hrv_value(metric_key, value)
            
            # Format value
            value_str = f"{value:.{decimals}f}"
            
            # Add status indicator
            status_indicators = {
                'very_low': '🔴',
                'low': '🟡',
                'normal': '🟢',
                'high': '🟡',
                'very_high': '🔴'
            }
            
            status_icon = status_indicators.get(assessment.get('status', 'unknown'), '⚪')
            
            # Get reference range for display
            ref_range = assessment.get('reference_range')
            if ref_range:
                range_str = f"({ref_range.percentile_25:.{decimals}f}-{ref_range.percentile_75:.{decimals}f})"
            else:
                range_str = ""
            
            # Create enhanced display
            enhanced_value = f"{value_str} {status_icon}"
            
            metrics_data.append([
                display_name,
                enhanced_value,
                f"Normal: {range_str}" if range_str else ""
            ])
        
        return metrics_data
    
    def _add_reference_bands_to_plot(self, fig, metric_key: str, row: int, col: int, 
                                   current_value: float) -> None:
        """
        Add reference range bands to a plot.
        
        Args:
            fig: Plotly figure object
            metric_key: Key for the metric
            row: Subplot row
            col: Subplot column
            current_value: Current measured value
        """
        if not hrv_reference_ranges:
            return
        
        ref_range = hrv_reference_ranges.get_range(metric_key)
        if not ref_range:
            return
        
        # Add normal range band (25th-75th percentile)
        if ref_range.percentile_25 and ref_range.percentile_75:
            fig.add_shape(
                type="rect",
                x0=0, x1=1,
                y0=ref_range.percentile_25, y1=ref_range.percentile_75,
                fillcolor="lightgreen",
                opacity=0.2,
                line=dict(width=0),
                row=row, col=col,
                xref="paper", yref="y"
            )
        
        # Add median line
        if ref_range.percentile_50:
            fig.add_hline(
                y=ref_range.percentile_50,
                line_dash="dash",
                line_color="green",
                annotation_text="Median",
                annotation_position="bottom right",
                row=row, col=col
            )
        
    def _add_reference_bands_to_gam_plot(self, fig, metric: str, row: int, col: int, 
                                       show_legend: bool, x_range: Tuple[float, float] = None) -> None:
        """
        Add reference range bands to GAM plots for clinical interpretation.
        
        Args:
            fig: Plotly figure object
            metric: Full metric name (e.g., 'time_domain_sdnn')
            row: Subplot row
            col: Subplot column  
            show_legend: Whether to show legend entries for this subplot
            x_range: Tuple of (min_x, max_x) for the data range, will auto-detect if None
        """
        if not hrv_reference_ranges:
            logger.debug("HRV reference ranges not available")
            return
        
        # Extract metric key from full metric name
        metric_key = self._extract_metric_key_for_reference(metric)
        if not metric_key:
            logger.debug(f"No reference key found for metric: {metric}")
            return
            
        ref_range = hrv_reference_ranges.get_range(metric_key)
        if not ref_range:
            logger.debug(f"No reference range found for key: {metric_key}")
            return
        
        # Auto-detect X-axis range if not provided
        if x_range is None:
            # Use a reasonable default range (0 to 16 days for Valquiria mission)
            x_range = (0, 16)
            logger.debug(f"Using default X-range for reference bands: {x_range}")
        else:
            logger.debug(f"Using provided X-range for reference bands: {x_range}")
        
        # Add normal range band (25th-75th percentile) spanning full X-axis
        if ref_range.percentile_25 and ref_range.percentile_75:
            # Create a semi-transparent rectangle that spans the entire X-axis
            fig.add_shape(
                type="rect",
                x0=x_range[0], 
                x1=x_range[1],
                y0=ref_range.percentile_25, 
                y1=ref_range.percentile_75,
                fillcolor="rgba(34, 139, 34, 0.15)",  # Light green with transparency
                line=dict(width=0),
                layer='below',  # Ensure it appears behind data points
                row=row, col=col
            )
            
            logger.debug(f"Added normal range band for {metric_key}: "
                        f"Y=[{ref_range.percentile_25:.2f}, {ref_range.percentile_75:.2f}], "
                        f"X=[{x_range[0]}, {x_range[1]}]")
            
            # Add legend entry for normal range (only on first subplot)
            if show_legend:
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(
                            size=12, 
                            color='rgba(34, 139, 34, 0.4)', 
                            symbol='square',
                            line=dict(width=1, color='rgba(34, 139, 34, 0.8)')
                        ),
                        name='Normal Range (25th-75th %ile)',
                        showlegend=True,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )
        
        # Add reference median line spanning full X-axis
        if ref_range.percentile_50:
            fig.add_shape(
                type="line",
                x0=x_range[0],
                x1=x_range[1],
                y0=ref_range.percentile_50,
                y1=ref_range.percentile_50,
                line=dict(
                    color='rgba(34, 139, 34, 0.8)',  # Slightly more opaque green
                    dash='dash',
                    width=2
                ),
                layer='below',  # Behind data but above normal range
                row=row, col=col
            )
            
            logger.debug(f"Added reference median line for {metric_key}: "
                        f"Y={ref_range.percentile_50:.2f}, X=[{x_range[0]}, {x_range[1]}]")
            
            # Add legend entry for median (only on first subplot)
            if show_legend:
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None],
                        mode='lines',
                        line=dict(color='rgba(34, 139, 34, 0.8)', dash='dash', width=2),
                        name='Reference Median',
                        showlegend=True,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )
        
        # Add clinical interpretation annotation (only on first subplot)
        if show_legend and ref_range.percentile_25 and ref_range.percentile_75:
            # Get population info for annotation
            try:
                citation_info = hrv_reference_ranges.get_citation_info(metric_key)
                if citation_info and 'population' in citation_info:
                    pop_text = citation_info['population'][:40] + "..." if len(citation_info['population']) > 40 else citation_info['population']
                    
                    fig.add_annotation(
                        x=0.02,
                        y=0.98,
                        xref="paper",
                        yref="paper",
                        text=f"<b>Normal Range Reference</b><br>"
                             f"Population: {pop_text}<br>"
                             f"25th-75th percentile: {ref_range.percentile_25:.1f} - {ref_range.percentile_75:.1f}",
                        showarrow=False,
                        font=dict(size=9, color='rgba(34, 139, 34, 0.9)'),
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="rgba(34, 139, 34, 0.3)",
                        borderwidth=1,
                        xanchor="left",
                        yanchor="top"
                    )
            except Exception as e:
                logger.debug(f"Could not add clinical annotation: {e}")
    
    def _extract_metric_key_for_reference(self, full_metric_name: str) -> Optional[str]:
        """Extract the metric key for reference range lookup."""
        # Mapping from full metric names to reference range keys
        metric_mapping = {
            'time_domain_sdnn': 'sdnn',
            'time_domain_rmssd': 'rmssd', 
            'time_domain_pnn50': 'pnn50',
            'frequency_domain_hf_power': 'hf_power',
            'frequency_domain_lf_power': 'lf_power',
            'frequency_domain_vlf_power': 'vlf_power',
            'frequency_domain_lf_hf_ratio': 'lf_hf_ratio',
            'frequency_domain_total_power': 'total_power',
            'nonlinear_sd1': 'sd1',
            'nonlinear_sd2': 'sd2',
            'nonlinear_dfa_alpha1': 'dfa_alpha1',
            'sympathetic_stress_index': 'stress_index'
        }
        
        return metric_mapping.get(full_metric_name)

    def get_available_metrics(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Get a list of available HRV metrics from the analysis results."""
        metrics = set()
        for key, result in analysis_results.items():
            if 'hrv_results' in result:
                for domain, domain_metrics in result['hrv_results'].items():
                    if isinstance(domain_metrics, dict):
                        for metric_name in domain_metrics.keys():
                            metrics.add(f"{domain}_{metric_name}")
        return sorted(list(metrics))

    def get_available_metrics_by_domain(self, analysis_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Get available HRV metrics organized by domain for better user experience."""
        domain_metrics = {
            'Time Domain': [],
            'Frequency Domain': [],
            'Nonlinear': [],
            'Parasympathetic': [],
            'Sympathetic': [],
            'ANS Balance': []
        }
        
        domain_mapping = {
            'time_domain': 'Time Domain',
            'frequency_domain': 'Frequency Domain', 
            'nonlinear': 'Nonlinear',
            'parasympathetic': 'Parasympathetic',
            'sympathetic': 'Sympathetic',
            'ans_balance': 'ANS Balance'
        }
        
        logger.debug(f"Searching for metrics in {len(analysis_results)} analysis results")
        
        for key, result in analysis_results.items():
            if 'hrv_results' in result:
                logger.debug(f"Processing {key}, hrv_results domains: {list(result['hrv_results'].keys())}")
                for domain, domain_values in result['hrv_results'].items():
                    if isinstance(domain_values, dict):
                        domain_name = domain_mapping.get(domain, 'Other')
                        logger.debug(f"Domain {domain} -> {domain_name}, metrics: {list(domain_values.keys())}")
                        for metric_name in domain_values.keys():
                            full_metric_name = f"{domain}_{metric_name}"
                            if full_metric_name not in domain_metrics.get(domain_name, []):
                                domain_metrics.setdefault(domain_name, []).append(full_metric_name)
                                logger.debug(f"Added {full_metric_name} to {domain_name}")
        
        # Sort metrics within each domain
        for domain in domain_metrics:
            domain_metrics[domain] = sorted(domain_metrics[domain])
        
        # Log final results
        total_metrics = sum(len(metrics) for metrics in domain_metrics.values())
        logger.info(f"Found {total_metrics} total metrics across {len([d for d in domain_metrics.values() if d])} domains")
        for domain, metrics in domain_metrics.items():
            if metrics:
                logger.debug(f"{domain}: {len(metrics)} metrics - {metrics[:3]}{'...' if len(metrics) > 3 else ''}")
        
        return domain_metrics

    def _get_comprehensive_gam_metrics(self) -> List[str]:
        """Get comprehensive list of GAM metrics covering all HRV domains."""
        return [
            # Time Domain - Core metrics
            'time_domain_sdnn', 'time_domain_rmssd', 'time_domain_pnn50', 'time_domain_mean_hr',
            'time_domain_cvnn', 'time_domain_nn20', 'time_domain_pnn20',
            
            # Frequency Domain - All frequency bands  
            'frequency_domain_lf_power', 'frequency_domain_hf_power', 'frequency_domain_vlf_power',
            'frequency_domain_lf_hf_ratio', 'frequency_domain_lf_nu', 'frequency_domain_hf_nu',
            'frequency_domain_total_power', 'frequency_domain_lf_percent', 'frequency_domain_hf_percent',
            
            # Nonlinear - Poincare and complexity measures
            'nonlinear_sd1', 'nonlinear_sd2', 'nonlinear_sd1_sd2_ratio', 'nonlinear_ellipse_area',
            'nonlinear_dfa_alpha1', 'nonlinear_dfa_alpha2', 'nonlinear_sample_entropy', 
            'nonlinear_triangular_index',
            
            # Parasympathetic - Vagal tone indicators
            'parasympathetic_parasympathetic_index', 'parasympathetic_rsa_amplitude',
            'parasympathetic_vagal_tone_index', 'parasympathetic_respiratory_coherence',
            'parasympathetic_hf_rmssd_ratio',
            
            # Sympathetic - Sympathetic activity indicators  
            'sympathetic_sympathetic_index', 'sympathetic_stress_index', 'sympathetic_autonomic_balance',
            'sympathetic_cardiac_sympathetic_index', 'sympathetic_sympathetic_modulation',
            
            # ANS Balance - Advanced autonomic measures
            'ans_balance_sympathovagal_index', 'ans_balance_ans_complexity',
            'ans_balance_cardiac_autonomic_balance', 'ans_balance_autonomic_reactivity',
            'ans_balance_baroreflex_sensitivity'
        ]

    def _get_recommended_gam_metrics(self) -> Dict[str, List[str]]:
        """Get recommended GAM metrics organized by domain with descriptions."""
        return {
            'Essential Time Domain': [
                'time_domain_sdnn',      # Overall HRV
                'time_domain_rmssd',     # Parasympathetic activity
                'time_domain_pnn50',     # Beat-to-beat variability
                'time_domain_mean_hr'    # Average heart rate
            ],
            'Core Frequency Domain': [
                'frequency_domain_lf_power',    # Low frequency power
                'frequency_domain_hf_power',    # High frequency power  
                'frequency_domain_lf_hf_ratio', # Sympathovagal balance
                'frequency_domain_total_power'  # Overall spectral power
            ],
            'Advanced Nonlinear': [
                'nonlinear_sd1',         # Short-term variability
                'nonlinear_sd2',         # Long-term variability
                'nonlinear_dfa_alpha1',  # Short-term scaling
                'nonlinear_sample_entropy' # Complexity measure
            ],
            'Autonomic Indices': [
                'parasympathetic_parasympathetic_index', # Vagal activity
                'sympathetic_sympathetic_index',         # Sympathetic activity
                'ans_balance_sympathovagal_index',       # Balance measure
                'ans_balance_ans_complexity'             # System complexity
            ]
        }

    def _get_metric_descriptions(self) -> Dict[str, str]:
        """Get comprehensive metric descriptions for GAM analysis."""
        return {
            # Time Domain Descriptions
            'time_domain_sdnn': 'SDNN - Standard deviation of RR intervals (overall HRV)',
            'time_domain_rmssd': 'RMSSD - Root mean square of successive RR differences (parasympathetic)',
            'time_domain_pnn50': 'pNN50 - Percentage of RR intervals differing >50ms (parasympathetic)',
            'time_domain_mean_hr': 'Mean HR - Average heart rate in beats per minute',
            'time_domain_cvnn': 'CVNN - Coefficient of variation of RR intervals',
            'time_domain_nn20': 'NN20 - Count of RR intervals differing >20ms',
            'time_domain_pnn20': 'pNN20 - Percentage of RR intervals differing >20ms',
            
            # Frequency Domain Descriptions
            'frequency_domain_lf_power': 'LF Power - Low frequency spectral power (0.04-0.15 Hz)',
            'frequency_domain_hf_power': 'HF Power - High frequency spectral power (0.15-0.4 Hz)',
            'frequency_domain_vlf_power': 'VLF Power - Very low frequency spectral power (0.003-0.04 Hz)',
            'frequency_domain_lf_hf_ratio': 'LF/HF Ratio - Sympathovagal balance indicator',
            'frequency_domain_lf_nu': 'LF (n.u.) - LF power in normalized units',
            'frequency_domain_hf_nu': 'HF (n.u.) - HF power in normalized units',
            'frequency_domain_total_power': 'Total Power - Total spectral power across all bands',
            
            # Nonlinear Descriptions
            'nonlinear_sd1': 'SD1 - Poincaré plot width (short-term variability)',
            'nonlinear_sd2': 'SD2 - Poincaré plot length (long-term variability)', 
            'nonlinear_sd1_sd2_ratio': 'SD1/SD2 - Poincaré ratio (autonomic balance)',
            'nonlinear_ellipse_area': 'Ellipse Area - Poincaré plot ellipse area',
            'nonlinear_dfa_alpha1': 'DFA α1 - Short-term detrended fluctuation scaling',
            'nonlinear_dfa_alpha2': 'DFA α2 - Long-term detrended fluctuation scaling',
            'nonlinear_sample_entropy': 'SampEn - Sample entropy (signal regularity)',
            'nonlinear_triangular_index': 'TINN - Triangular interpolation index',
            
            # Parasympathetic Descriptions
            'parasympathetic_parasympathetic_index': 'Para Index - Composite parasympathetic activity',
            'parasympathetic_vagal_tone_index': 'Vagal Tone - Composite vagal tone measure',
            'parasympathetic_rsa_amplitude': 'RSA Amplitude - Respiratory sinus arrhythmia strength',
            'parasympathetic_respiratory_coherence': 'Resp Coherence - Respiratory-cardiac coupling',
            
            # Sympathetic Descriptions
            'sympathetic_sympathetic_index': 'Sympa Index - Composite sympathetic activity',
            'sympathetic_stress_index': 'Stress Index - Baevsky stress indicator',
            'sympathetic_cardiac_sympathetic_index': 'Cardiac Sympa - Cardiac sympathetic activity',
            
            # ANS Balance Descriptions
            'ans_balance_sympathovagal_index': 'Sympathovagal - Advanced autonomic balance',
            'ans_balance_ans_complexity': 'ANS Complexity - Autonomic system complexity',
            'ans_balance_cardiac_autonomic_balance': 'Cardiac Balance - Cardiac autonomic balance',
            'ans_balance_autonomic_reactivity': 'ANS Reactivity - Autonomic responsiveness',
            'ans_balance_baroreflex_sensitivity': 'Baroreflex Sens - Baroreflex sensitivity estimate'
        } 

    def create_elegant_metric_selector(self, analysis_results: Dict[str, Any]) -> go.Figure:
        """
        Create a publication-quality time series visualization with statistical aggregation.
        
        Shows crew-wide statistics across Sol:
        - Median values with smoothed trend curves
        - 95% Confidence intervals 
        - Interquartile range (IQR) shading
        - Professional scientific journal styling
        
        This approach provides publication-ready visualizations focusing on population
        trends rather than individual variability, suitable for scientific manuscripts.
        """
        logger.info("Creating publication-quality metric-focused statistical visualization")
        
        # Log analysis results structure for debugging
        logger.debug(f"Analysis results keys: {list(analysis_results.keys())}")
        sample_key = list(analysis_results.keys())[0] if analysis_results else None
        if sample_key:
            sample_result = analysis_results[sample_key]
            logger.debug(f"Sample result structure: {list(sample_result.keys())}")
            if 'hrv_results' in sample_result:
                logger.debug(f"Sample HRV results domains: {list(sample_result['hrv_results'].keys())}")
        
        # Get comprehensive metrics organized by domain
        available_metrics_by_domain = self.get_available_metrics_by_domain(analysis_results)
        
        if not any(available_metrics_by_domain.values()):
            logger.error("No HRV metrics found in any domain")
            logger.error(f"Available domains searched: {list(available_metrics_by_domain.keys())}")
            raise ValueError("No HRV metrics available for visualization. Check that HRV analysis has been run successfully.")
        
        logger.info(f"Found metrics in {len([d for d, m in available_metrics_by_domain.items() if m])} domains")
        
        # Create figure
        fig = go.Figure()
        
        # Prepare statistical data for all metrics
        statistical_data = self._prepare_statistical_metrics_data(analysis_results, available_metrics_by_domain)
        
        if not statistical_data:
            logger.error("No statistical metric data could be calculated")
            logger.error("This usually means:")
            logger.error("1. HRV analysis hasn't been run yet")
            logger.error("2. Analysis results don't contain expected structure")
            logger.error("3. All metrics have NaN values")
            raise ValueError("No valid statistical data found for visualization. Please run HRV analysis first.")
        
        logger.info(f"Successfully calculated statistics for {len(statistical_data)} metrics")
        
        # Add all statistical traces for each metric (make only first metric visible initially)
        first_metric = None
        trace_index = 0
        buttons = []
        
        for metric_name, stats in statistical_data.items():
            if first_metric is None:
                first_metric = metric_name
            
            is_visible = (metric_name == first_metric)
            
            # Create visibility indices for this metric's traces
            metric_trace_indices = []
            
            # 1. IQR shading (Q1 to Q3)
            fig.add_trace(go.Scatter(
                x=stats['sols'] + stats['sols'][::-1],  # x, then x reversed
                y=stats['q3'] + stats['q1'][::-1],      # upper, then lower reversed
                fill='toself',
                fillcolor='rgba(135, 206, 235, 0.2)',  # Light blue with transparency
                line=dict(color='rgba(255,255,255,0)'),  # Invisible line
                hoverinfo="skip",
                showlegend=True,
                name="IQR (25%-75%)",
                visible=is_visible
            ))
            metric_trace_indices.append(trace_index)
            trace_index += 1
            
            # 2. 95% CI shading
            fig.add_trace(go.Scatter(
                x=stats['sols'] + stats['sols'][::-1],
                y=stats['ci_upper'] + stats['ci_lower'][::-1],
                fill='toself',
                fillcolor='rgba(70, 130, 180, 0.15)',   # Steel blue with transparency
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name="95% CI",
                visible=is_visible
            ))
            metric_trace_indices.append(trace_index)
            trace_index += 1
            
            # 3. Median points
            fig.add_trace(go.Scatter(
                x=stats['sols'],
                y=stats['median'],
                mode='markers',
                marker=dict(
                    size=8,
                    color='#1f77b4',  # Professional blue
                    symbol='circle',
                    line=dict(width=2, color='white')
                ),
                name="Median",
                hovertemplate=(
                    f'<b>{self._get_metric_display_name(metric_name)}</b><br>'
                    f'Sol: %{{x:.0f}}<br>'
                    f'Median: %{{y:.2f}}<br>'
                    f'IQR: %{{customdata[0]:.2f}} - %{{customdata[1]:.2f}}<br>'
                    f'95% CI: %{{customdata[2]:.2f}} - %{{customdata[3]:.2f}}<br>'
                    f'N subjects: %{{customdata[4]:.0f}}<br>'
                    '<extra></extra>'
                ),
                customdata=list(zip(stats['q1'], stats['q3'], stats['ci_lower'], stats['ci_upper'], stats['n_subjects'])),
                visible=is_visible
            ))
            metric_trace_indices.append(trace_index)
            trace_index += 1
            
            # 4. Smoothed trend curve
            if len(stats['sols']) >= 4:  # Need at least 4 points for smoothing
                try:
                    from scipy.interpolate import UnivariateSpline
                    
                    # Create smoothed curve
                    sols_smooth = np.linspace(min(stats['sols']), max(stats['sols']), len(stats['sols']) * 3)
                    spline = UnivariateSpline(stats['sols'], stats['median'], s=len(stats['sols']) * 0.1)
                    median_smooth = spline(sols_smooth)
                    
                    fig.add_trace(go.Scatter(
                        x=sols_smooth,
                        y=median_smooth,
                        mode='lines',
                        line=dict(
                            width=3,
                            color='#d62728',  # Professional red
                            dash='solid'
                        ),
                        name="Trend (Smoothed)",
                        hovertemplate=(
                            f'<b>Smoothed Trend</b><br>'
                            f'Sol: %{{x:.1f}}<br>'
                            f'Value: %{{y:.2f}}<br>'
                            '<extra></extra>'
                        ),
                        visible=is_visible
                    ))
                    metric_trace_indices.append(trace_index)
                    trace_index += 1
                except ImportError:
                    # Fallback: simple linear trend
                    z = np.polyfit(stats['sols'], stats['median'], 1)
                    trend_line = np.poly1d(z)
                    
                    fig.add_trace(go.Scatter(
                        x=stats['sols'],
                        y=trend_line(stats['sols']),
                        mode='lines',
                        line=dict(
                            width=2,
                            color='#d62728',
                            dash='dash'
                        ),
                        name="Linear Trend",
                        hovertemplate=(
                            f'<b>Linear Trend</b><br>'
                            f'Sol: %{{x:.0f}}<br>'
                            f'Value: %{{y:.2f}}<br>'
                            '<extra></extra>'
                        ),
                        visible=is_visible
                    ))
                    metric_trace_indices.append(trace_index)
                    trace_index += 1
            
            # Create dropdown button for this metric
            if metric_trace_indices:
                # Create visibility array (False for all traces, True for this metric's traces)
                visibility = [False] * trace_index
                for idx in metric_trace_indices:
                    visibility[idx] = True
                
                # Get metric unit for Y-axis label
                unit = self._get_metric_unit(metric_name.split('_')[-1]) if '_' in metric_name else ""
                y_title = f"{self._get_metric_display_name(metric_name)}"
                if unit:
                    y_title += f" ({unit})"
                
                button = dict(
                    label=self._get_metric_display_name(metric_name),
                    method="update",
                    args=[
                        {"visible": visibility},
                        {
                            "yaxis.title": y_title,
                            "title": f'<b>Crew HRV Statistical Analysis - {self._get_metric_display_name(metric_name)}</b><br>'
                                    f'<span style="font-size:14px">Population statistics across mission timeline (N={stats["total_subjects"]} crew members)</span>'
                        }
                    ]
                )
                buttons.append(button)
        
        if not buttons:
            logger.error("No dropdown buttons could be created - no metrics have statistical data")
            raise ValueError("No metrics with valid statistical data found for visualization")
        
        logger.info(f"Created {len(buttons)} dropdown options for statistical metric analysis")
        
        # Calculate Sol range for X-axis formatting
        all_sols = []
        for stats in statistical_data.values():
            all_sols.extend(stats['sols'])
        
        sol_range = (min(all_sols), max(all_sols)) if all_sols else (0, 10)
        logger.info(f"X-axis Sol range: {sol_range[0]} to {sol_range[1]}")
        
        # Get first metric info for initial display
        first_stats = statistical_data[first_metric] if first_metric else None
        first_unit = self._get_metric_unit(first_metric.split('_')[-1]) if first_metric and '_' in first_metric else ""
        first_y_title = f"{self._get_metric_display_name(first_metric)}"
        if first_unit:
            first_y_title += f" ({first_unit})"
        
        # Professional scientific journal layout
        fig.update_layout(
            title=dict(
                text=f'<b>Crew HRV Statistical Analysis - {self._get_metric_display_name(first_metric) if first_metric else "HRV Metrics"}</b><br>'
                     f'<span style="font-size:14px">Population statistics across mission timeline (N={first_stats["total_subjects"] if first_stats else "X"} crew members)</span>',
                x=0.5,
                font=dict(size=18, color='#1a1a1a', family="Arial")
            ),
            xaxis=dict(
                title='Mission Sol (Days)',
                title_font=dict(size=14, color='#1a1a1a', family="Arial"),
                tickfont=dict(size=12, color='#1a1a1a', family="Arial"),
                showgrid=True,
                gridcolor='rgba(128,128,128,0.3)',
                gridwidth=1,
                zeroline=True,
                zerolinecolor='rgba(128,128,128,0.5)',
                zerolinewidth=1,
                # Set explicit range to ensure proper display of Sol values
                range=[sol_range[0] - 0.5, sol_range[1] + 0.5] if sol_range[1] > sol_range[0] else [0, 10],
                tickmode='linear',
                dtick=max(1, (sol_range[1] - sol_range[0]) / 10) if sol_range[1] > sol_range[0] else 1,
                tickformat='.0f',
                linecolor='#1a1a1a',
                linewidth=1,
                mirror=True
            ),
            yaxis=dict(
                title=first_y_title,
                title_font=dict(size=14, color='#1a1a1a', family="Arial"),
                tickfont=dict(size=12, color='#1a1a1a', family="Arial"),
                showgrid=True,
                gridcolor='rgba(128,128,128,0.3)',
                gridwidth=1,
                zeroline=False,
                linecolor='#1a1a1a',
                linewidth=1,
                mirror=True
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.95,
                xanchor="right",
                x=0.98,
                font=dict(size=11, color='#1a1a1a', family="Arial"),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#cccccc",
                borderwidth=1
            ),
            width=1200,
            height=700,
            margin=dict(l=100, r=120, t=120, b=80),
            font=dict(family="Arial, sans-serif", size=12, color="#1a1a1a"),
            updatemenus=[{
                'buttons': buttons,
                'direction': 'down',
                'pad': {'r': 10, 't': 10},
                'showactive': True,
                'x': 0.02,
                'xanchor': 'left',
                'y': 1.08,
                'yanchor': 'top',
                'bgcolor': '#f8f9fa',
                'bordercolor': '#dee2e6',
                'borderwidth': 1,
                'font': dict(size=12, color='#1a1a1a', family="Arial")
            }],
            annotations=[
                dict(
                    text="<b>Select HRV Metric:</b>",
                    x=0.02,
                    y=1.12,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=13, color='#1a1a1a', family="Arial"),
                    xanchor="left"
                ),
                dict(
                    text="📊 Statistical visualization: Median ± IQR ± 95% CI with trend analysis",
                    x=0.98,
                    y=1.12,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=11, color='#6c757d', family="Arial"),
                    xanchor="right"
                )
            ]
        )
        
        return fig
    
    def _prepare_statistical_metrics_data(self, analysis_results: Dict[str, Any], 
                                         available_metrics_by_domain: Dict[str, List[str]]) -> Dict[str, Dict]:
        """
        Prepare statistical metrics data for publication-quality visualization.
        
        Calculates crew-wide statistics for each Sol day:
        - Median values
        - Interquartile range (Q1, Q3)
        - 95% Confidence intervals
        - Sample sizes
        
        Returns:
            Dict with statistical data for each metric organized by Sol
        """
        from scipy import stats
        import numpy as np
        
        logger.debug(f"Calculating statistical aggregation for metrics by domain: {available_metrics_by_domain}")
        
        # First, collect all raw data organized by metric and Sol
        raw_metrics_by_sol = {}
        all_subjects = set()
        
        for subject_session, results in analysis_results.items():
            if not isinstance(results, dict) or 'hrv_results' not in results:
                continue
                
            hrv_results = results['hrv_results']
            
            # Extract Sol number and subject name
            if '_Sol' in subject_session:
                try:
                    subject_name, sol_str = subject_session.rsplit('_Sol', 1)
                    sol_number = float(sol_str)
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse Sol number from {subject_session}, skipping")
                    continue
            else:
                sol_number = results.get('sol_number', 0)
                subject_name = subject_session.split('_')[0] if '_' in subject_session else subject_session
                
            all_subjects.add(subject_name)
            
            logger.debug(f"Processing {subject_session}: sol={sol_number}, subject={subject_name}")
            
            # Extract metrics from all domains
            for domain_name, domain_metrics in available_metrics_by_domain.items():
                domain_key = domain_name.lower().replace(' ', '_')
                
                if domain_key in hrv_results and isinstance(hrv_results[domain_key], dict):
                    domain_data = hrv_results[domain_key]
                    
                    for full_metric_name in domain_metrics:
                        # Extract the metric key from the full name
                        if full_metric_name.startswith(f"{domain_key}_"):
                            metric_key = full_metric_name[len(domain_key) + 1:]
                        else:
                            metric_key = full_metric_name
                        
                        # Try different variations of the metric key
                        possible_keys = [metric_key, metric_key.lower(), metric_key.replace(' ', '_').lower()]
                        
                        metric_value = None
                        for key in possible_keys:
                            if key in domain_data:
                                try:
                                    value = float(domain_data[key])
                                    if not np.isnan(value):
                                        metric_value = value
                                        break
                                except (ValueError, TypeError):
                                    continue
                        
                        if metric_value is not None:
                            # Initialize metric structure if needed
                            if full_metric_name not in raw_metrics_by_sol:
                                raw_metrics_by_sol[full_metric_name] = {}
                            
                            if sol_number not in raw_metrics_by_sol[full_metric_name]:
                                raw_metrics_by_sol[full_metric_name][sol_number] = []
                            
                            raw_metrics_by_sol[full_metric_name][sol_number].append(metric_value)
        
        logger.info(f"Collected raw data for {len(raw_metrics_by_sol)} metrics across {len(all_subjects)} subjects")
        
        # Calculate statistics for each metric
        statistical_data = {}
        
        for metric_name, sol_data in raw_metrics_by_sol.items():
            if not sol_data:
                continue
                
            # Get sorted list of Sol days
            sols = sorted(sol_data.keys())
            
            medians = []
            q1s = []
            q3s = []
            ci_lowers = []
            ci_uppers = []
            n_subjects_list = []
            
            for sol in sols:
                values = np.array(sol_data[sol])
                n = len(values)
                
                if n == 0:
                    continue
                    
                # Calculate basic statistics
                median_val = np.median(values)
                q1_val = np.percentile(values, 25)
                q3_val = np.percentile(values, 75)
                
                # Calculate 95% confidence interval for the median
                if n >= 3:  # Need at least 3 points for meaningful CI
                    try:
                        # Bootstrap confidence interval for median
                        bootstrap_medians = []
                        n_bootstrap = min(1000, max(100, n * 10))  # Adaptive bootstrap samples
                        
                        np.random.seed(42)  # For reproducibility
                        for _ in range(n_bootstrap):
                            bootstrap_sample = np.random.choice(values, size=n, replace=True)
                            bootstrap_medians.append(np.median(bootstrap_sample))
                        
                        bootstrap_medians = np.array(bootstrap_medians)
                        ci_lower = np.percentile(bootstrap_medians, 2.5)
                        ci_upper = np.percentile(bootstrap_medians, 97.5)
                        
                    except Exception as e:
                        logger.warning(f"Bootstrap CI calculation failed for {metric_name} Sol {sol}: {e}")
                        # Fallback: use standard error approximation
                        se = stats.sem(values)
                        ci_lower = median_val - 1.96 * se
                        ci_upper = median_val + 1.96 * se
                else:
                    # For small samples, use wider intervals based on data range
                    data_range = np.ptp(values) if n > 1 else 0
                    ci_lower = median_val - data_range * 0.5
                    ci_upper = median_val + data_range * 0.5
                
                medians.append(median_val)
                q1s.append(q1_val)
                q3s.append(q3_val)
                ci_lowers.append(ci_lower)
                ci_uppers.append(ci_upper)
                n_subjects_list.append(n)
            
            if medians:  # Only include metrics with valid data
                statistical_data[metric_name] = {
                    'sols': sols,
                    'median': medians,
                    'q1': q1s,
                    'q3': q3s,
                    'ci_lower': ci_lowers,
                    'ci_upper': ci_uppers,
                    'n_subjects': n_subjects_list,
                    'total_subjects': len(all_subjects)
                }
                
                logger.debug(f"Calculated statistics for {metric_name}: {len(sols)} Sol days, "
                           f"median range {min(medians):.2f}-{max(medians):.2f}")
        
        logger.info(f"Successfully calculated statistical data for {len(statistical_data)} metrics")
        
        # Log summary statistics
        if statistical_data:
            sample_metric = list(statistical_data.keys())[0]
            sample_stats = statistical_data[sample_metric]
            logger.info(f"Sample metric '{sample_metric}': {len(sample_stats['sols'])} Sol days, "
                       f"N={sample_stats['total_subjects']} subjects")
        
        return statistical_data
    
    def _get_metric_display_name(self, metric_name: str) -> str:
        """Convert metric name to display-friendly format."""
        # Remove domain prefix if present
        if '_' in metric_name and len(metric_name.split('_')) > 1:
            clean_name = '_'.join(metric_name.split('_')[1:])
        else:
            clean_name = metric_name
        
        # Mapping for common metrics
        display_mapping = {
            'sdnn': 'SDNN (Standard Deviation of RR Intervals)',
            'rmssd': 'RMSSD (Root Mean Square of Successive Differences)',
            'pnn50': 'pNN50 (% of RR intervals differing >50ms)',
            'mean_hr': 'Mean Heart Rate (BPM)',
            'lf_power': 'LF Power (Low Frequency Spectral Power)',
            'hf_power': 'HF Power (High Frequency Spectral Power)', 
            'lf_hf_ratio': 'LF/HF Ratio (Sympathovagal Balance)',
            'total_power': 'Total Spectral Power',
            'vlf_power': 'VLF Power (Very Low Frequency)',
            'sd1': 'SD1 (Poincaré Plot Width)',
            'sd2': 'SD2 (Poincaré Plot Length)',
            'sample_entropy': 'Sample Entropy',
            'dfa_alpha1': 'DFA α1 (Short-term Scaling)',
            'dfa_alpha2': 'DFA α2 (Long-term Scaling)',
            'parasympathetic_index': 'Parasympathetic Activity Index',
            'sympathetic_index': 'Sympathetic Activity Index',
            'stress_index': 'Stress Index',
            'ans_complexity': 'ANS Complexity Index'
        }
        
        return display_mapping.get(clean_name.lower(), clean_name.replace('_', ' ').title())
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get the unit for a metric."""
        units_mapping = {
            'sdnn': 'ms',
            'rmssd': 'ms',
            'pnn50': '%',
            'mean_hr': 'BPM',
            'lf_power': 'ms²',
            'hf_power': 'ms²',
            'vlf_power': 'ms²',
            'total_power': 'ms²',
            'lf_hf_ratio': 'ratio',
            'sd1': 'ms',
            'sd2': 'ms',
            'sample_entropy': 'au',
            'dfa_alpha1': 'au',
            'dfa_alpha2': 'au'
        }
        return units_mapping.get(metric_name.lower(), 'au')  # au = arbitrary units