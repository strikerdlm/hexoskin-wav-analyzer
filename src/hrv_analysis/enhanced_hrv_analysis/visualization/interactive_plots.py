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
import plotly.express as px
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
            # Default metrics to plot if none specified
            if metrics_to_plot is None:
                metrics_to_plot = [
                    'time_domain_sdnn', 'time_domain_rmssd', 'time_domain_pnn50', 'time_domain_mean_hr',
                    'frequency_domain_lf_power', 'frequency_domain_hf_power', 'frequency_domain_lf_hf_ratio',
                    'frequency_domain_lf_nu', 'frequency_domain_hf_nu'
                ]
            
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
            # Default to key HRV metrics if none specified
            if metrics_to_plot is None:
                metrics_to_plot = [
                    'time_domain_rmssd', 'time_domain_sdnn', 'time_domain_pnn50',
                    'frequency_domain_lf_power', 'frequency_domain_hf_power', 
                    'frequency_domain_lf_hf_ratio'
                ]
            
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
                        # Add confidence interval first (so it appears behind the line)
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
                        
                        # Add GAM trend line
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
            # Prepare data
            df = pd.DataFrame({'x': x_data, 'y': y_data})
            df = df.sort_values('x')
            
            # Create smoothing spline for GAM
            x_spline = BSplines(df[['x']], df=[6], degree=[3])
            
            # Fit GAM model
            gam_model = GLMGam(df['y'], smoother=x_spline, family=Gaussian())
            gam_results = gam_model.fit()
            
            # Create smooth x values for prediction
            x_range = np.linspace(df['x'].min(), df['x'].max(), 50)
            x_smooth_df = pd.DataFrame({'x': x_range})
            x_smooth_spline = BSplines(x_smooth_df[['x']], df=[6], degree=[3])
            
            # Predict with confidence intervals
            predictions = gam_results.get_prediction(x_smooth_spline)
            y_smooth = predictions.predicted_mean
            conf_int = predictions.conf_int(alpha=1-confidence_level)
            
            return {
                'x_smooth': x_range,
                'y_smooth': y_smooth,
                'lower_ci': conf_int[:, 0],
                'upper_ci': conf_int[:, 1]
            }
            
        except Exception as e:
            logger.warning(f"GAM fitting failed: {e}, using polynomial fallback")
            return self._fit_polynomial_trend(x_data, y_data, confidence_level)
    
    def _fit_polynomial_trend(self, x_data: np.ndarray, y_data: np.ndarray, confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """Fallback polynomial trend fitting with confidence intervals."""
        try:
            # Sort data
            sorted_indices = np.argsort(x_data)
            x_sorted = x_data[sorted_indices]
            y_sorted = y_data[sorted_indices]
            
            # Fit polynomial (degree 2 for smooth curve)
            degree = min(2, len(x_data) - 1)
            poly_coeffs = np.polyfit(x_sorted, y_sorted, degree)
            
            # Create smooth x values
            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 50)
            y_smooth = np.polyval(poly_coeffs, x_smooth)
            
            # Calculate confidence intervals using bootstrap
            n_bootstrap = 100
            predictions = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(x_sorted), len(x_sorted), replace=True)
                x_boot = x_sorted[indices]
                y_boot = y_sorted[indices]
                
                # Fit polynomial
                boot_coeffs = np.polyfit(x_boot, y_boot, degree)
                boot_pred = np.polyval(boot_coeffs, x_smooth)
                predictions.append(boot_pred)
            
            predictions = np.array(predictions)
            alpha = 1 - confidence_level
            lower_ci = np.percentile(predictions, 100 * alpha/2, axis=0)
            upper_ci = np.percentile(predictions, 100 * (1-alpha/2), axis=0)
            
            return {
                'x_smooth': x_smooth,
                'y_smooth': y_smooth,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci
            }
            
        except Exception as e:
            logger.error(f"Polynomial trend fitting failed: {e}")
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
                
                metrics_data = [
                    ['SDNN (ms)', f"{time_metrics.get('sdnn', 0):.1f}"],
                    ['RMSSD (ms)', f"{time_metrics.get('rmssd', 0):.1f}"],
                    ['pNN50 (%)', f"{time_metrics.get('pnn50', 0):.1f}"],
                    ['Mean HR (BPM)', f"{time_metrics.get('mean_hr', 0):.1f}"],
                    ['LF Power (ms²)', f"{freq_metrics.get('lf_power', 0):.0f}"],
                    ['HF Power (ms²)', f"{freq_metrics.get('hf_power', 0):.0f}"],
                    ['LF/HF Ratio', f"{freq_metrics.get('lf_hf_ratio', 0):.2f}"]
                ]
                
                fig.add_trace(
                    go.Table(
                        header=dict(values=['Metric', 'Value'],
                                  fill_color=self.color_palette[0],
                                  font=dict(color='white', size=12)),
                        cells=dict(values=list(zip(*metrics_data)),
                                 fill_color='white',
                                 font=dict(size=11))
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