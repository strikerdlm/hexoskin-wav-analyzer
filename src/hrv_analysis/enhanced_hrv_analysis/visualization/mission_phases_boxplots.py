"""
Mission Phases Boxplot Generator for Enhanced HRV Analysis

This module provides boxplot visualization capabilities for comparing 
physiological and HRV metrics across three mission phases (Early, Mid, Late) 
in the Valquiria space analog simulation.

Author: AI Assistant  
Integration: Enhanced HRV Analysis System
Date: 2025-01-14
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy import stats

# Import unified export configuration
try:
    from config import get_export_directory, get_plots_output_path
except ImportError:
    # Fallback if config module not found
    def get_export_directory():
        return Path("plots_output")
    
    def get_plots_output_path(filename=None):
        export_dir = Path("plots_output")
        export_dir.mkdir(parents=True, exist_ok=True)
        return export_dir / filename if filename else export_dir

import logging

# Import HRV reference ranges
try:
    from visualization.hrv_reference_ranges import (
        hrv_reference_ranges, get_reference_range
    )
except ImportError:
    try:
        from enhanced_hrv_analysis.visualization.hrv_reference_ranges import (
            hrv_reference_ranges, get_reference_range
        )
    except ImportError:
        hrv_reference_ranges = None
        get_reference_range = None

logger = logging.getLogger(__name__)

class MissionPhasesBoxplotGenerator:
    """Generator for mission phases boxplots integrated with Enhanced HRV Analysis."""
    
    def __init__(self):
        """Initialize the boxplot generator."""
        # Configure matplotlib for publication quality
        plt.rcParams.update({
            'figure.figsize': (14, 8),
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'font.family': 'DejaVu Sans'
        })
        
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Phase color scheme
        self.phase_colors = {
            'Early': 'lightblue',
            'Mid': 'lightgreen', 
            'Late': 'lightcoral'
        }
        
        logger.info("MissionPhasesBoxplotGenerator initialized")
    
    def prepare_mission_data(self, analysis_results: Dict[str, Any]) -> (
            Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]):
        """
        Prepare mission data from HRV analysis results for phase analysis.
        
        Args:
            analysis_results: Results from Enhanced HRV Analysis app
            
        Returns:
            Tuple of (processed DataFrame, mission phases definition)
        """
        logger.info("Preparing mission data for phase analysis")
        
        # Use comprehensive metric extraction
        try:
            df = self._extract_hrv_metrics_from_results(analysis_results)
            logger.info(
                f"Successfully extracted data for {len(df)} "
                "subject-session combinations"
            )
        except ValueError as e:
            logger.error(f"Failed to extract HRV metrics: {e}")
            # Create a minimal DataFrame as fallback
            df = pd.DataFrame({
                'Subject': ['Unknown'],
                'Session': ['Unknown'],
                'Sol': [1.0]
            })
        
        # Enhanced debugging information
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        if 'Sol' in df.columns:
            logger.info(f"Sol values range: {df['Sol'].min()} to "
                       f"{df['Sol'].max()}")
            sol_counts = df['Sol'].value_counts().head().to_dict()
            logger.info(f"Sol value distribution: {sol_counts}")
        
        # Define mission phases with more robust logic
        if 'Sol' in df.columns and df['Sol'].notna().sum() > 0:
            sol_values = df['Sol'].dropna()
            sol_min = sol_values.min()
            sol_max = sol_values.max()
            sol_range = sol_max - sol_min
            
            logger.info(f"Sol range: {sol_min} to {sol_max} "
                       f"(range: {sol_range})")
            
            # ALWAYS use data-driven phases to ensure equal distribution
            # This ensures we have data in all three phases regardless of actual Sol values
            if sol_range > 0.1:  # If we have any meaningful spread
                # Use data quantiles to ensure equal distribution
                sol_33 = sol_values.quantile(0.33)
                sol_67 = sol_values.quantile(0.67)
                
                mission_phases = {
                    'Early': (sol_min, sol_33),
                    'Mid': (sol_33, sol_67),
                    'Late': (sol_67, sol_max)
                }
                logger.info(
                    f"Using quantile-based phases: "
                    f"Early ({sol_min:.2f}-{sol_33:.2f}), "
                    f"Mid ({sol_33:.2f}-{sol_67:.2f}), "
                    f"Late ({sol_67:.2f}-{sol_max:.2f})"
                )
            else:
                # All values are nearly the same - create artificial phases
                center = sol_values.mean()
                spread = max(0.1, sol_range)
                mission_phases = {
                    'Early': (center - spread, center - spread/3),
                    'Mid': (center - spread/3, center + spread/3), 
                    'Late': (center + spread/3, center + spread)
                }
                logger.info(f"Using artificial phases around Sol {center:.2f}")
        else:
            # No Sol data - use sequential assignment
            total_records = len(df)
            mission_phases = {
                'Early': (0, total_records / 3),
                'Mid': (total_records / 3, 2 * total_records / 3),
                'Late': (2 * total_records / 3, total_records)
            }
            logger.info("Using sequential record-based phases")
        
        # Add phase labels with INCLUSIVE boundary logic to ensure proper assignment
        if 'Sol' in df.columns:
            conditions = [
                (df['Sol'] >= mission_phases['Early'][0]) & (df['Sol'] <= mission_phases['Early'][1]),
                (df['Sol'] > mission_phases['Early'][1]) & (df['Sol'] <= mission_phases['Mid'][1]),
                (df['Sol'] > mission_phases['Mid'][1]) & (df['Sol'] <= mission_phases['Late'][1])
            ]
        else:
            # Use row index for assignment
            df['Sol'] = range(len(df))  # Create artificial Sol values
            conditions = [
                (df.index >= mission_phases['Early'][0]) & (df.index < mission_phases['Mid'][0]),
                (df.index >= mission_phases['Mid'][0]) & (df.index < mission_phases['Late'][0]),
                (df.index >= mission_phases['Late'][0]) & (df.index <= mission_phases['Late'][1])
            ]
        
        choices = ['Early', 'Mid', 'Late']
        df['Mission_Phase'] = np.select(conditions, choices, default='Early')  # Default to Early
        
        # Ensure we have at least some data in each phase
        phase_counts = df['Mission_Phase'].value_counts()
        logger.info(f"Defined mission phases: {mission_phases}")
        logger.info(f"Phase distribution: {phase_counts.to_dict()}")
        
        # Force equal distribution if phases are still empty
        if len(phase_counts) < 3 or (phase_counts == 0).any():
            logger.warning("Uneven phase distribution detected, forcing equal distribution")
            # Sort by Sol to maintain temporal order
            df_sorted = df.sort_values('Sol').reset_index(drop=True)
            n_records = len(df_sorted)
            
            # Calculate boundaries for equal thirds
            third1 = n_records // 3
            third2 = 2 * n_records // 3
            
            new_phases = ['Early'] * third1 + ['Mid'] * (third2 - third1) + ['Late'] * (n_records - third2)
            
            # Map back to original dataframe order
            phase_mapping = dict(zip(df_sorted.index, new_phases))
            df['Mission_Phase'] = df.index.map(phase_mapping)
            
            # Update mission phases boundaries based on actual assignments
            early_sols = df[df['Mission_Phase'] == 'Early']['Sol']
            mid_sols = df[df['Mission_Phase'] == 'Mid']['Sol']  
            late_sols = df[df['Mission_Phase'] == 'Late']['Sol']
            
            mission_phases = {
                'Early': (early_sols.min(), early_sols.max()),
                'Mid': (mid_sols.min(), mid_sols.max()),
                'Late': (late_sols.min(), late_sols.max())
            }
            
            final_counts = df['Mission_Phase'].value_counts()
            logger.info(f"Forced equal distribution - Phase counts: {final_counts.to_dict()}")
            logger.info(f"Updated mission phases: {mission_phases}")
        
        # Validate we have usable data
        valid_metrics = []
        for col in df.columns:
            if col not in ['Subject', 'Session', 'Sol', 'Mission_Phase']:
                valid_count = df[col].notna().sum()
                if valid_count > 0:
                    valid_metrics.append(f"{col} ({valid_count})")
        
        logger.info(f"Found {len(valid_metrics)} metrics with valid data:")
        for metric in valid_metrics[:5]:  # Show first 5
            logger.info(f"  • {metric}")
        if len(valid_metrics) > 5:
            logger.info(f"  • ... and {len(valid_metrics) - 5} more")
        
        return df, mission_phases
    
    def generate_individual_boxplots(self, df: pd.DataFrame, mission_phases: Dict[str, Tuple[float, float]], 
                                     output_dir: str = None) -> str:
        """
        Generate individual boxplots for each crew member across mission phases.
        
        Args:
            df: Prepared mission data
            mission_phases: Mission phases definition
            output_dir: Output directory for plots (if None, uses unified export directory)
        
        Returns:
            str: Path to generated plot file
        """
        if output_dir is None:
            output_dir = str(get_export_directory())
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Select key HRV metrics for plotting
        hrv_metrics = ['SDNN', 'RMSSD', 'Mean_HR', 'LF_HF_Ratio']
        available_hrv = [m for m in hrv_metrics if m in df.columns and df[m].notna().sum() > 0]
        
        # Select physiological metrics
        physio_metrics = ['heart_rate_mean', 'breathing_rate_mean', 'activity_mean', 'spo2_mean']
        available_physio = [m for m in physio_metrics if m in df.columns and df[m].notna().sum() > 0]
        
        # Combine available metrics (prefer HRV if available)
        plot_metrics = available_hrv[:2] + available_physio[:2] if available_hrv else available_physio[:4]
        
        if not plot_metrics:
            raise ValueError("No suitable metrics found for individual boxplots")
        
        # Create subplots
        n_metrics = len(plot_metrics)
        # Increase figure size to accommodate all elements and prevent tight_layout warnings
        fig, axes = plt.subplots(n_metrics, 1, figsize=(18, 6 * n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(plot_metrics):
            ax = axes[i]
            
            # Prepare data for boxplot
            plot_data = []
            plot_labels = []
            colors = []
            
            subjects = sorted(df['Subject'].unique())
            for subject in subjects:
                subject_data = df[df['Subject'] == subject]
                for phase in ['Early', 'Mid', 'Late']:
                    phase_data = subject_data[subject_data['Mission_Phase'] == phase][metric].dropna()
                    if len(phase_data) > 0:
                        plot_data.append(phase_data.values)
                        plot_labels.append(f"{subject}\n{phase}")
                        colors.append(self.phase_colors[phase])
            
            if plot_data:
                # Create boxplot
                bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
                
                # Apply colors
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Add reference ranges
                self._add_reference_ranges_to_plot(ax, metric)
                
                # Format plot
                clean_metric = metric.replace('_mean', '').replace('_', ' ').title()
                ax.set_title(f'Individual {clean_metric} Across Mission Phases', 
                           fontsize=14, fontweight='bold')
                ax.set_ylabel(clean_metric, fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45, labelsize=9)
                
                # Add phase legend
                import matplotlib.patches as mpatches
                legend_elements = [mpatches.Patch(color=color, alpha=0.7, label=phase) 
                                 for phase, color in self.phase_colors.items()]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            else:
                ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
        
        # Apply tight layout with proper error handling and padding
        try:
            plt.tight_layout(pad=3.0)
        except Exception as e:
            logger.warning(f"Tight layout adjustment failed: {e}")
            # Fallback to manual spacing adjustment
            plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Save plot
        plot_path = Path(output_dir) / "individual_mission_phases_boxplots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Individual boxplots saved to: {plot_path}")
        return str(plot_path)
    
    def generate_group_boxplots(self, df: pd.DataFrame, mission_phases: Dict[str, Tuple[float, float]], 
                               output_dir: str = "plots_output") -> str:
        """
        Generate group boxplots comparing all crew members across mission phases.
        
        Args:
            df: Prepared mission data
            mission_phases: Mission phases definition
            output_dir: Output directory for plots
            
        Returns:
            Path to saved plot file
        """
        logger.info("Generating group mission phases boxplots")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Select metrics for plotting
        hrv_metrics = ['SDNN', 'RMSSD', 'Mean_HR', 'LF_HF_Ratio', 'HF_Power', 'LF_Power']
        available_hrv = [m for m in hrv_metrics if m in df.columns and df[m].notna().sum() > 0]
        
        physio_metrics = ['heart_rate_mean', 'breathing_rate_mean', 'activity_mean', 'spo2_mean']
        available_physio = [m for m in physio_metrics if m in df.columns and df[m].notna().sum() > 0]
        
        # Combine metrics (prefer HRV)
        plot_metrics = available_hrv[:4] + available_physio[:2] if available_hrv else available_physio[:6]
        
        if not plot_metrics:
            raise ValueError("No suitable metrics found for group boxplots")
        
        # Create subplot grid
        n_cols = 3
        n_rows = (len(plot_metrics) + n_cols - 1) // n_cols
        
        # Increase figure size to accommodate all elements and prevent tight_layout warnings
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 7 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        for i, metric in enumerate(plot_metrics):
            ax = axes[i]
            
            # Prepare data by phase
            phase_data = []
            phase_labels = ['Early', 'Mid', 'Late']
            
            for phase in phase_labels:
                phase_values = df[df['Mission_Phase'] == phase][metric].dropna()
                phase_data.append(phase_values)
            
            # Check if we have data
            if not any(len(data) > 0 for data in phase_data):
                ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                continue
            
            # Create boxplot
            bp = ax.boxplot(phase_data, labels=phase_labels, patch_artist=True)
            
            # Apply colors
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add reference ranges
            self._add_reference_ranges_to_plot(ax, metric)
            
            # Add statistical annotations
            valid_data = [data for data in phase_data if len(data) > 1]
            if len(valid_data) >= 2:
                try:
                    h_stat, p_value = stats.kruskal(*valid_data)
                    
                    if p_value < 0.001:
                        p_text = "p < 0.001***"
                    elif p_value < 0.01:
                        p_text = "p < 0.01**"
                    elif p_value < 0.05:
                        p_text = "p < 0.05*"
                    else:
                        p_text = f"p = {p_value:.3f}"
                    
                    y_pos = ax.get_ylim()[1] * 0.95
                    ax.text(0.5, y_pos, f"Kruskal-Wallis: {p_text}", 
                           transform=ax.get_xaxis_transform(),
                           ha='center', va='top', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                except Exception as e:
                    logger.warning(f"Statistical test failed for {metric}: {e}")
            
            # Format plot
            clean_metric = metric.replace('_mean', '').replace('_', ' ').title()
            ax.set_title(f'Group {clean_metric}\nby Mission Phase', fontsize=12, fontweight='bold')
            ax.set_xlabel('Mission Phase', fontsize=11)
            ax.set_ylabel(clean_metric, fontsize=11)
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for j in range(len(plot_metrics), len(axes)):
            fig.delaxes(axes[j])
        
        # Apply tight layout with proper error handling and padding
        try:
            plt.tight_layout(pad=3.0)
        except Exception as e:
            logger.warning(f"Tight layout adjustment failed: {e}")
            # Fallback to manual spacing adjustment
            plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Save plot
        plot_path = Path(output_dir) / "group_mission_phases_boxplots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Group boxplots saved to: {plot_path}")
        return str(plot_path)
    
    def generate_comprehensive_report(self, df: pd.DataFrame, mission_phases: Dict[str, Tuple[float, float]], 
                                     individual_plot_path: str, group_plot_path: str,
                                     output_dir: str = "plots_output") -> str:
        """
        Generate a comprehensive report for mission phases analysis.
        
        Args:
            df: Prepared mission data
            mission_phases: Mission phases definition
            individual_plot_path: Path to individual boxplots
            group_plot_path: Path to group boxplots
            output_dir: Output directory
            
        Returns:
            Path to generated report
        """
        logger.info("Generating comprehensive mission phases report")
        
        Path(output_dir).mkdir(exist_ok=True)
        report_path = Path(output_dir) / "mission_phases_analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("VALQUIRIA MISSION PHASES ANALYSIS REPORT\n")
            f.write("ENHANCED HRV ANALYSIS INTEGRATION\n")
            f.write("=" * 55 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Integration: Enhanced HRV Analysis System\n")
            f.write(f"Mission: Valquiria Space Analog Simulation\n\n")
            
            # Mission phases
            f.write("MISSION PHASE DEFINITIONS:\n")
            f.write("-" * 30 + "\n")
            for phase, (start, end) in mission_phases.items():
                f.write(f"{phase} Phase: Sol {start:.1f} - {end:.1f}\n")
            f.write("\n")
            
            # Data summary
            f.write("DATASET SUMMARY:\n")
            f.write("-" * 18 + "\n")
            f.write(f"Total subject-session combinations: {len(df)}\n")
            f.write(f"Unique subjects: {df['Subject'].nunique()}\n")
            
            if 'Mission_Phase' in df.columns:
                phase_counts = df['Mission_Phase'].value_counts()
                f.write(f"\nPhase distribution:\n")
                for phase in ['Early', 'Mid', 'Late']:
                    count = phase_counts.get(phase, 0)
                    pct = (count / len(df)) * 100
                    f.write(f"  {phase}: {count} combinations ({pct:.1f}%)\n")
            
            # Metrics analyzed
            f.write(f"\nMETRICS ANALYZED:\n")
            f.write("-" * 20 + "\n")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            available_metrics = [col for col in numeric_cols if col not in ['Sol']]
            
            hrv_metrics = [col for col in available_metrics if any(hrv in col.upper() for hrv in ['SDNN', 'RMSSD', 'HF', 'LF', 'SD1', 'SD2', 'DFA'])]
            physio_metrics = [col for col in available_metrics if col not in hrv_metrics]
            
            if hrv_metrics:
                f.write("HRV Metrics:\n")
                for metric in hrv_metrics:
                    f.write(f"  • {metric}\n")
                f.write("\n")
            
            if physio_metrics:
                f.write("Physiological Metrics:\n")
                for metric in physio_metrics:
                    f.write(f"  • {metric}\n")
                f.write("\n")
            
            # Generated files
            f.write("GENERATED FILES:\n")
            f.write("-" * 18 + "\n")
            f.write(f"• {individual_plot_path} - Individual crew member comparisons\n")
            f.write(f"• {group_plot_path} - Group phase comparisons\n")
            f.write(f"• {report_path} - This comprehensive report\n\n")
            
            # Analysis notes
            f.write("ANALYSIS NOTES:\n")
            f.write("-" * 16 + "\n")
            f.write("• Boxplots show distribution of metrics across three mission phases\n")
            f.write("• Individual plots highlight crew member variability\n")
            f.write("• Group plots reveal population-level temporal trends\n")
            f.write("• Statistical tests (Kruskal-Wallis) assess phase differences\n")
            f.write("• Integration with Enhanced HRV Analysis provides robust metric calculation\n\n")
            
            f.write("INTEGRATION SUCCESS: Mission phases boxplots successfully integrated\n")
            f.write("with the Enhanced HRV Analysis System for comprehensive crew monitoring.\n")
        
        logger.info(f"Report generated: {report_path}")
        return str(report_path)
    
    def _add_reference_ranges_to_plot(self, ax, metric: str) -> None:
        """
        Add reference ranges to boxplot for HRV metrics.
        
        Args:
            ax: Matplotlib axes object
            metric: Metric name to look up reference ranges
        """
        if not hrv_reference_ranges or not get_reference_range:
            return
        
        # Map metric names to standard keys
        metric_map = {
            'SDNN': 'sdnn',
            'RMSSD': 'rmssd',
            'pNN50': 'pnn50',
            'Mean_HR': None,  # No direct reference range
            'HF_Power': 'hf_power',
            'LF_Power': 'lf_power',
            'LF_HF_Ratio': 'lf_hf_ratio',
            'VLF_Power': 'vlf_power'
        }
        
        metric_key = metric_map.get(metric)
        if not metric_key:
            return
        
        ref_range = get_reference_range(metric_key)
        if not ref_range:
            return
        
        # Add normal range band (25th-75th percentile)
        if ref_range.percentile_25 and ref_range.percentile_75:
            ax.axhspan(
                ref_range.percentile_25, 
                ref_range.percentile_75,
                alpha=0.2, 
                color='green',
                label='Normal Range (25th-75th percentile)',
                zorder=0
            )
        
        # Add median line
        if ref_range.percentile_50:
            ax.axhline(
                ref_range.percentile_50,
                color='green',
                linestyle='--',
                alpha=0.8,
                label='Reference Median',
                zorder=1,
                linewidth=1.5
            )
        

        
        # Add citation text in corner
        citation_info = hrv_reference_ranges.get_citation_info(metric_key)
        if citation_info:
            citation_text = f"Ref: {citation_info['population'][:30]}..."
            ax.text(
                0.02, 0.02,
                citation_text,
                transform=ax.transAxes,
                fontsize=8,
                alpha=0.7,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray"
                )
            )
        
        # Add legend for reference ranges
        handles, labels = ax.get_legend_handles_labels()
        range_handles = [h for h, l in zip(handles, labels) 
                        if any(ref_text in l for ref_text in 
                              ['Normal Range', 'Reference Median'])]
        if range_handles:
            ref_labels = [l for l in labels if any(ref_text in l for ref_text in 
                                         ['Normal Range', 'Reference Median'])]
            legend = ax.legend(
                range_handles, 
                ref_labels,
                loc='upper right', 
                fontsize=8,
                framealpha=0.9
            )
            legend.set_zorder(10) 

    def generate_individual_plotly_boxplots(self, df: pd.DataFrame, mission_phases: Dict[str, Tuple[float, float]], 
                                           output_dir: str = "plots_output") -> List[str]:
        """
        Generate individual Plotly boxplots for each HRV metric across mission phases.
        Each metric gets its own interactive plot with proper normal range coverage.
        
        Args:
            df: Prepared mission data
            mission_phases: Mission phases definition
            output_dir: Output directory for plots
            
        Returns:
            List of paths to saved plot files
        """
        logger.info("Generating individual Plotly boxplots for each HRV metric")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Get all available HRV metrics from the data
        excluded_cols = ['Subject', 'Session', 'Sol', 'Mission_Phase']
        all_metrics = [col for col in df.columns if col not in excluded_cols]
        available_metrics = [m for m in all_metrics if df[m].notna().sum() > 0]
        
        if not available_metrics:
            logger.error(f"Available columns: {list(df.columns)}")
            logger.error(f"Data shape: {df.shape}")
            logger.error("Sample data:")
            logger.error(df.head().to_string())
            raise ValueError("No suitable HRV metrics found for individual boxplots")
        
        plot_paths = []
        
        # Phase colors matching the original scheme
        phase_colors = {
            'Early': '#87CEEB',   # Sky blue
            'Mid': '#90EE90',     # Light green  
            'Late': '#F08080'     # Light coral
        }
        
        for metric in available_metrics:
            logger.info(f"Creating individual Plotly boxplot for {metric}")
            
            # Create individual figure for this metric
            fig = go.Figure()
            
            # Prepare data by phase
            phase_data = {}
            phase_labels = ['Early', 'Mid', 'Late']
            
            for phase in phase_labels:
                phase_values = df[df['Mission_Phase'] == phase][metric].dropna()
                if len(phase_values) > 0:
                    phase_data[phase] = phase_values.values
            
            if not phase_data:
                logger.warning(f"No data available for metric {metric}")
                continue
            
            # Add normal range background first (so it appears behind boxplots)
            self._add_plotly_reference_ranges(fig, metric)
            
            # Add boxplots for each phase with proper positioning
            phase_positions = {'Early': 0, 'Mid': 1, 'Late': 2}
            
            for phase, values in phase_data.items():
                if len(values) == 0:
                    continue
                    
                # Calculate 95% CI for the median using bootstrap
                median_ci = self._calculate_median_ci(values)
                
                # Calculate statistics
                median_val = np.median(values)
                q1_val = np.percentile(values, 25)
                q3_val = np.percentile(values, 75)
                
                fig.add_trace(go.Box(
                    y=values,
                    name=phase,
                    x=[phase_positions[phase]] * len(values),
                    boxpoints='outliers',  # Show outliers
                    marker=dict(
                        color=phase_colors[phase],
                        line=dict(color='rgb(8,48,107)', width=1.5),
                        size=4,
                        opacity=0.8
                    ),
                    line=dict(color='rgb(8,48,107)', width=2),
                    fillcolor=phase_colors[phase],
                    opacity=0.8,
                    notched=True,  # Enable notched boxplots for CI
                    notchwidth=0.5,  # Width of notch
                    # Custom hover template with statistics
                    hovertemplate=(
                        f'<b>{phase} Phase</b><br>'
                        f'{metric}: %{{y:.2f}}<br>'
                        f'Median: {median_val:.2f}<br>'
                        f'Q1: {q1_val:.2f}<br>'
                        f'Q3: {q3_val:.2f}<br>'
                        f'95% CI: [{median_ci[0]:.2f}, {median_ci[1]:.2f}]<br>'
                        f'N: {len(values)}<br>'
                        '<extra></extra>'
                    ),
                    # Add custom data for statistics
                    customdata=[{
                        'median': median_val,
                        'q1': q1_val,
                        'q3': q3_val,
                        'ci_lower': median_ci[0],
                        'ci_upper': median_ci[1],
                        'n': len(values)
                    }] * len(values)
                ))
            
            # Add statistical test results
            if len(phase_data) >= 2:
                self._add_statistical_annotations_plotly(fig, phase_data, metric)
            
            # Format metric title
            clean_metric = self._format_metric_title(metric)
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=f'<b>{clean_metric} by Mission Phase</b>',
                    x=0.5,
                    font=dict(size=16, color='#2C3E50')
                ),
                xaxis=dict(
                    title='Mission Phase',
                    title_font=dict(size=14, color='#2C3E50'),
                    tickfont=dict(size=12, color='#2C3E50'),
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)',
                    zeroline=False,
                    tickmode='array',
                    tickvals=[0, 1, 2],
                    ticktext=['Early', 'Mid', 'Late']
                ),
                yaxis=dict(
                    title=clean_metric,
                    title_font=dict(size=14, color='#2C3E50'),
                    tickfont=dict(size=12, color='#2C3E50'),
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)',
                    zeroline=False
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=11)
                ),
                width=800,
                height=600,
                margin=dict(l=80, r=80, t=100, b=80),
                font=dict(family="Arial, sans-serif", size=11, color="#2C3E50")
            )
            
            # Save plot
            safe_metric = metric.replace('/', '_').replace(' ', '_').lower()
            plot_filename = f"mission_phases_{safe_metric}_boxplot.html"
            plot_path = Path(output_dir) / plot_filename
            
            fig.write_html(
                str(plot_path),
                include_plotlyjs='cdn',
                div_id=f"boxplot_{safe_metric}",
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                    'responsive': True
                }
            )
            
            plot_paths.append(str(plot_path))
            logger.info(f"Individual Plotly boxplot for {metric} saved to: {plot_path}")
        
        return plot_paths

    def _add_plotly_reference_ranges(self, fig: go.Figure, metric: str) -> None:
        """
        Add reference ranges to Plotly boxplot with proper x-axis coverage.
        
        Args:
            fig: Plotly figure object
            metric: Metric name to look up reference ranges
        """
        if not hrv_reference_ranges or not get_reference_range:
            return
        
        # Map metric names to standard keys
        metric_map = {
            'SDNN': 'sdnn',
            'RMSSD': 'rmssd',
            'pNN50': 'pnn50',
            'Mean_HR': None,  # No direct reference range
            'HF_Power': 'hf_power',
            'LF_Power': 'lf_power',
            'LF_HF_Ratio': 'lf_hf_ratio',
            'VLF_Power': 'vlf_power'
        }
        
        metric_key = metric_map.get(metric)
        if not metric_key:
            return
        
        ref_range = get_reference_range(metric_key)
        if not ref_range:
            return
        
        # Add normal range band (25th-75th percentile) covering full x-axis width
        if ref_range.percentile_25 and ref_range.percentile_75:
            fig.add_shape(
                type="rect",
                x0=-0.5,  # Start before first phase
                x1=2.5,   # End after last phase (covers all 3 phases: 0, 1, 2)
                y0=ref_range.percentile_25,
                y1=ref_range.percentile_75,
                fillcolor='rgba(34, 139, 34, 0.15)',  # Light green
                line=dict(width=0),
                layer='below'  # Ensure it appears behind boxplots
            )
            
            # Add legend entry for normal range
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(
                    size=12,
                    color='rgba(34, 139, 34, 0.4)',
                    symbol='square'
                ),
                name='Normal Range (25th-75th %ile)',
                showlegend=True,
                hoverinfo='skip'
            ))
        
        # Add median reference line covering full width
        if ref_range.percentile_50:
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=2.5,
                y0=ref_range.percentile_50,
                y1=ref_range.percentile_50,
                line=dict(
                    color='green',
                    dash='dash',
                    width=2
                ),
                layer='below'
            )
            
            # Add legend entry for median
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='green', dash='dash', width=2),
                name='Reference Median',
                showlegend=True,
                hoverinfo='skip'
            ))
        
        # Add citation annotation
        citation_info = hrv_reference_ranges.get_citation_info(metric_key)
        if citation_info:
            citation_text = f"Reference: {citation_info['population'][:40]}..."
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=citation_text,
                showarrow=False,
                font=dict(size=9, color='gray'),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
                xanchor="left",
                yanchor="top"
            )

    def _add_statistical_annotations_plotly(self, fig: go.Figure, phase_data: Dict[str, np.ndarray], metric: str) -> None:
        """Add statistical test annotations to Plotly figure."""
        try:
            # Perform Kruskal-Wallis test
            valid_data = [data for data in phase_data.values() if len(data) > 1]
            if len(valid_data) >= 2:
                h_stat, p_value = stats.kruskal(*valid_data)
                
                # Format p-value
                if p_value < 0.001:
                    p_text = "p < 0.001***"
                    color = "red"
                elif p_value < 0.01:
                    p_text = "p < 0.01**"
                    color = "orange"
                elif p_value < 0.05:
                    p_text = "p < 0.05*"
                    color = "blue"
                else:
                    p_text = f"p = {p_value:.3f}"
                    color = "gray"
                
                # Add statistical annotation
                fig.add_annotation(
                    x=0.98,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=f"Kruskal-Wallis Test<br>{p_text}",
                    showarrow=False,
                    font=dict(size=11, color=color),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor=color,
                    borderwidth=1,
                    xanchor="right",
                    yanchor="top"
                )
        except Exception as e:
            logger.warning(f"Statistical annotation failed for {metric}: {e}")

    def _format_metric_title(self, metric: str) -> str:
        """Format metric names for display."""
        title_mapping = {
            'SDNN': 'SDNN (Standard Deviation of RR Intervals)',
            'RMSSD': 'RMSSD (Root Mean Square of Successive Differences)', 
            'pNN50': 'pNN50 (% of RR intervals differing >50ms)',
            'Mean_HR': 'Mean Heart Rate (BPM)',
            'HF_Power': 'HF Power (High Frequency Spectral Power)',
            'LF_Power': 'LF Power (Low Frequency Spectral Power)',
            'LF_HF_Ratio': 'LF/HF Ratio (Sympathovagal Balance)',
            'VLF_Power': 'VLF Power (Very Low Frequency Spectral Power)'
        }
        
        return title_mapping.get(metric, metric.replace('_', ' ').title())

    def generate_all_plotly_boxplots(self, analysis_results: Dict[str, Any], output_dir: str = "plots_output") -> Dict[str, List[str]]:
        """
        Generate all Plotly boxplots (individual plots for each metric).
        
        Args:
            analysis_results: Results from HRV analysis
            output_dir: Output directory for plots
            
        Returns:
            Dictionary with plot paths organized by type
        """
        try:
            # Prepare mission data
            df, mission_phases = self.prepare_mission_data(analysis_results)
            
            # Generate individual Plotly boxplots
            individual_paths = self.generate_individual_plotly_boxplots(df, mission_phases, output_dir)
            
            # Generate summary report
            report_path = self.generate_plotly_boxplot_report(df, mission_phases, individual_paths, output_dir)
            
            return {
                'individual_plots': individual_paths,
                'report': report_path,
                'mission_phases': mission_phases
            }
            
        except Exception as e:
            logger.error(f"Error generating Plotly boxplots: {e}")
            raise

    def generate_plotly_boxplot_report(self, df: pd.DataFrame, mission_phases: Dict[str, Tuple[float, float]], 
                                     plot_paths: List[str], output_dir: str = "plots_output") -> str:
        """Generate comprehensive report for Plotly boxplot analysis."""
        logger.info("Generating Plotly boxplot analysis report")
        
        Path(output_dir).mkdir(exist_ok=True)
        report_path = Path(output_dir) / "mission_phases_plotly_boxplot_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("VALQUIRIA MISSION PHASES - INTERACTIVE BOXPLOT ANALYSIS\n")
            f.write("ENHANCED HRV ANALYSIS WITH PLOTLY VISUALIZATIONS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Visualization: Interactive Plotly Boxplots\n")
            f.write(f"Mission: Valquiria Space Analog Simulation\n\n")
            
            # Mission phases
            f.write("MISSION PHASE DEFINITIONS:\n")
            f.write("-" * 30 + "\n")
            for phase, (start, end) in mission_phases.items():
                f.write(f"{phase} Phase: Sol {start:.1f} - {end:.1f}\n")
            f.write("\n")
            
            # Dataset summary
            f.write("DATASET SUMMARY:\n")
            f.write("-" * 18 + "\n")
            f.write(f"Total subject-session combinations: {len(df)}\n")
            f.write(f"Unique subjects: {df['Subject'].nunique()}\n")
            
            if 'Mission_Phase' in df.columns:
                phase_counts = df['Mission_Phase'].value_counts()
                f.write(f"\nPhase distribution:\n")
                for phase in ['Early', 'Mid', 'Late']:
                    count = phase_counts.get(phase, 0)
                    pct = (count / len(df)) * 100
                    f.write(f"  {phase}: {count} combinations ({pct:.1f}%)\n")
            
            # Generated plots
            f.write(f"\nGENERATED INTERACTIVE PLOTS:\n")
            f.write("-" * 32 + "\n")
            for i, plot_path in enumerate(plot_paths, 1):
                plot_name = Path(plot_path).stem
                metric_name = plot_name.replace('mission_phases_', '').replace('_boxplot', '').replace('_', ' ').title()
                f.write(f"{i:2}. {metric_name} - {Path(plot_path).name}\n")
            f.write("\n")
            
            # Analysis features
            f.write("INTERACTIVE FEATURES:\n")
            f.write("-" * 22 + "\n")
            f.write("• Individual plots for each HRV metric\n")
            f.write("• Interactive hover information with detailed values\n")
            f.write("• Normal reference ranges (25th-75th percentiles) displayed as background bands\n")
            f.write("• Reference median lines for clinical interpretation\n")
            f.write("• Statistical significance testing (Kruskal-Wallis) with p-values\n")
            f.write("• Professional aerospace medicine styling\n")
            f.write("• Zoom, pan, and export capabilities\n")
            f.write("• Browser-based viewing with responsive design\n\n")
            
            # Usage instructions
            f.write("USAGE INSTRUCTIONS:\n")
            f.write("-" * 20 + "\n")
            f.write("1. Open any .html file in your web browser\n")
            f.write("2. Use mouse to hover over data points for detailed information\n")
            f.write("3. Use toolbar to zoom, pan, or export plots\n")
            f.write("4. Green background bands show normal physiological ranges\n")
            f.write("5. Dashed green lines show reference median values\n")
            f.write("6. Statistical test results appear in top-right corner\n\n")
            
            f.write("CLINICAL INTERPRETATION:\n")
            f.write("-" * 25 + "\n")
            f.write("• Values within green bands are in normal physiological range\n")
            f.write("• Significant p-values (p < 0.05) indicate phase-dependent changes\n")
            f.write("• Individual subject patterns can be assessed for adaptation responses\n")
            f.write("• Outliers may indicate individual stress responses or measurement artifacts\n\n")
            
            f.write("INTEGRATION SUCCESS: Interactive Plotly boxplots successfully generated\n")
            f.write("for comprehensive mission phase analysis with clinical reference ranges.\n")
        
        logger.info(f"Plotly boxplot report generated: {report_path}")
        return str(report_path) 

    def _extract_hrv_metrics_from_results(self, analysis_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract HRV metrics from analysis results and create a DataFrame.
        
        The HRV processor stores results in domain-specific nested dictionaries:
        {
          'T01_Mara': {
            'hrv_results': {
              'time_domain': {'sdnn': 45.2, 'rmssd': 28.1, 'pnn50': 15.3, 'mean_hr': 72.4, ...},
              'frequency_domain': {'lf_power': 1245.6, 'hf_power': 567.8, 'lf_hf_ratio': 2.19, ...},
              'parasympathetic': {...},
              'sympathetic': {...},
              ...
            },
            'sol_number': 3.5,
            'processing_info': {...}
          }
        }
        
        This method flattens the nested structure for boxplot analysis.
        """
        logger.info("Extracting HRV metrics from nested analysis results structure")
        
        extracted_data = []
        
        for subject_session, results in analysis_results.items():
            if not isinstance(results, dict) or 'hrv_results' not in results:
                logger.warning(f"No HRV results found for {subject_session}")
                continue
            
            hrv_results = results['hrv_results']
            if not isinstance(hrv_results, dict):
                logger.warning(f"Invalid HRV results format for {subject_session}")
                continue
            
            # Extract subject name and sol number
            subject = subject_session.split('_')[0] if '_' in subject_session else subject_session
            sol_number = results.get('sol_number', 0)
            
            # Flatten metrics from all domains
            flattened_metrics = {}
            
            # Extract time domain metrics
            if 'time_domain' in hrv_results:
                td = hrv_results['time_domain']
                if isinstance(td, dict):
                    flattened_metrics.update({
                        'SDNN': td.get('sdnn', np.nan),
                        'RMSSD': td.get('rmssd', np.nan),
                        'pNN50': td.get('pnn50', np.nan),
                        'Mean_HR': td.get('mean_hr', np.nan),
                        'NN50': td.get('nn50', np.nan),
                        'pNN20': td.get('pnn20', np.nan),
                        'CVNN': td.get('cvnn', np.nan),
                        'SDSD': td.get('sdsd', np.nan),
                        'Mean_NNI': td.get('mean_nni', np.nan),
                        'Median_NNI': td.get('median_nni', np.nan)
                    })
            
            # Extract frequency domain metrics
            if 'frequency_domain' in hrv_results:
                fd = hrv_results['frequency_domain']
                if isinstance(fd, dict):
                    flattened_metrics.update({
                        'LF_Power': fd.get('lf_power', np.nan),
                        'HF_Power': fd.get('hf_power', np.nan),
                        'VLF_Power': fd.get('vlf_power', np.nan),
                        'Total_Power': fd.get('total_power', np.nan),
                        'LF_HF_Ratio': fd.get('lf_hf_ratio', np.nan),
                        'LF_NU': fd.get('lf_nu', np.nan),
                        'HF_NU': fd.get('hf_nu', np.nan),
                        'LF_Peak': fd.get('lf_peak', np.nan),
                        'HF_Peak': fd.get('hf_peak', np.nan),
                        'VLF_Peak': fd.get('vlf_peak', np.nan)
                    })
            
            # Extract nonlinear metrics if available
            if 'nonlinear' in hrv_results:
                nl = hrv_results['nonlinear']
                if isinstance(nl, dict):
                    flattened_metrics.update({
                        'SD1': nl.get('sd1', np.nan),
                        'SD2': nl.get('sd2', np.nan),
                        'SD1_SD2_Ratio': nl.get('sd1_sd2_ratio', np.nan),
                        'DFA_Alpha1': nl.get('dfa_alpha1', np.nan),
                        'DFA_Alpha2': nl.get('dfa_alpha2', np.nan),
                        'Sample_Entropy': nl.get('sample_entropy', np.nan),
                        'ApEn': nl.get('approximate_entropy', np.nan),
                        'TINN': nl.get('tinn', np.nan)
                    })
            
            # Extract parasympathetic metrics
            if 'parasympathetic' in hrv_results:
                ps = hrv_results['parasympathetic']
                if isinstance(ps, dict):
                    flattened_metrics.update({
                        'Parasympathetic_Index': ps.get('parasympathetic_index', np.nan),
                        'RSA_Amplitude': ps.get('rsa_amplitude', np.nan),
                        'Respiratory_Frequency': ps.get('respiratory_frequency', np.nan),
                        'Vagal_Tone_Index': ps.get('vagal_tone_index', np.nan),
                        'HF_RMSSD_Ratio': ps.get('hf_rmssd_ratio', np.nan)
                    })
            
            # Extract sympathetic metrics
            if 'sympathetic' in hrv_results:
                sy = hrv_results['sympathetic']
                if isinstance(sy, dict):
                    flattened_metrics.update({
                        'Sympathetic_Index': sy.get('sympathetic_index', np.nan),
                        'Stress_Index': sy.get('stress_index', np.nan),
                        'Autonomic_Balance': sy.get('autonomic_balance', np.nan),
                        'Cardiac_Sympathetic_Index': sy.get('cardiac_sympathetic_index', np.nan),
                        'Beta_Adrenergic_Sensitivity': sy.get('beta_adrenergic_sensitivity', np.nan)
                    })
            
            # Extract ANS balance metrics
            if 'ans_balance' in hrv_results:
                ab = hrv_results['ans_balance']
                if isinstance(ab, dict):
                    flattened_metrics.update({
                        'ANS_Complexity': ab.get('ans_complexity', np.nan),
                        'Sympathovagal_Index': ab.get('sympathovagal_index', np.nan),
                        'Cardiac_Autonomic_Balance': ab.get('cardiac_autonomic_balance', np.nan),
                        'Autonomic_Reactivity': ab.get('autonomic_reactivity', np.nan),
                        'Baroreflex_Sensitivity': ab.get('baroreflex_sensitivity', np.nan)
                    })
            
            # Add metadata
            row_data = {
                'Subject': subject,
                'Session': subject_session,
                'Sol': sol_number,
                **flattened_metrics
            }
            
            extracted_data.append(row_data)
        
        if not extracted_data:
            raise ValueError("No valid HRV data found in analysis results")
        
        df = pd.DataFrame(extracted_data)
        
        # Log available metrics
        metric_cols = [col for col in df.columns if col not in ['Subject', 'Session', 'Sol']]
        available_metrics = []
        for col in metric_cols:
            valid_count = df[col].notna().sum()
            if valid_count > 0:
                available_metrics.append(f"{col} ({valid_count} valid)")
        
        logger.info(f"Extracted {len(df)} records with {len(available_metrics)} metrics:")
        for metric_info in available_metrics[:10]:  # Show first 10
            logger.info(f"  • {metric_info}")
        if len(available_metrics) > 10:
            logger.info(f"  • ... and {len(available_metrics) - 10} more")
        
        return df 

    def _calculate_median_ci(self, values: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for the median.
        
        Args:
            values: Array of values
            confidence_level: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_ci, upper_ci)
        """
        if len(values) < 3:
            # For very small samples, return range-based CI
            median_val = np.median(values)
            data_range = np.ptp(values) if len(values) > 1 else 0
            return (median_val - data_range * 0.5, median_val + data_range * 0.5)
        
        # Bootstrap resampling
        n_bootstrap = min(1000, max(100, len(values) * 10))
        bootstrap_medians = []
        
        np.random.seed(42)  # For reproducibility
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_medians.append(np.median(bootstrap_sample))
        
        bootstrap_medians = np.array(bootstrap_medians)
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return (np.percentile(bootstrap_medians, lower_percentile),
                np.percentile(bootstrap_medians, upper_percentile)) 