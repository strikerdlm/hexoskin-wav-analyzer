#!/usr/bin/env python3
"""
Comprehensive Parasympathetic Time Analysis for Space Crew
===========================================================

This script provides a detailed analysis of parasympathetic nervous system activity
across all crew members throughout the mission timeline. Based on the HRV review,
parasympathetic tone is primarily assessed through:

1. Time-domain metrics: RMSSD, pNN50, pNN20
2. Frequency-domain metrics: HF power (0.15-0.4 Hz), HFnu (normalized units)
3. Nonlinear metrics: SD1 from Poincaré plot analysis

The analysis includes:
- Longitudinal trends for each crew member
- Statistical comparisons between crew members
- Mission phase analysis
- Autonomic balance assessment
- Comprehensive visualizations

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, pearsonr, spearmanr
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
import warnings
warnings.filterwarnings('ignore')

# Set style for high-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11

class ParasympatheticAnalyzer:
    """
    Comprehensive analyzer for parasympathetic nervous system activity
    in space crew members using validated HRV metrics.
    """
    
    def __init__(self, data_path='working_folder/hrv_results/hrv_complete.csv'):
        """
        Initialize the analyzer with HRV data.
        
        Parameters:
        -----------
        data_path : str
            Path to the complete HRV results CSV file
        """
        self.data_path = data_path
        self.data = None
        self.parasympathetic_metrics = ['rmssd', 'pnni_50', 'pnni_20', 'hf', 'hfnu', 'sd1']
        self.crew_names = {
            'T01_Mara': 'Mara',
            'T02_Laura': 'Laura', 
            'T03_Nancy': 'Nancy',
            'T04_Michelle': 'Michelle',
            'T05_Felicitas': 'Felicitas',
            'T06_Mara_Selena': 'Mara Selena',
            'T07_Geraldinn': 'Geraldinn',
            'T08_Karina': 'Karina'
        }
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load and prepare HRV data for analysis."""
        try:
            self.data = pd.read_csv(self.data_path)
            
            # Clean column names and ensure proper data types
            self.data['Subject'] = self.data['Subject'].astype(str)
            self.data['Sol'] = self.data['Sol'].astype(int)
            
            # Add friendly names
            self.data['Crew_Name'] = self.data['Subject'].map(self.crew_names)
            
            # Sort by subject and sol for proper time series analysis
            self.data = self.data.sort_values(['Subject', 'Sol'])
            
            # Create mission phase categories
            self.data['Mission_Phase'] = self.data['Sol'].apply(self._categorize_mission_phase)
            
            print(f"✓ Data loaded successfully: {len(self.data)} recordings")
            print(f"✓ Crew members: {len(self.data['Subject'].unique())}")
            print(f"✓ Sol range: {self.data['Sol'].min()} - {self.data['Sol'].max()}")
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def _categorize_mission_phase(self, sol):
        """Categorize mission phases based on Sol number."""
        if sol <= 5:
            return 'Early Mission'
        elif sol <= 10:
            return 'Mid Mission'
        else:
            return 'Late Mission'
    
    def calculate_parasympathetic_summary(self):
        """
        Calculate summary statistics for parasympathetic metrics.
        
        Returns:
        --------
        pd.DataFrame
            Summary statistics for each crew member and metric
        """
        summary_stats = []
        
        for crew in self.data['Subject'].unique():
            crew_data = self.data[self.data['Subject'] == crew]
            
            for metric in self.parasympathetic_metrics:
                if metric in crew_data.columns:
                    values = crew_data[metric].dropna()
                    if len(values) > 0:
                        summary_stats.append({
                            'Crew': self.crew_names[crew],
                            'Subject_ID': crew,
                            'Metric': metric,
                            'Mean': values.mean(),
                            'Median': values.median(),
                            'Std': values.std(),
                            'Min': values.min(),
                            'Max': values.max(),
                            'CV': values.std() / values.mean() * 100 if values.mean() != 0 else 0,
                            'N_recordings': len(values)
                        })
        
        return pd.DataFrame(summary_stats)
    
    def perform_statistical_analysis(self):
        """
        Perform comprehensive statistical analysis of parasympathetic metrics.
        
        Returns:
        --------
        dict
            Dictionary containing various statistical test results
        """
        results = {}
        
        # One-way ANOVA for each metric across crew members
        print("Performing One-Way ANOVA Tests")
        print("=" * 40)
        
        for metric in self.parasympathetic_metrics:
            if metric in self.data.columns:
                groups = [self.data[self.data['Subject'] == crew][metric].dropna().values 
                         for crew in self.data['Subject'].unique()]
                
                # Remove empty groups
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) >= 2:
                    f_stat, p_value = f_oneway(*groups)
                    results[f'anova_{metric}'] = {'F': f_stat, 'p': p_value}
                    
                    print(f"{metric.upper():8} - F={f_stat:.3f}, p={p_value:.4f} " +
                          f"{'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        
        # Correlations with mission time (Sol)
        print("\nCorrelations with Mission Time (Sol)")
        print("=" * 40)
        
        for metric in self.parasympathetic_metrics:
            if metric in self.data.columns:
                clean_data = self.data[['Sol', metric]].dropna()
                if len(clean_data) > 3:
                    r_pearson, p_pearson = pearsonr(clean_data['Sol'], clean_data[metric])
                    r_spearman, p_spearman = spearmanr(clean_data['Sol'], clean_data[metric])
                    
                    results[f'corr_{metric}'] = {
                        'pearson_r': r_pearson, 'pearson_p': p_pearson,
                        'spearman_r': r_spearman, 'spearman_p': p_spearman
                    }
                    
                    print(f"{metric.upper():8} - Pearson r={r_pearson:.3f} (p={p_pearson:.4f}), " +
                          f"Spearman r={r_spearman:.3f} (p={p_spearman:.4f})")
        
        # Post-hoc tests (Tukey HSD) for significant ANOVA results
        print("\nPost-hoc Analysis (Tukey HSD)")
        print("=" * 40)
        
        for metric in self.parasympathetic_metrics:
            if metric in self.data.columns and f'anova_{metric}' in results:
                if results[f'anova_{metric}']['p'] < 0.05:
                    clean_data = self.data[['Subject', metric]].dropna()
                    if len(clean_data) > 0:
                        tukey_result = pairwise_tukeyhsd(
                            clean_data[metric], 
                            clean_data['Subject'], 
                            alpha=0.05
                        )
                        results[f'tukey_{metric}'] = tukey_result
                        print(f"\n{metric.upper()} - Significant differences found:")
                        print(tukey_result.summary())
        
        return results
    
    def plot_longitudinal_trends(self, save_path='working_folder/parasympathetic_longitudinal.png'):
        """
        Create comprehensive longitudinal trend plots for parasympathetic metrics.
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        colors = sns.color_palette("husl", len(self.data['Subject'].unique()))
        
        for i, metric in enumerate(self.parasympathetic_metrics):
            if metric in self.data.columns:
                ax = axes[i]
                
                # Plot individual trajectories
                for j, crew in enumerate(self.data['Subject'].unique()):
                    crew_data = self.data[self.data['Subject'] == crew].sort_values('Sol')
                    
                    # Plot line
                    ax.plot(crew_data['Sol'], crew_data[metric], 
                           color=colors[j], alpha=0.7, linewidth=2,
                           label=self.crew_names[crew], marker='o', markersize=4)
                    
                    # Add trend line
                    if len(crew_data) > 2:
                        z = np.polyfit(crew_data['Sol'], crew_data[metric], 1)
                        p = np.poly1d(z)
                        ax.plot(crew_data['Sol'], p(crew_data['Sol']), 
                               color=colors[j], linestyle='--', alpha=0.5, linewidth=1)
                
                ax.set_title(f'{metric.upper()} Over Mission Time', fontsize=14, fontweight='bold')
                ax.set_xlabel('Sol (Mission Day)', fontsize=12)
                ax.set_ylabel(self._get_metric_label(metric), fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Longitudinal trends plot saved to: {save_path}")
    
    def plot_crew_comparisons(self, save_path='working_folder/parasympathetic_comparisons.png'):
        """
        Create box plots and violin plots for crew comparisons.
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.parasympathetic_metrics):
            if metric in self.data.columns:
                ax = axes[i]
                
                # Create violin plot with box plot overlay
                sns.violinplot(data=self.data, x='Crew_Name', y=metric, ax=ax, alpha=0.7)
                sns.boxplot(data=self.data, x='Crew_Name', y=metric, ax=ax, 
                           boxprops=dict(alpha=0.8), width=0.3)
                
                ax.set_title(f'{metric.upper()} by Crew Member', fontsize=14, fontweight='bold')
                ax.set_xlabel('Crew Member', fontsize=12)
                ax.set_ylabel(self._get_metric_label(metric), fontsize=12)
                ax.tick_params(axis='x', rotation=45)
                
                # Add statistical annotations
                self._add_statistical_annotations(ax, metric)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Crew comparisons plot saved to: {save_path}")
    
    def plot_mission_phase_analysis(self, save_path='working_folder/parasympathetic_phases.png'):
        """
        Analyze parasympathetic activity across mission phases.
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.parasympathetic_metrics):
            if metric in self.data.columns:
                ax = axes[i]
                
                # Create box plot by mission phase
                sns.boxplot(data=self.data, x='Mission_Phase', y=metric, ax=ax)
                
                # Add individual points
                sns.stripplot(data=self.data, x='Mission_Phase', y=metric, 
                             ax=ax, color='black', alpha=0.5, size=3)
                
                ax.set_title(f'{metric.upper()} by Mission Phase', fontsize=14, fontweight='bold')
                ax.set_xlabel('Mission Phase', fontsize=12)
                ax.set_ylabel(self._get_metric_label(metric), fontsize=12)
                
                # Calculate mean values for each phase
                phase_means = self.data.groupby('Mission_Phase')[metric].mean()
                for j, phase in enumerate(phase_means.index):
                    ax.text(j, ax.get_ylim()[1] * 0.9, f'μ={phase_means[phase]:.2f}', 
                           ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Mission phase analysis plot saved to: {save_path}")
    
    def plot_autonomic_balance(self, save_path='working_folder/autonomic_balance.png'):
        """
        Create autonomic balance analysis using LF/HF ratio and parasympathetic metrics.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: LF/HF ratio vs RMSSD
        ax1 = axes[0, 0]
        for crew in self.data['Subject'].unique():
            crew_data = self.data[self.data['Subject'] == crew]
            ax1.scatter(crew_data['lf_hf_ratio'], crew_data['rmssd'], 
                       label=self.crew_names[crew], alpha=0.7, s=50)
        
        ax1.set_xlabel('LF/HF Ratio (Sympathetic/Parasympathetic Balance)', fontsize=12)
        ax1.set_ylabel('RMSSD (ms)', fontsize=12)
        ax1.set_title('Autonomic Balance: LF/HF vs RMSSD', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: HF power vs SD1
        ax2 = axes[0, 1]
        for crew in self.data['Subject'].unique():
            crew_data = self.data[self.data['Subject'] == crew]
            ax2.scatter(crew_data['hf'], crew_data['sd1'], 
                       label=self.crew_names[crew], alpha=0.7, s=50)
        
        ax2.set_xlabel('HF Power (ms²)', fontsize=12)
        ax2.set_ylabel('SD1 (ms)', fontsize=12)
        ax2.set_title('Parasympathetic Concordance: HF Power vs SD1', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Parasympathetic composite score over time
        ax3 = axes[1, 0]
        
        # Create composite parasympathetic score (z-score normalized)
        parasympa_metrics = ['rmssd', 'pnni_50', 'hf', 'sd1']
        composite_data = self.data.copy()
        
        for metric in parasympa_metrics:
            if metric in composite_data.columns:
                composite_data[f'{metric}_z'] = stats.zscore(composite_data[metric], nan_policy='omit')
        
        # Calculate composite score
        z_columns = [f'{metric}_z' for metric in parasympa_metrics if f'{metric}_z' in composite_data.columns]
        composite_data['parasympathetic_score'] = composite_data[z_columns].mean(axis=1)
        
        for crew in self.data['Subject'].unique():
            crew_data = composite_data[composite_data['Subject'] == crew].sort_values('Sol')
            ax3.plot(crew_data['Sol'], crew_data['parasympathetic_score'], 
                    label=self.crew_names[crew], marker='o', linewidth=2)
        
        ax3.set_xlabel('Sol (Mission Day)', fontsize=12)
        ax3.set_ylabel('Composite Parasympathetic Score (Z-score)', fontsize=12)
        ax3.set_title('Parasympathetic Activity Over Mission Time', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 4: Correlation heatmap
        ax4 = axes[1, 1]
        
        # Select parasympathetic metrics for correlation
        para_cols = [col for col in parasympa_metrics if col in self.data.columns]
        corr_data = self.data[para_cols + ['Sol']].corr()
        
        sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax4, cbar_kws={'label': 'Correlation Coefficient'})
        ax4.set_title('Parasympathetic Metrics Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Autonomic balance analysis plot saved to: {save_path}")
    
    def _get_metric_label(self, metric):
        """Get proper label for metric with units."""
        labels = {
            'rmssd': 'RMSSD (ms)',
            'pnni_50': 'pNN50 (%)',
            'pnni_20': 'pNN20 (%)',
            'hf': 'HF Power (ms²)',
            'hfnu': 'HF Power (nu)',
            'sd1': 'SD1 (ms)'
        }
        return labels.get(metric, metric)
    
    def _add_statistical_annotations(self, ax, metric):
        """Add statistical significance annotations to plots."""
        # This is a simplified version - in practice, you'd want to perform
        # pairwise tests and add significance bars
        pass
    
    def generate_scientific_report(self, save_path='working_folder/parasympathetic_report.txt'):
        """
        Generate a comprehensive scientific report of the parasympathetic analysis.
        """
        report = []
        
        # Header
        report.append("PARASYMPATHETIC NERVOUS SYSTEM ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Introduction
        report.append("INTRODUCTION")
        report.append("-" * 20)
        report.append("This analysis examines parasympathetic nervous system activity in space crew")
        report.append("members using validated heart rate variability (HRV) metrics. Parasympathetic")
        report.append("activity reflects the body's 'rest and digest' response and is crucial for")
        report.append("cardiovascular health, stress recovery, and overall autonomic balance.")
        report.append("")
        
        # Methodology
        report.append("METHODOLOGY")
        report.append("-" * 20)
        report.append("Parasympathetic activity was assessed using established HRV metrics:")
        report.append("• RMSSD: Root mean square of successive RR interval differences")
        report.append("• pNN50: Percentage of successive RR intervals differing by >50ms")
        report.append("• pNN20: Percentage of successive RR intervals differing by >20ms")
        report.append("• HF Power: High frequency power (0.15-0.4 Hz) from spectral analysis")
        report.append("• HFnu: HF power in normalized units")
        report.append("• SD1: Short-term variability from Poincaré plot analysis")
        report.append("")
        
        # Results
        report.append("RESULTS")
        report.append("-" * 20)
        
        # Summary statistics
        summary_stats = self.calculate_parasympathetic_summary()
        report.append("Summary Statistics by Crew Member:")
        report.append("")
        
        for crew in summary_stats['Crew'].unique():
            crew_stats = summary_stats[summary_stats['Crew'] == crew]
            report.append(f"{crew}:")
            for _, row in crew_stats.iterrows():
                report.append(f"  {row['Metric'].upper()}: {row['Mean']:.2f} ± {row['Std']:.2f} "
                             f"(range: {row['Min']:.2f}-{row['Max']:.2f})")
            report.append("")
        
        # Statistical analysis
        stats_results = self.perform_statistical_analysis()
        
        report.append("Statistical Analysis:")
        report.append("")
        
        # ANOVA results
        report.append("Between-crew differences (One-way ANOVA):")
        for metric in self.parasympathetic_metrics:
            if f'anova_{metric}' in stats_results:
                result = stats_results[f'anova_{metric}']
                sig_level = "***" if result['p'] < 0.001 else "**" if result['p'] < 0.01 else "*" if result['p'] < 0.05 else "ns"
                report.append(f"  {metric.upper()}: F={result['F']:.3f}, p={result['p']:.4f} {sig_level}")
        
        report.append("")
        
        # Correlations with time
        report.append("Correlations with mission time:")
        for metric in self.parasympathetic_metrics:
            if f'corr_{metric}' in stats_results:
                result = stats_results[f'corr_{metric}']
                report.append(f"  {metric.upper()}: r={result['pearson_r']:.3f}, p={result['pearson_p']:.4f}")
        
        # Clinical interpretation
        report.append("")
        report.append("CLINICAL INTERPRETATION")
        report.append("-" * 30)
        report.append("Higher values in parasympathetic metrics (RMSSD, pNN50, HF power, SD1)")
        report.append("indicate greater parasympathetic activity and better autonomic balance.")
        report.append("Lower values may suggest sympathetic dominance, stress, or fatigue.")
        report.append("")
        
        # Conclusions
        report.append("CONCLUSIONS")
        report.append("-" * 20)
        report.append("This analysis provides comprehensive insights into parasympathetic nervous")
        report.append("system activity across crew members throughout the mission timeline.")
        report.append("Individual variations and temporal patterns can inform personalized")
        report.append("countermeasures and health monitoring strategies.")
        report.append("")
        
        # Save report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✓ Scientific report saved to: {save_path}")
        
        return '\n'.join(report)
    
    def run_complete_analysis(self):
        """
        Run the complete parasympathetic analysis pipeline.
        """
        print("COMPREHENSIVE PARASYMPATHETIC ANALYSIS")
        print("=" * 50)
        print("Analyzing parasympathetic nervous system activity across crew members")
        print("Based on validated HRV metrics and scientific literature")
        print()
        
        # Generate all plots
        print("1. Creating longitudinal trend analysis...")
        self.plot_longitudinal_trends()
        
        print("2. Creating crew comparison analysis...")
        self.plot_crew_comparisons()
        
        print("3. Creating mission phase analysis...")
        self.plot_mission_phase_analysis()
        
        print("4. Creating autonomic balance analysis...")
        self.plot_autonomic_balance()
        
        # Generate report
        print("5. Generating scientific report...")
        report = self.generate_scientific_report()
        
        # Statistical analysis
        print("6. Performing statistical analysis...")
        stats_results = self.perform_statistical_analysis()
        
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE!")
        print("=" * 50)
        print("Generated files:")
        print("• parasympathetic_longitudinal.png - Longitudinal trends")
        print("• parasympathetic_comparisons.png - Crew comparisons")
        print("• parasympathetic_phases.png - Mission phase analysis")
        print("• autonomic_balance.png - Autonomic balance analysis")
        print("• parasympathetic_report.txt - Scientific report")
        
        return stats_results


# Main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ParasympatheticAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    print("\nParasympathetic analysis completed successfully!")
    print("All plots and reports have been saved to the working_folder directory.") 