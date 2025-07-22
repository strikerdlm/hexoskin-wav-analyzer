#!/usr/bin/env python3
"""
Mission Phase Analysis - Boxplots for Three Moments During the Mission
=====================================================================

This script analyzes physiological and HRV data from the Valquiria space analog simulation,
dividing the mission into three phases (Early, Mid, Late) and creating comprehensive
boxplots for both individual subjects and the whole group.

Author: AI Assistant
Date: 2025-01-14
Mission: Valquiria Space Analog Simulation
Subjects: T01-T08 (8 crew members)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlite3
import warnings
from typing import Dict, List, Tuple, Optional
from scipy import stats
import os
import sys

# Configure matplotlib and seaborn for publication-quality plots
plt.rcParams.update({
    'figure.figsize': (12, 8),
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class MissionPhaseAnalyzer:
    """Comprehensive analyzer for mission phase comparisons with boxplots."""
    
    def __init__(self, base_path: str = "Data"):
        """Initialize the analyzer with base data path."""
        self.base_path = Path(base_path)
        self.data = None
        self.subjects = ["T01_Mara", "T02_Laura", "T03_Nancy", "T04_Michelle", 
                        "T05_Felicitas", "T06_Mara_Selena", "T07_Geraldinn", "T08_Karina"]
        self.mission_phases = None
        self.hrv_metrics = {}
        self.physiological_metrics = {}
        
        # Define key physiological variables to analyze
        self.key_variables = [
            'heart_rate [bpm]',
            'breathing_rate [rpm]', 
            'activity [g]',
            'SPO2 [%]',
            'minute_ventilation [mL/min]',
            'systolic_pressure [mmHg]',
            'temperature_celcius [C]'
        ]
        
        # HRV metrics that will be calculated
        self.hrv_variables = [
            'SDNN', 'RMSSD', 'pNN50', 'Mean_HR', 
            'HF_Power', 'LF_Power', 'VLF_Power', 'LF_HF_Ratio',
            'SD1', 'SD2', 'DFA_alpha1', 'DFA_alpha2'
        ]
        
    def load_mission_data(self) -> pd.DataFrame:
        """Load mission data from database or CSV files."""
        print("🚀 Loading Valquiria Mission Data...")
        print("=" * 50)
        
        # Try to load from merged database first
        db_path = self.base_path / "merged_data.db"
        if db_path.exists():
            print(f"📁 Found database: {db_path}")
            return self._load_from_database(db_path)
        else:
            print("📁 Database not found, loading from CSV files...")
            return self._load_from_csv_files()
    
    def _load_from_database(self, db_path: Path) -> pd.DataFrame:
        """Load data from SQLite database."""
        try:
            conn = sqlite3.connect(str(db_path))
            
            # Get table names
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
            if len(tables) == 0:
                raise ValueError("No tables found in database")
                
            table_name = tables.iloc[0]['name']
            print(f"📊 Loading from table: {table_name}")
            
            # Load data with subject and SOL information
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, conn)
            conn.close()
            
            print(f"✅ Loaded {len(df):,} records from database")
            return self._process_loaded_data(df)
            
        except Exception as e:
            print(f"❌ Error loading from database: {e}")
            print("📁 Falling back to CSV files...")
            return self._load_from_csv_files()
    
    def _load_from_csv_files(self) -> pd.DataFrame:
        """Load data from individual CSV files."""
        enhanced_data_path = Path("src/hrv_analysis/enhanced_hrv_analysis/Data")
        
        all_data = []
        successful_loads = 0
        
        for subject in self.subjects:
            csv_file = enhanced_data_path / f"{subject}.csv"
            
            if csv_file.exists():
                try:
                    print(f"📊 Loading {subject}...")
                    df = pd.read_csv(csv_file)
                    
                    # Add subject identifier
                    df['Subject'] = subject
                    df['Subject_ID'] = subject.split('_')[0]  # T01, T02, etc.
                    
                    print(f"   ✅ {len(df):,} records loaded")
                    all_data.append(df)
                    successful_loads += 1
                    
                except Exception as e:
                    print(f"   ❌ Error loading {subject}: {e}")
            else:
                print(f"   ⚠️  File not found: {csv_file}")
        
        if successful_loads == 0:
            raise FileNotFoundError("No data files could be loaded")
            
        print(f"\n✅ Successfully loaded {successful_loads}/{len(self.subjects)} subjects")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"📊 Total combined records: {len(combined_df):,}")
        
        return self._process_loaded_data(combined_df)
    
    def _process_loaded_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean the loaded data."""
        print("\n🔧 Processing Mission Data...")
        print("-" * 30)
        
        # Convert time to datetime if needed
        time_columns = [col for col in df.columns if 'time' in col.lower() and ('s/1000' in col or 'ms' in col)]
        if time_columns:
            time_col = time_columns[0]
            print(f"⏰ Converting time column: {time_col}")
            df['datetime'] = pd.to_datetime(df[time_col], unit='ms', errors='coerce')
        
        # Ensure SOL column exists
        sol_columns = [col for col in df.columns if 'sol' in col.lower()]
        if sol_columns:
            sol_col = sol_columns[0]
            if sol_col != 'Sol':
                df['Sol'] = df[sol_col]
        else:
            print("⚠️  Warning: No SOL column found. Creating artificial SOL based on time.")
            if 'datetime' in df.columns:
                # Create SOL based on days from start
                df['Sol'] = (df['datetime'] - df['datetime'].min()).dt.days + 1
        
        # Ensure Subject column exists
        if 'Subject' not in df.columns and 'subject' in df.columns:
            df['Subject'] = df['subject']
        elif 'Subject' not in df.columns:
            print("⚠️  Warning: No subject identification found")
        
        # Clean data
        df = df.dropna(subset=['Sol'])  # Remove records without SOL
        
        print(f"✅ Data processing complete")
        print(f"📈 SOL range: {df['Sol'].min():.0f} - {df['Sol'].max():.0f}")
        print(f"👥 Subjects: {df['Subject'].nunique() if 'Subject' in df.columns else 'Unknown'}")
        
        return df
    
    def define_mission_phases(self) -> Dict[str, Tuple[float, float]]:
        """Define three mission phases based on SOL distribution."""
        print("\n📅 Defining Mission Phases...")
        print("-" * 30)
        
        if self.data is None or 'Sol' not in self.data.columns:
            raise ValueError("Data not loaded or SOL column missing")
        
        # Get SOL statistics
        sol_min = self.data['Sol'].min()
        sol_max = self.data['Sol'].max()
        sol_range = sol_max - sol_min
        
        # Define three equal phases
        phase_duration = sol_range / 3
        
        phases = {
            'Early': (sol_min, sol_min + phase_duration),
            'Mid': (sol_min + phase_duration, sol_min + 2 * phase_duration),  
            'Late': (sol_min + 2 * phase_duration, sol_max)
        }
        
        # Print phase definitions
        print(f"🔹 Early Phase: Sol {phases['Early'][0]:.1f} - {phases['Early'][1]:.1f}")
        print(f"🔹 Mid Phase:   Sol {phases['Mid'][0]:.1f} - {phases['Mid'][1]:.1f}")  
        print(f"🔹 Late Phase:  Sol {phases['Late'][0]:.1f} - {phases['Late'][1]:.1f}")
        
        # Add phase labels to data
        conditions = [
            (self.data['Sol'] >= phases['Early'][0]) & (self.data['Sol'] <= phases['Early'][1]),
            (self.data['Sol'] > phases['Mid'][0]) & (self.data['Sol'] <= phases['Mid'][1]),
            (self.data['Sol'] > phases['Late'][0]) & (self.data['Sol'] <= phases['Late'][1])
        ]
        choices = ['Early', 'Mid', 'Late']
        self.data['Mission_Phase'] = np.select(conditions, choices, default='Unknown')
        
        # Verify phase distribution
        phase_counts = self.data['Mission_Phase'].value_counts()
        print(f"\n📊 Phase Distribution:")
        for phase in ['Early', 'Mid', 'Late']:
            count = phase_counts.get(phase, 0)
            pct = (count / len(self.data)) * 100
            print(f"   {phase}: {count:,} records ({pct:.1f}%)")
        
        self.mission_phases = phases
        return phases
    
    def calculate_hrv_metrics(self) -> Dict[str, pd.DataFrame]:
        """Calculate HRV metrics using the enhanced HRV analysis system."""
        print("\n💓 Calculating HRV Metrics...")
        print("-" * 30)
        
        try:
            # Import the enhanced HRV analysis components
            sys.path.append('src/hrv_analysis/enhanced_hrv_analysis')
            from core.hrv_processor import HRVProcessor
            from core.signal_processing import SignalProcessor
            
            hrv_processor = HRVProcessor()
            signal_processor = SignalProcessor()
            
            hrv_results = []
            
            # Calculate HRV for each subject and phase combination
            for subject in self.data['Subject'].unique():
                if pd.isna(subject):
                    continue
                    
                subject_data = self.data[self.data['Subject'] == subject]
                print(f"👤 Processing {subject}...")
                
                for phase in ['Early', 'Mid', 'Late']:
                    phase_data = subject_data[subject_data['Mission_Phase'] == phase]
                    
                    if len(phase_data) < 100:  # Minimum data requirement
                        print(f"   ⚠️  Insufficient data for {phase} phase ({len(phase_data)} records)")
                        continue
                    
                    # Extract heart rate data
                    if 'heart_rate [bpm]' in phase_data.columns:
                        hr_data = phase_data['heart_rate [bpm]'].dropna()
                        
                        if len(hr_data) < 50:
                            continue
                        
                        try:
                            # Convert to RR intervals
                            rr_intervals, _ = signal_processor.compute_rr_intervals(hr_data)
                            
                            if len(rr_intervals) < 50:
                                continue
                            
                            # Calculate HRV metrics
                            hrv_result = hrv_processor.compute_hrv_metrics(rr_intervals)
                            
                            # Extract key metrics
                            result_dict = {
                                'Subject': subject,
                                'Phase': phase,
                                'n_intervals': len(rr_intervals)
                            }
                            
                            # Time domain metrics
                            if 'time_domain' in hrv_result:
                                td = hrv_result['time_domain']
                                result_dict.update({
                                    'SDNN': td.sdnn,
                                    'RMSSD': td.rmssd, 
                                    'pNN50': td.pnn50,
                                    'Mean_HR': td.mean_hr
                                })
                            
                            # Frequency domain metrics
                            if 'frequency_domain' in hrv_result:
                                fd = hrv_result['frequency_domain']
                                result_dict.update({
                                    'HF_Power': fd.hf_power,
                                    'LF_Power': fd.lf_power,
                                    'VLF_Power': fd.vlf_power,
                                    'LF_HF_Ratio': fd.lf_hf_ratio
                                })
                            
                            # Nonlinear metrics
                            if 'nonlinear' in hrv_result:
                                nl = hrv_result['nonlinear']
                                result_dict.update({
                                    'SD1': nl.get('sd1', np.nan),
                                    'SD2': nl.get('sd2', np.nan),
                                    'DFA_alpha1': nl.get('dfa_alpha1', np.nan),
                                    'DFA_alpha2': nl.get('dfa_alpha2', np.nan)
                                })
                            
                            hrv_results.append(result_dict)
                            print(f"   ✅ {phase} phase: {len(rr_intervals)} RR intervals")
                            
                        except Exception as e:
                            print(f"   ❌ Error calculating HRV for {phase}: {e}")
                            continue
            
            if hrv_results:
                hrv_df = pd.DataFrame(hrv_results)
                print(f"\n✅ HRV calculation complete: {len(hrv_df)} phase analyses")
                self.hrv_metrics = {'combined': hrv_df}
                return self.hrv_metrics
            else:
                print("❌ No HRV metrics could be calculated")
                return {}
                
        except ImportError as e:
            print(f"❌ Could not import HRV analysis components: {e}")
            print("📊 Will analyze only physiological metrics")
            return {}
        except Exception as e:
            print(f"❌ Error in HRV calculation: {e}")
            return {}
    
    def aggregate_physiological_metrics(self) -> Dict[str, pd.DataFrame]:
        """Aggregate physiological metrics by subject and phase."""
        print("\n📊 Aggregating Physiological Metrics...")
        print("-" * 40)
        
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # Find available physiological variables
        available_vars = [var for var in self.key_variables if var in self.data.columns]
        print(f"📈 Available variables: {len(available_vars)}")
        for var in available_vars:
            print(f"   • {var}")
        
        physio_results = []
        
        for subject in self.data['Subject'].unique():
            if pd.isna(subject):
                continue
                
            subject_data = self.data[self.data['Subject'] == subject]
            print(f"👤 Processing {subject}...")
            
            for phase in ['Early', 'Mid', 'Late']:
                phase_data = subject_data[subject_data['Mission_Phase'] == phase]
                
                if len(phase_data) == 0:
                    continue
                
                result_dict = {
                    'Subject': subject,
                    'Phase': phase,
                    'n_records': len(phase_data)
                }
                
                # Calculate statistics for each available variable
                for var in available_vars:
                    if var in phase_data.columns:
                        values = phase_data[var].dropna()
                        if len(values) > 0:
                            result_dict[f"{var}_mean"] = values.mean()
                            result_dict[f"{var}_std"] = values.std()
                            result_dict[f"{var}_median"] = values.median()
                            result_dict[f"{var}_min"] = values.min()
                            result_dict[f"{var}_max"] = values.max()
                            result_dict[f"{var}_count"] = len(values)
                
                physio_results.append(result_dict)
        
        if physio_results:
            physio_df = pd.DataFrame(physio_results)
            print(f"\n✅ Physiological aggregation complete: {len(physio_df)} phase analyses")
            self.physiological_metrics = {'combined': physio_df}
            return self.physiological_metrics
        else:
            print("❌ No physiological metrics could be calculated")
            return {}
    
    def create_individual_boxplots(self, output_dir: str = "plots_output") -> None:
        """Create boxplots for each individual across mission phases."""
        print("\n👤 Creating Individual Boxplots...")
        print("-" * 35)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # HRV boxplots for individuals
        if self.hrv_metrics:
            hrv_df = self.hrv_metrics['combined']
            self._create_individual_hrv_boxplots(hrv_df, output_path)
        
        # Physiological boxplots for individuals  
        if self.physiological_metrics:
            physio_df = self.physiological_metrics['combined']
            self._create_individual_physio_boxplots(physio_df, output_path)
    
    def _create_individual_hrv_boxplots(self, hrv_df: pd.DataFrame, output_path: Path) -> None:
        """Create individual HRV boxplots."""
        print("💓 Creating individual HRV boxplots...")
        
        # Select key HRV metrics for plotting
        plot_metrics = ['SDNN', 'RMSSD', 'Mean_HR', 'LF_HF_Ratio']
        available_metrics = [m for m in plot_metrics if m in hrv_df.columns]
        
        if not available_metrics:
            print("⚠️  No HRV metrics available for plotting")
            return
        
        # Create subplot grid
        n_metrics = len(available_metrics)
        n_subjects = hrv_df['Subject'].nunique()
        
        fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 4 * n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            
            # Create boxplot for each subject
            subjects = sorted(hrv_df['Subject'].unique())
            plot_data = []
            plot_labels = []
            
            for subject in subjects:
                subject_data = hrv_df[hrv_df['Subject'] == subject]
                for phase in ['Early', 'Mid', 'Late']:
                    phase_data = subject_data[subject_data['Phase'] == phase]
                    if len(phase_data) > 0 and not phase_data[metric].isna().all():
                        plot_data.append(phase_data[metric].values)
                        plot_labels.append(f"{subject}\n{phase}")
            
            if plot_data:
                bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
                
                # Color by phase
                colors = ['lightblue', 'lightgreen', 'lightcoral']
                phase_colors = colors * (len(plot_data) // 3 + 1)
                
                for patch, color in zip(bp['boxes'], phase_colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax.set_title(f'Individual {metric} Across Mission Phases', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path / 'individual_hrv_boxplots.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"   ✅ Saved: individual_hrv_boxplots.png")
    
    def _create_individual_physio_boxplots(self, physio_df: pd.DataFrame, output_path: Path) -> None:
        """Create individual physiological boxplots."""
        print("📊 Creating individual physiological boxplots...")
        
        # Select key physiological metrics (mean values)
        mean_metrics = [col for col in physio_df.columns if col.endswith('_mean')]
        plot_metrics = mean_metrics[:4]  # Limit to 4 for readability
        
        if not plot_metrics:
            print("⚠️  No physiological metrics available for plotting")
            return
        
        # Create subplot grid
        n_metrics = len(plot_metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 4 * n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(plot_metrics):
            ax = axes[i]
            
            # Create boxplot for each subject
            subjects = sorted(physio_df['Subject'].unique())
            plot_data = []
            plot_labels = []
            
            for subject in subjects:
                subject_data = physio_df[physio_df['Subject'] == subject]
                for phase in ['Early', 'Mid', 'Late']:
                    phase_data = subject_data[subject_data['Phase'] == phase]
                    if len(phase_data) > 0 and not phase_data[metric].isna().all():
                        plot_data.append(phase_data[metric].values)
                        plot_labels.append(f"{subject}\n{phase}")
            
            if plot_data:
                bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
                
                # Color by phase
                colors = ['lightblue', 'lightgreen', 'lightcoral']
                phase_colors = colors * (len(plot_data) // 3 + 1)
                
                for patch, color in zip(bp['boxes'], phase_colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            # Clean up metric name for title
            clean_metric = metric.replace('_mean', '').replace('[', ' (').replace(']', ')')
            ax.set_title(f'Individual {clean_metric} Across Mission Phases', fontsize=14, fontweight='bold')
            ax.set_ylabel(clean_metric, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path / 'individual_physio_boxplots.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"   ✅ Saved: individual_physio_boxplots.png")
    
    def create_group_boxplots(self, output_dir: str = "plots_output") -> None:
        """Create boxplots for the whole group comparing mission phases."""
        print("\n👥 Creating Group Boxplots...")
        print("-" * 30)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # HRV group boxplots
        if self.hrv_metrics:
            hrv_df = self.hrv_metrics['combined']
            self._create_group_hrv_boxplots(hrv_df, output_path)
        
        # Physiological group boxplots
        if self.physiological_metrics:
            physio_df = self.physiological_metrics['combined']
            self._create_group_physio_boxplots(physio_df, output_path)
    
    def _create_group_hrv_boxplots(self, hrv_df: pd.DataFrame, output_path: Path) -> None:
        """Create group HRV boxplots."""
        print("💓 Creating group HRV boxplots...")
        
        # Select key HRV metrics for plotting
        plot_metrics = ['SDNN', 'RMSSD', 'Mean_HR', 'LF_HF_Ratio', 'HF_Power', 'LF_Power']
        available_metrics = [m for m in plot_metrics if m in hrv_df.columns]
        
        if not available_metrics:
            print("⚠️  No HRV metrics available for plotting")
            return
        
        # Create subplot grid (2x3 layout)
        n_metrics = len(available_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            
            # Prepare data for boxplot
            phase_data = []
            phase_labels = ['Early', 'Mid', 'Late']
            
            for phase in phase_labels:
                values = hrv_df[hrv_df['Phase'] == phase][metric].dropna()
                phase_data.append(values)
            
            if any(len(data) > 0 for data in phase_data):
                bp = ax.boxplot(phase_data, labels=phase_labels, patch_artist=True)
                
                # Color boxes
                colors = ['lightblue', 'lightgreen', 'lightcoral']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Add statistical annotations
                self._add_statistical_annotations(ax, phase_data, phase_labels, metric)
            
            ax.set_title(f'Group {metric} by Mission Phase', fontsize=12, fontweight='bold')
            ax.set_xlabel('Mission Phase', fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(available_metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(output_path / 'group_hrv_boxplots.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"   ✅ Saved: group_hrv_boxplots.png")
    
    def _create_group_physio_boxplots(self, physio_df: pd.DataFrame, output_path: Path) -> None:
        """Create group physiological boxplots."""
        print("📊 Creating group physiological boxplots...")
        
        # Select key physiological metrics (mean values)
        mean_metrics = [col for col in physio_df.columns if col.endswith('_mean')]
        plot_metrics = mean_metrics[:6]  # Limit to 6 for readability
        
        if not plot_metrics:
            print("⚠️  No physiological metrics available for plotting")
            return
        
        # Create subplot grid (2x3 layout)
        n_metrics = len(plot_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, metric in enumerate(plot_metrics):
            ax = axes[i]
            
            # Prepare data for boxplot
            phase_data = []
            phase_labels = ['Early', 'Mid', 'Late']
            
            for phase in phase_labels:
                values = physio_df[physio_df['Phase'] == phase][metric].dropna()
                phase_data.append(values)
            
            if any(len(data) > 0 for data in phase_data):
                bp = ax.boxplot(phase_data, labels=phase_labels, patch_artist=True)
                
                # Color boxes
                colors = ['lightblue', 'lightgreen', 'lightcoral']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Add statistical annotations
                self._add_statistical_annotations(ax, phase_data, phase_labels, metric)
            
            # Clean up metric name for title
            clean_metric = metric.replace('_mean', '').replace('[', ' (').replace(']', ')')
            ax.set_title(f'Group {clean_metric} by Mission Phase', fontsize=12, fontweight='bold')
            ax.set_xlabel('Mission Phase', fontsize=10)
            ax.set_ylabel(clean_metric, fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(plot_metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(output_path / 'group_physio_boxplots.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"   ✅ Saved: group_physio_boxplots.png")
    
    def _add_statistical_annotations(self, ax, phase_data: List, phase_labels: List, metric: str) -> None:
        """Add statistical test results to boxplots."""
        # Perform Kruskal-Wallis test (non-parametric)
        valid_data = [data for data in phase_data if len(data) > 0]
        
        if len(valid_data) >= 2:
            try:
                h_stat, p_value = stats.kruskal(*valid_data)
                
                # Add p-value annotation
                if p_value < 0.001:
                    p_text = "p < 0.001"
                elif p_value < 0.01:
                    p_text = f"p < 0.01"
                elif p_value < 0.05:
                    p_text = f"p < 0.05"
                else:
                    p_text = f"p = {p_value:.3f}"
                
                # Position text at top of plot
                y_pos = ax.get_ylim()[1] * 0.95
                ax.text(0.5, y_pos, f"Kruskal-Wallis: {p_text}", 
                       transform=ax.get_xaxis_transform(), 
                       ha='center', va='top', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                       
            except Exception as e:
                print(f"   ⚠️  Statistical test failed for {metric}: {e}")
    
    def perform_statistical_analysis(self) -> Dict[str, pd.DataFrame]:
        """Perform comprehensive statistical analysis comparing mission phases."""
        print("\n📈 Performing Statistical Analysis...")
        print("-" * 40)
        
        results = {}
        
        # HRV statistical analysis
        if self.hrv_metrics:
            print("💓 Analyzing HRV metrics...")
            hrv_stats = self._analyze_metrics_statistically(
                self.hrv_metrics['combined'], 
                ['SDNN', 'RMSSD', 'Mean_HR', 'LF_HF_Ratio'],
                "HRV"
            )
            results['hrv'] = hrv_stats
        
        # Physiological statistical analysis
        if self.physiological_metrics:
            print("📊 Analyzing physiological metrics...")
            mean_metrics = [col for col in self.physiological_metrics['combined'].columns 
                          if col.endswith('_mean')][:4]
            physio_stats = self._analyze_metrics_statistically(
                self.physiological_metrics['combined'],
                mean_metrics,
                "Physiological"
            )
            results['physiological'] = physio_stats
        
        return results
    
    def _analyze_metrics_statistically(self, df: pd.DataFrame, metrics: List[str], 
                                     analysis_type: str) -> pd.DataFrame:
        """Perform statistical analysis on specific metrics."""
        statistical_results = []
        
        for metric in metrics:
            if metric not in df.columns:
                continue
                
            print(f"   📊 Testing {metric}...")
            
            # Extract data for each phase
            early_data = df[df['Phase'] == 'Early'][metric].dropna()
            mid_data = df[df['Phase'] == 'Mid'][metric].dropna()
            late_data = df[df['Phase'] == 'Late'][metric].dropna()
            
            # Skip if insufficient data
            if len(early_data) < 2 or len(mid_data) < 2 or len(late_data) < 2:
                print(f"      ⚠️  Insufficient data for {metric}")
                continue
            
            # Descriptive statistics
            result = {
                'Metric': metric,
                'Analysis_Type': analysis_type,
                'Early_Mean': early_data.mean(),
                'Early_Std': early_data.std(),
                'Early_N': len(early_data),
                'Mid_Mean': mid_data.mean(),
                'Mid_Std': mid_data.std(),
                'Mid_N': len(mid_data),
                'Late_Mean': late_data.mean(),
                'Late_Std': late_data.std(),
                'Late_N': len(late_data)
            }
            
            # Kruskal-Wallis test (non-parametric ANOVA)
            try:
                h_stat, p_value = stats.kruskal(early_data, mid_data, late_data)
                result['Kruskal_Wallis_H'] = h_stat
                result['Kruskal_Wallis_p'] = p_value
                
                # Effect size (eta-squared approximation)
                n = len(early_data) + len(mid_data) + len(late_data)
                result['Effect_Size_eta2'] = (h_stat - 2) / (n - 3) if n > 3 else 0
                
                # Interpretation
                if p_value < 0.001:
                    result['Significance'] = '***'
                    result['Interpretation'] = 'Highly significant difference'
                elif p_value < 0.01:
                    result['Significance'] = '**'
                    result['Interpretation'] = 'Very significant difference'
                elif p_value < 0.05:
                    result['Significance'] = '*'
                    result['Interpretation'] = 'Significant difference'
                else:
                    result['Significance'] = 'ns'
                    result['Interpretation'] = 'No significant difference'
                
                print(f"      ✅ H = {h_stat:.3f}, p = {p_value:.3f} ({result['Interpretation']})")
                
            except Exception as e:
                print(f"      ❌ Statistical test failed: {e}")
                result['Kruskal_Wallis_H'] = np.nan
                result['Kruskal_Wallis_p'] = np.nan
                result['Significance'] = 'Error'
                result['Interpretation'] = 'Test failed'
            
            statistical_results.append(result)
        
        return pd.DataFrame(statistical_results) if statistical_results else pd.DataFrame()
    
    def generate_summary_report(self, statistical_results: Dict[str, pd.DataFrame], 
                              output_dir: str = "plots_output") -> None:
        """Generate a comprehensive summary report."""
        print("\n📝 Generating Summary Report...")
        print("-" * 35)
        
        output_path = Path(output_dir)
        report_file = output_path / "mission_phases_analysis_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("VALQUIRIA MISSION PHASES ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mission Phases: {list(self.mission_phases.keys()) if self.mission_phases else 'Not defined'}\n\n")
            
            if self.mission_phases:
                f.write("MISSION PHASE DEFINITIONS:\n")
                f.write("-" * 30 + "\n")
                for phase, (start, end) in self.mission_phases.items():
                    f.write(f"{phase} Phase: Sol {start:.1f} - {end:.1f}\n")
                f.write("\n")
            
            # Data summary
            if self.data is not None:
                f.write("DATA SUMMARY:\n")
                f.write("-" * 15 + "\n")
                f.write(f"Total Records: {len(self.data):,}\n")
                f.write(f"Subjects: {self.data['Subject'].nunique() if 'Subject' in self.data.columns else 'Unknown'}\n")
                f.write(f"SOL Range: {self.data['Sol'].min():.0f} - {self.data['Sol'].max():.0f}\n\n")
                
                # Phase distribution
                if 'Mission_Phase' in self.data.columns:
                    phase_counts = self.data['Mission_Phase'].value_counts()
                    f.write("Phase Distribution:\n")
                    for phase in ['Early', 'Mid', 'Late']:
                        count = phase_counts.get(phase, 0)
                        pct = (count / len(self.data)) * 100
                        f.write(f"  {phase}: {count:,} records ({pct:.1f}%)\n")
                    f.write("\n")
            
            # Statistical results
            for analysis_type, results_df in statistical_results.items():
                if not results_df.empty:
                    f.write(f"{analysis_type.upper()} STATISTICAL RESULTS:\n")
                    f.write("-" * (len(analysis_type) + 20) + "\n")
                    
                    for _, row in results_df.iterrows():
                        f.write(f"\n{row['Metric']}:\n")
                        f.write(f"  Early Phase:  {row['Early_Mean']:.3f} ± {row['Early_Std']:.3f} (n={row['Early_N']})\n")
                        f.write(f"  Mid Phase:    {row['Mid_Mean']:.3f} ± {row['Mid_Std']:.3f} (n={row['Mid_N']})\n")
                        f.write(f"  Late Phase:   {row['Late_Mean']:.3f} ± {row['Late_Std']:.3f} (n={row['Late_N']})\n")
                        
                        if not pd.isna(row['Kruskal_Wallis_p']):
                            f.write(f"  Kruskal-Wallis: H = {row['Kruskal_Wallis_H']:.3f}, p = {row['Kruskal_Wallis_p']:.3f}\n")
                            f.write(f"  Effect Size (η²): {row['Effect_Size_eta2']:.3f}\n")
                            f.write(f"  Result: {row['Interpretation']}\n")
                        else:
                            f.write(f"  Statistical test failed\n")
                    
                    f.write("\n" + "="*50 + "\n\n")
            
            # Interpretation and conclusions
            f.write("SCIENTIFIC INTERPRETATION:\n")
            f.write("-" * 25 + "\n")
            f.write("This analysis examined physiological and HRV changes across three\n")
            f.write("mission phases (Early, Mid, Late) in the Valquiria space analog simulation.\n\n")
            
            # Count significant results
            total_sig = 0
            total_tests = 0
            for results_df in statistical_results.values():
                if not results_df.empty:
                    total_tests += len(results_df)
                    total_sig += len(results_df[results_df['Significance'].isin(['*', '**', '***'])])
            
            if total_tests > 0:
                f.write(f"Statistical Tests Performed: {total_tests}\n")
                f.write(f"Significant Results (p < 0.05): {total_sig}\n")
                f.write(f"Percentage Significant: {(total_sig/total_tests)*100:.1f}%\n\n")
            
            f.write("The analysis follows aerospace medicine standards and uses\n")
            f.write("non-parametric statistical methods appropriate for physiological data.\n\n")
            
            f.write("FILES GENERATED:\n")
            f.write("-" * 15 + "\n")
            f.write("• individual_hrv_boxplots.png - Individual HRV comparisons\n")
            f.write("• individual_physio_boxplots.png - Individual physiological comparisons\n")
            f.write("• group_hrv_boxplots.png - Group HRV comparisons\n")
            f.write("• group_physio_boxplots.png - Group physiological comparisons\n")
            f.write("• mission_phases_analysis_report.txt - This comprehensive report\n\n")
            
        print(f"   ✅ Report saved: {report_file}")
    
    def run_complete_analysis(self) -> None:
        """Run the complete mission phase analysis."""
        print("🚀 STARTING COMPLETE MISSION PHASE ANALYSIS")
        print("=" * 55)
        print("📊 Analyzing Three Moments During the Valquiria Mission")
        print("👥 Individual and Group Comparisons")
        print("=" * 55)
        
        try:
            # Step 1: Load data
            self.data = self.load_mission_data()
            
            # Step 2: Define mission phases
            self.define_mission_phases()
            
            # Step 3: Calculate HRV metrics
            self.calculate_hrv_metrics()
            
            # Step 4: Aggregate physiological metrics
            self.aggregate_physiological_metrics()
            
            # Step 5: Create individual boxplots
            self.create_individual_boxplots()
            
            # Step 6: Create group boxplots
            self.create_group_boxplots()
            
            # Step 7: Perform statistical analysis
            statistical_results = self.perform_statistical_analysis()
            
            # Step 8: Generate summary report
            self.generate_summary_report(statistical_results)
            
            print("\n🎉 ANALYSIS COMPLETE!")
            print("=" * 25)
            print("✅ All boxplots generated successfully")
            print("✅ Statistical analysis completed")
            print("✅ Summary report created")
            print(f"📁 Check the 'plots_output' folder for all results")
            
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the mission phase analysis."""
    print("🌟 VALQUIRIA MISSION PHASE BOXPLOT ANALYSIS")
    print("=" * 50)
    print("🎯 Objective: Compare three mission phases (Early, Mid, Late)")
    print("📊 Output: Individual and group boxplots + statistical analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = MissionPhaseAnalyzer()
    
    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main() 