import os
import json
import wave
import struct
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import scipy
from scipy import stats, signal, interpolate
import warnings
import csv
from datetime import datetime, timedelta

class HexoskinWavLoader:
    """
    A class for loading and processing Hexoskin WAV files containing physiological data.
    
    Hexoskin WAV files contain binary data where:
    - First column: timestamp (seconds from start of recording)
    - Second column: physiological measurement value
    """
    
    def __init__(self):
        self.data = None
        self.timestamps = None
        self.values = None
        self.sample_rate = None
        self.file_path = None
        self.metadata = {}
        self.start_timestamp = None
        
    def load_wav_file(self, file_path):
        """
        Load a Hexoskin WAV file using the struct module to unpack binary data.
        
        Args:
            file_path (str): Path to the WAV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.file_path = file_path
            
            # Get filename without extension for metadata
            self.metadata['sensor_name'] = os.path.splitext(os.path.basename(file_path))[0]
            
            # Try to load info.json for timestamp information
            self._load_info_json()
            
            # Open the WAV file to get sample rate
            with wave.open(file_path, 'rb') as wav_file:
                self.sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()
                
                self.metadata['sample_rate'] = self.sample_rate
                self.metadata['n_channels'] = n_channels
                self.metadata['sample_width'] = sample_width
                self.metadata['n_frames'] = n_frames
                
                print(f"Sample rate: {self.sample_rate} Hz")
                print(f"Channels: {n_channels}")
                print(f"Sample width: {sample_width} bytes")
                print(f"Number of frames: {n_frames}")
            
            # Using struct to unpack binary data
            timestamps = []
            values = []
            
            with open(file_path, 'rb') as f:
                # Skip WAV header (44 bytes)
                f.seek(44)
                
                # For synchronous data, Hexoskin uses 'h' (short) format 
                data_format = struct.Struct("h")
                
                # Read data
                byte_data = f.read(data_format.size)
                frame_count = 0
                
                while byte_data:
                    # Extract value
                    value = data_format.unpack(byte_data)[0]
                    
                    # Calculate timestamp based on sample rate and frame count
                    timestamp = frame_count / self.sample_rate
                    
                    timestamps.append(timestamp)
                    values.append(value)
                    
                    # Read next frame
                    byte_data = f.read(data_format.size)
                    frame_count += 1
            
            self.timestamps = np.array(timestamps)
            self.values = np.array(values)
            
            # Create basic DataFrame with relative timestamps
            self.data = pd.DataFrame({
                'timestamp': self.timestamps,
                'value': self.values
            })
            
            # Add absolute timestamps if available
            if self.start_timestamp is not None:
                self.data['abs_timestamp'] = self.start_timestamp + self.data['timestamp']
                self.data['datetime'] = pd.to_datetime(self.data['abs_timestamp'], unit='s')
            
            print(f"Loaded {len(self.data)} data points")
            return True
            
        except Exception as e:
            print(f"Error loading WAV file: {e}")
            return False
    
    def _load_info_json(self):
        """Load info.json file to get timestamp information"""
        try:
            # Get directory of the WAV file
            dir_path = os.path.dirname(self.file_path)
            
            # Look for info.json in the directory
            info_path = os.path.join(dir_path, 'info.json')
            
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    info_data = json.load(f)
                
                # Extract timestamp information
                if 'timestamp' in info_data:
                    self.start_timestamp = info_data['timestamp']
                    self.metadata['start_timestamp'] = self.start_timestamp
                
                if 'start_date' in info_data:
                    self.metadata['start_date'] = info_data['start_date']
                
                # Store other useful metadata
                if 'user' in info_data:
                    self.metadata['user_id'] = info_data['user']
                
                if 'devices' in info_data:
                    self.metadata['devices'] = ', '.join(info_data['devices'])
                
                print(f"Loaded timestamp data from info.json: {self.start_timestamp}")
            else:
                print("No info.json found, using relative timestamps")
        except Exception as e:
            print(f"Error loading info.json: {e}")
    
    def get_metadata(self):
        """Return metadata about the loaded file"""
        return self.metadata
    
    def get_data(self):
        """Return the DataFrame with timestamp and value columns"""
        return self.data
    
    def get_descriptive_stats(self):
        """
        Calculate descriptive statistics for the loaded data
        
        Returns:
            dict: Comprehensive descriptive statistics
        """
        if self.data is None:
            print("No data loaded")
            return None
        
        # Basic descriptive statistics from pandas
        basic_stats = self.data['value'].describe().to_dict()
        
        # Additional statistics
        data_values = self.data['value'].values
        
        # Calculate additional statistics
        variance = np.var(data_values, ddof=1)
        skewness = stats.skew(data_values)
        kurtosis = stats.kurtosis(data_values)
        
        # Interquartile range and outlier thresholds
        q1 = np.percentile(data_values, 25)
        q3 = np.percentile(data_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Range statistics
        data_range = basic_stats['max'] - basic_stats['min']
        
        # Coefficient of variation
        cv = basic_stats['std'] / basic_stats['mean'] if basic_stats['mean'] != 0 else float('nan')
        
        # Median absolute deviation
        median = basic_stats['50%']
        mad = np.median(np.abs(data_values - median))
        
        # Percentiles
        percentiles = {}
        for p in [1, 5, 10, 20, 30, 40, 60, 70, 80, 90, 95, 99]:
            percentiles[f'p{p}'] = np.percentile(data_values, p)
        
        # Count outliers
        outliers = np.sum((data_values < lower_bound) | (data_values > upper_bound))
        
        # Run normality tests
        normality_tests = self.test_normality()
        
        # Find mode - handle both SciPy 1.x and 1.7+ return values
        try:
            # Check if mode_result.mode is a scalar or array-like
            mode_result = stats.mode(data_values)
            
            # Handle both old and new SciPy versions
            if hasattr(mode_result, 'mode'):
                if np.isscalar(mode_result.mode):
                    mode_value = mode_result.mode
                    mode_count = mode_result.count
                else:
                    mode_value = mode_result.mode[0]
                    mode_count = mode_result.count[0]
            else:
                # For newer versions where mode_result is a tuple (mode, count)
                mode_value = mode_result[0]
                mode_count = mode_result[1]
        except Exception as e:
            # Fallback method if stats.mode fails
            print(f"Error calculating mode: {e}")
            unique_values, counts = np.unique(data_values, return_counts=True)
            mode_idx = np.argmax(counts)
            mode_value = unique_values[mode_idx]
            mode_count = counts[mode_idx]
        
        # Combine all statistics
        return {
            'basic': basic_stats,
            'additional': {
                'variance': variance,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower_outlier_bound': lower_bound,
                'upper_outlier_bound': upper_bound,
                'range': data_range,
                'coefficient_of_variation': cv,
                'median_absolute_deviation': mad,
                'mode': mode_value,
                'mode_count': mode_count,
                'outlier_count': outliers,
                'percentiles': percentiles
            },
            'normality': normality_tests
        }

    @staticmethod
    def align_datasets(dataset1, dataset2, target_hz=None):
        """
        Align two datasets to have the same time points and length.
        
        Args:
            dataset1 (pd.DataFrame): First dataset with timestamp and value columns
            dataset2 (pd.DataFrame): Second dataset with timestamp and value columns
            target_hz (float, optional): Target sampling rate for resampling. 
                                         If None, uses the lower of the two sampling rates.
                                         
        Returns:
            tuple: (aligned_dataset1, aligned_dataset2) - DataFrames with aligned data
        """
        # Check if datasets are DataFrames
        if not isinstance(dataset1, pd.DataFrame) or not isinstance(dataset2, pd.DataFrame):
            raise ValueError("Datasets must be pandas DataFrames")
            
        # Check for required columns
        if 'timestamp' not in dataset1.columns or 'value' not in dataset1.columns:
            raise ValueError("Dataset1 must have 'timestamp' and 'value' columns")
            
        if 'timestamp' not in dataset2.columns or 'value' not in dataset2.columns:
            raise ValueError("Dataset2 must have 'timestamp' and 'value' columns")
            
        # Calculate original sampling rates
        if len(dataset1) > 1:
            time_diff1 = dataset1['timestamp'].iloc[1] - dataset1['timestamp'].iloc[0]
            sr1 = 1 / time_diff1 if time_diff1 > 0 else 1
        else:
            sr1 = 1
            
        if len(dataset2) > 1:
            time_diff2 = dataset2['timestamp'].iloc[1] - dataset2['timestamp'].iloc[0]
            sr2 = 1 / time_diff2 if time_diff2 > 0 else 1
        else:
            sr2 = 1
            
        # Determine target sampling rate if not provided
        if target_hz is None:
            target_hz = min(sr1, sr2)
            
        # Determine common time range
        start_time = max(dataset1['timestamp'].min(), dataset2['timestamp'].min())
        end_time = min(dataset1['timestamp'].max(), dataset2['timestamp'].max())
        
        # If no overlap, return empty DataFrames
        if end_time <= start_time:
            print("Datasets don't have overlapping time range")
            return pd.DataFrame(columns=['timestamp', 'value']), pd.DataFrame(columns=['timestamp', 'value'])
            
        # Create new time points at target_hz frequency
        time_points = np.arange(start_time, end_time, 1/target_hz)
        
        # Interpolate both datasets to the new time points
        interp_func1 = interpolate.interp1d(
            dataset1['timestamp'], 
            dataset1['value'],
            bounds_error=False,
            fill_value="extrapolate"
        )
        
        interp_func2 = interpolate.interp1d(
            dataset2['timestamp'], 
            dataset2['value'],
            bounds_error=False,
            fill_value="extrapolate"
        )
        
        aligned_values1 = interp_func1(time_points)
        aligned_values2 = interp_func2(time_points)
        
        # Create aligned DataFrames
        aligned_df1 = pd.DataFrame({
            'timestamp': time_points,
            'value': aligned_values1
        })
        
        aligned_df2 = pd.DataFrame({
            'timestamp': time_points,
            'value': aligned_values2
        })
        
        return aligned_df1, aligned_df2
    
    @staticmethod
    def normalize_dataset(dataset, method='min_max'):
        """
        Normalize the values in a dataset.
        
        Args:
            dataset (pd.DataFrame): Dataset with timestamp and value columns
            method (str): Normalization method
                - 'min_max': Scales data to range [0, 1]
                - 'z_score': Standardizes data to mean=0, std=1
                - 'robust': Uses median and IQR for robust scaling
                
        Returns:
            pd.DataFrame: Normalized dataset
        """
        # Check if dataset is a DataFrame
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("Dataset must be a pandas DataFrame")
            
        # Check for required columns
        if 'timestamp' not in dataset.columns or 'value' not in dataset.columns:
            raise ValueError("Dataset must have 'timestamp' and 'value' columns")
            
        # Create a copy of the dataset
        normalized_df = dataset.copy()
        
        # Apply normalization based on method
        if method == 'min_max':
            min_val = dataset['value'].min()
            max_val = dataset['value'].max()
            
            if max_val > min_val:
                normalized_df['value'] = (dataset['value'] - min_val) / (max_val - min_val)
            else:
                normalized_df['value'] = 0
                
        elif method == 'z_score':
            mean_val = dataset['value'].mean()
            std_val = dataset['value'].std()
            
            if std_val > 0:
                normalized_df['value'] = (dataset['value'] - mean_val) / std_val
            else:
                normalized_df['value'] = 0
                
        elif method == 'robust':
            median_val = dataset['value'].median()
            q1 = dataset['value'].quantile(0.25)
            q3 = dataset['value'].quantile(0.75)
            iqr = q3 - q1
            
            if iqr > 0:
                normalized_df['value'] = (dataset['value'] - median_val) / iqr
            else:
                normalized_df['value'] = 0
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        return normalized_df
    
    @staticmethod
    def compare_datasets(dataset1, dataset2, test_type='mann_whitney'):
        """
        Perform statistical comparison between two datasets.
        
        Args:
            dataset1 (pd.DataFrame or np.ndarray or list): First dataset with values
            dataset2 (pd.DataFrame or np.ndarray or list): Second dataset with values
            test_type (str): Type of statistical test to perform:
                - 'mann_whitney': Mann-Whitney U test (default)
                - 'wilcoxon': Wilcoxon signed-rank test (for paired samples)
                - 'ks_2samp': Two-sample Kolmogorov-Smirnov test
                - 't_test': Independent t-test (for samples with different lengths)
                - 'welch_t_test': Welch's t-test (for unequal variances)
            
        Returns:
            dict: Results of the statistical test
        """
        # Ensure datasets are numpy arrays
        if isinstance(dataset1, pd.DataFrame):
            if 'value' in dataset1.columns:
                data1 = dataset1['value'].values
            else:
                data1 = dataset1.iloc[:, 0].values
        elif isinstance(dataset1, (list, tuple)):
            data1 = np.array(dataset1)
        else:
            data1 = dataset1
            
        if isinstance(dataset2, pd.DataFrame):
            if 'value' in dataset2.columns:
                data2 = dataset2['value'].values
            else:
                data2 = dataset2.iloc[:, 0].values
        elif isinstance(dataset2, (list, tuple)):
            data2 = np.array(dataset2)
        else:
            data2 = dataset2
        
        # Handle datasets of different lengths for Wilcoxon test
        if len(data1) != len(data2) and test_type == 'wilcoxon':
            # If lengths differ and user requested Wilcoxon, switch to Mann-Whitney test
            test_type = 'mann_whitney'
            test_switch_message = "Switched to Mann-Whitney U test due to different dataset lengths"
        else:
            test_switch_message = None
        
        # Calculate descriptive statistics for each dataset
        desc1 = {
            'mean': np.mean(data1),
            'median': np.median(data1),
            'std': np.std(data1, ddof=1),  # Using n-1 for sample standard deviation
            'min': np.min(data1),
            'max': np.max(data1),
            'count': len(data1)
        }
        
        desc2 = {
            'mean': np.mean(data2),
            'median': np.median(data2),
            'std': np.std(data2, ddof=1),  # Using n-1 for sample standard deviation
            'min': np.min(data2),
            'max': np.max(data2),
            'count': len(data2)
        }
        
        # Perform statistical test
        alpha = 0.05
        ci_level = 0.95
        
        if test_type == 'mann_whitney':
            # Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            test_name = "Mann-Whitney U test"
            test_description = "Non-parametric test to determine if two independent samples are drawn from the same distribution"
            null_hypothesis = "The distributions of both samples are equal"
            
            # Compute approximate confidence interval using bootstrap
            n_bootstrap = 1000
            bootstrap_differences = []
            
            for _ in range(n_bootstrap):
                sample1 = np.random.choice(data1, size=len(data1), replace=True)
                sample2 = np.random.choice(data2, size=len(data2), replace=True)
                bootstrap_differences.append(np.median(sample1) - np.median(sample2))
                
            ci_lower = np.percentile(bootstrap_differences, 2.5)
            ci_upper = np.percentile(bootstrap_differences, 97.5)
            
            # Degrees of freedom
            df = len(data1) + len(data2) - 2
            
        elif test_type == 'wilcoxon':
            # Wilcoxon signed-rank test
            statistic, p_value = stats.wilcoxon(data1, data2)
            test_name = "Wilcoxon signed-rank test"
            test_description = "Non-parametric test for paired samples to determine if the differences come from a distribution with zero median"
            null_hypothesis = "The differences between paired samples have zero median"
            
            # Compute confidence interval for median difference
            differences = data1 - data2
            ci_lower = np.percentile(differences, 2.5)
            ci_upper = np.percentile(differences, 97.5)
            
            # Degrees of freedom
            df = len(differences) - 1
            
        elif test_type == 'ks_2samp':
            # Two-sample Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(data1, data2)
            test_name = "Two-sample Kolmogorov-Smirnov test"
            test_description = "Non-parametric test to determine if two independent samples are drawn from the same continuous distribution"
            null_hypothesis = "The two samples come from the same distribution"
            
            # For KS test, CI is typically not provided, but we can report the statistic range
            ci_lower = 0
            ci_upper = 1
            
            # Degrees of freedom
            df = len(data1) + len(data2) - 2
            
        elif test_type == 't_test':
            # Independent t-test
            statistic, p_value = stats.ttest_ind(data1, data2, equal_var=True)
            test_name = "Independent t-test"
            test_description = "Parametric test to determine if two independent samples have different means"
            null_hypothesis = "The means of the two samples are equal"
            
            # Compute confidence interval for mean difference
            mean_diff = np.mean(data1) - np.mean(data2)
            
            # Pooled standard deviation
            n1, n2 = len(data1), len(data2)
            s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
            pooled_var = ((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2)
            pooled_std = np.sqrt(pooled_var)
            
            # Standard error of difference between means
            se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
            
            # Critical t-value
            df = n1 + n2 - 2
            t_crit = stats.t.ppf(1 - alpha/2, df)
            
            # Confidence interval
            ci_lower = mean_diff - t_crit * se_diff
            ci_upper = mean_diff + t_crit * se_diff
            
        elif test_type == 'welch_t_test':
            # Welch's t-test
            statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False)
            test_name = "Welch's t-test"
            test_description = "Parametric test to determine if two independent samples have different means (not assuming equal variances)"
            null_hypothesis = "The means of the two samples are equal"
            
            # Compute confidence interval for mean difference
            mean_diff = np.mean(data1) - np.mean(data2)
            
            # Standard deviation and sample size
            n1, n2 = len(data1), len(data2)
            s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
            
            # Standard error of difference between means
            se_diff = np.sqrt(s1/n1 + s2/n2)
            
            # Welch-Satterthwaite degrees of freedom
            df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
            
            # Critical t-value
            t_crit = stats.t.ppf(1 - alpha/2, df)
            
            # Confidence interval
            ci_lower = mean_diff - t_crit * se_diff
            ci_upper = mean_diff + t_crit * se_diff
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Determine if null hypothesis should be rejected
        reject_null = p_value < alpha
        
        # Calculate effect size
        if test_type == 'mann_whitney':
            # Calculate rank-biserial correlation (effect size for Mann-Whitney U)
            n1, n2 = len(data1), len(data2)
            u = statistic
            # Formula for rank-biserial correlation
            effect_size = 1 - (2 * u) / (n1 * n2)
            effect_size_name = "Rank-biserial correlation"
        elif test_type == 'wilcoxon':
            # Calculate r (effect size for Wilcoxon)
            n = len(data1)
            effect_size = statistic / (n * (n + 1) / 2)
            effect_size_name = "r (effect size)"
        elif test_type == 'ks_2samp':
            # KS statistic is already an effect size measure
            effect_size = statistic
            effect_size_name = "D statistic"
        elif test_type in ['t_test', 'welch_t_test']:
            # Calculate Cohen's d (effect size for t-test)
            mean_diff = np.mean(data1) - np.mean(data2)
            pooled_std = np.sqrt((np.std(data1, ddof=1)**2 + np.std(data2, ddof=1)**2) / 2)
            effect_size = mean_diff / pooled_std
            effect_size_name = "Cohen's d"
        
        # Interpret effect size
        if abs(effect_size) < 0.3:
            effect_interpretation = "Small effect"
        elif abs(effect_size) < 0.5:
            effect_interpretation = "Medium effect"
        else:
            effect_interpretation = "Large effect"
        
        # Create scientific report format
        if test_type == 'mann_whitney':
            report = f"Mann-Whitney U({df}) = {statistic:.2f}, p = {p_value:.3f}, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}]"
            if p_value < 0.001:
                report = f"Mann-Whitney U({df}) = {statistic:.2f}, p < 0.001, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}]"
        elif test_type == 'wilcoxon':
            report = f"Wilcoxon W({df}) = {statistic:.2f}, p = {p_value:.3f}, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}]"
            if p_value < 0.001:
                report = f"Wilcoxon W({df}) = {statistic:.2f}, p < 0.001, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}]"
        elif test_type == 'ks_2samp':
            report = f"Kolmogorov-Smirnov D({df}) = {statistic:.2f}, p = {p_value:.3f}"
            if p_value < 0.001:
                report = f"Kolmogorov-Smirnov D({df}) = {statistic:.2f}, p < 0.001"
        elif test_type in ['t_test', 'welch_t_test']:
            if test_type == 't_test':
                test_name_display = "t"
            else:
                test_name_display = "Welch's t"
            report = f"{test_name_display}({df:.1f}) = {statistic:.2f}, p = {p_value:.3f}, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}]"
            if p_value < 0.001:
                report = f"{test_name_display}({df:.1f}) = {statistic:.2f}, p < 0.001, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}]"
        
        # Interpret the effect
        if reject_null:
            if test_type in ['t_test', 'welch_t_test']:
                mean_diff = np.mean(data1) - np.mean(data2)
                direction = "higher" if mean_diff > 0 else "lower"
                interpretation = f"The results demonstrate a statistically significant difference between the two datasets. The first dataset shows {direction} values than the second dataset ({report}). {effect_interpretation} observed ({effect_size_name} = {effect_size:.2f})."
            else:
                interpretation = f"The results demonstrate a statistically significant difference between the two datasets ({report}). {effect_interpretation} observed ({effect_size_name} = {effect_size:.2f})."
        else:
            interpretation = f"The results do not demonstrate a statistically significant difference between the two datasets ({report}). {effect_interpretation} observed ({effect_size_name} = {effect_size:.2f})."
        
        # Perform post-hoc analysis if the result is significant
        post_hoc_results = None
        if reject_null:
            try:
                post_hoc_results = HexoskinWavLoader.perform_post_hoc_analysis(data1, data2, test_type)
            except Exception as e:
                post_hoc_results = {"error": str(e)}
            
        # Return results
        return {
            'test_name': test_name,
            'test_description': test_description,
            'null_hypothesis': null_hypothesis,
            'statistic': statistic,
            'p_value': p_value,
            'alpha': alpha,
            'reject_null': reject_null,
            'effect_size': effect_size,
            'effect_size_name': effect_size_name,
            'effect_interpretation': effect_interpretation,
            'descriptive_stats_1': desc1,
            'descriptive_stats_2': desc2,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'df': df,
            'scientific_report': report,
            'interpretation': interpretation,
            'test_switch_message': test_switch_message,
            'post_hoc_results': post_hoc_results
        }
    
    def test_normality(self):
        """
        Test the normality of the data distribution using multiple methods
        
        Returns:
            dict: Results of normality tests with interpretations
        """
        if self.data is None:
            print("No data loaded")
            return None
        
        # For large datasets, sample to speed up calculations
        if len(self.data) > 5000:
            sample_size = 5000
            data_sample = np.random.choice(self.data['value'], size=sample_size, replace=False)
        else:
            data_sample = self.data['value']
            
        results = {}
        
        # 1. Shapiro-Wilk test (best for sample sizes < 5000)
        try:
            shapiro_test = stats.shapiro(data_sample)
            results['shapiro_wilk'] = {
                'statistic': shapiro_test[0],
                'p_value': shapiro_test[1],
                'normal': shapiro_test[1] > 0.05,
                'description': "The Shapiro-Wilk test tests the null hypothesis that the data comes from a normal distribution."
            }
        except Exception as e:
            results['shapiro_wilk'] = {'error': str(e)}
        
        # 2. D'Agostino's K-squared test
        try:
            dagostino_test = stats.normaltest(data_sample)
            results['dagostino_k2'] = {
                'statistic': dagostino_test[0],
                'p_value': dagostino_test[1],
                'normal': dagostino_test[1] > 0.05,
                'description': "D'Agostino's K-squared test tests if skewness and kurtosis are compatible with a normal distribution."
            }
        except Exception as e:
            results['dagostino_k2'] = {'error': str(e)}
        
        # 3. Kolmogorov-Smirnov test against normal distribution
        try:
            ks_test = stats.kstest(
                data_sample, 
                'norm', 
                args=(np.mean(data_sample), np.std(data_sample, ddof=1))
            )
            results['kolmogorov_smirnov'] = {
                'statistic': ks_test[0],
                'p_value': ks_test[1],
                'normal': ks_test[1] > 0.05,
                'description': "The Kolmogorov-Smirnov test compares the empirical distribution with a normal distribution."
            }
        except Exception as e:
            results['kolmogorov_smirnov'] = {'error': str(e)}
        
        # 4. Anderson-Darling test
        try:
            ad_test = stats.anderson(data_sample, 'norm')
            # Anderson-Darling test returns critical values at different significance levels
            # We'll use 5% significance level (index 2 in the critical_values array)
            is_normal = ad_test.statistic < ad_test.critical_values[2]
            results['anderson_darling'] = {
                'statistic': ad_test.statistic,
                'critical_values': {
                    '15%': ad_test.critical_values[0],
                    '10%': ad_test.critical_values[1],
                    '5%': ad_test.critical_values[2],
                    '2.5%': ad_test.critical_values[3],
                    '1%': ad_test.critical_values[4]
                },
                'normal': is_normal,
                'description': "The Anderson-Darling test is more sensitive to deviations in the tails of the distribution."
            }
        except Exception as e:
            results['anderson_darling'] = {'error': str(e)}
        
        # 5. Jarque-Bera test
        try:
            # Calculate skewness and kurtosis for JB test
            skewness = stats.skew(data_sample)
            kurtosis = stats.kurtosis(data_sample)
            n = len(data_sample)
            jb_stat = (n / 6) * (skewness**2 + (kurtosis**2) / 4)
            jb_p = 1 - stats.chi2.cdf(jb_stat, df=2)
            
            results['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_p,
                'normal': jb_p > 0.05,
                'description': "The Jarque-Bera test is based on sample skewness and kurtosis."
            }
        except Exception as e:
            results['jarque_bera'] = {'error': str(e)}
        
        # 6. Calculate skewness and kurtosis with confidence intervals
        try:
            skewness = stats.skew(data_sample)
            kurtosis = stats.kurtosis(data_sample)
            
            # Standard errors
            se_skewness = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
            se_kurtosis = 2 * se_skewness * np.sqrt((n**2 - 1) / ((n - 3) * (n + 5)))
            
            # 95% confidence intervals
            skew_ci_lower = skewness - 1.96 * se_skewness
            skew_ci_upper = skewness + 1.96 * se_skewness
            kurt_ci_lower = kurtosis - 1.96 * se_kurtosis
            kurt_ci_upper = kurtosis + 1.96 * se_kurtosis
            
            # Interpret skewness
            if abs(skewness / se_skewness) < 2:
                skew_interpretation = "approximately symmetric"
            elif skewness < 0:
                skew_interpretation = "negatively skewed (left-tailed)"
            else:
                skew_interpretation = "positively skewed (right-tailed)"
                
            # Interpret kurtosis
            if abs(kurtosis / se_kurtosis) < 2:
                kurt_interpretation = "approximately mesokurtic (normal)"
            elif kurtosis < 0:
                kurt_interpretation = "platykurtic (flatter than normal)"
            else:
                kurt_interpretation = "leptokurtic (more peaked than normal)"
            
            results['distribution_shape'] = {
                'skewness': skewness,
                'skewness_se': se_skewness,
                'skewness_ci': [skew_ci_lower, skew_ci_upper],
                'skewness_interpretation': skew_interpretation,
                'kurtosis': kurtosis,
                'kurtosis_se': se_kurtosis,
                'kurtosis_ci': [kurt_ci_lower, kurt_ci_upper],
                'kurtosis_interpretation': kurt_interpretation,
                'description': "Skewness and kurtosis measure the shape of the probability distribution."
            }
        except Exception as e:
            results['distribution_shape'] = {'error': str(e)}
        
        # 7. Quantile-Quantile plot data
        try:
            # Generate QQ-plot data points
            from scipy.stats import probplot
            qq_data = probplot(data_sample, dist='norm', plot=None)
            
            # Extract the QQ plot points (theoretical and observed)
            theoretical_quantiles = qq_data[0][0]
            observed_quantiles = qq_data[0][1]
            
            # Store only a subset of points for visualization (max 200 points)
            max_points = 200
            if len(theoretical_quantiles) > max_points:
                step = len(theoretical_quantiles) // max_points
                theoretical_quantiles = theoretical_quantiles[::step]
                observed_quantiles = observed_quantiles[::step]
            
            results['qq_plot_data'] = {
                'theoretical_quantiles': theoretical_quantiles.tolist(),
                'observed_quantiles': observed_quantiles.tolist(),
                'slope': qq_data[1][0],
                'intercept': qq_data[1][1],
                'description': "Quantile-Quantile plot compares the distribution of the data to a normal distribution."
            }
        except Exception as e:
            results['qq_plot_data'] = {'error': str(e)}
        
        # 8. Summary assessment
        try:
            # Count how many tests suggest normal distribution
            normality_tests = ['shapiro_wilk', 'dagostino_k2', 'kolmogorov_smirnov', 'anderson_darling', 'jarque_bera']
            normal_count = sum(1 for test in normality_tests if test in results and results[test].get('normal', False))
            total_tests = sum(1 for test in normality_tests if test in results and 'error' not in results[test])
            
            # Overall assessment
            if total_tests == 0:
                overall_assessment = "Could not determine normality due to errors in all tests."
            else:
                normal_ratio = normal_count / total_tests
                if normal_ratio >= 0.8:
                    overall_assessment = "Strong evidence for normal distribution ({}% of tests)".format(int(normal_ratio * 100))
                elif normal_ratio >= 0.5:
                    overall_assessment = "Moderate evidence for normal distribution ({}% of tests)".format(int(normal_ratio * 100))
                elif normal_ratio >= 0.2:
                    overall_assessment = "Weak evidence for normal distribution ({}% of tests)".format(int(normal_ratio * 100))
                else:
                    overall_assessment = "Strong evidence against normal distribution ({}% of tests)".format(100 - int(normal_ratio * 100))
            
            results['overall_assessment'] = {
                'normal_test_count': normal_count,
                'total_test_count': total_tests,
                'assessment': overall_assessment,
                'recommendation': self._get_normality_recommendation(results)
            }
        except Exception as e:
            results['overall_assessment'] = {'error': str(e)}
        
        return results
    
    def _get_normality_recommendation(self, normality_results):
        """Generate recommendations based on normality test results"""
        if not normality_results:
            return "No recommendations available due to missing test results."
        
        recommendations = []
        
        # Check overall normality assessment
        overall = normality_results.get('overall_assessment', {})
        normal_count = overall.get('normal_test_count', 0)
        total_count = overall.get('total_test_count', 0)
        
        if total_count > 0:
            normal_ratio = normal_count / total_count
            
            # Recommendations based on normality
            if normal_ratio < 0.5:
                recommendations.append("Consider using non-parametric statistical tests for your analysis.")
                recommendations.append("Data transformations might help achieve normality. Options include: log, square root, or Box-Cox transformations.")
                
                # Check skewness
                if 'distribution_shape' in normality_results:
                    skewness = normality_results['distribution_shape'].get('skewness', 0)
                    if skewness > 1:
                        recommendations.append("Data shows positive skew. Consider log or inverse transformations.")
                    elif skewness < -1:
                        recommendations.append("Data shows negative skew. Consider square or cube transformations.")
            else:
                recommendations.append("Parametric tests are appropriate for this dataset.")
        
        # Check sample size
        sample_size = len(self.data) if self.data is not None else 0
        if sample_size > 5000:
            recommendations.append("With large sample sizes (>5000), statistical tests often detect minor deviations from normality. Visual inspection of QQ-plots may be more informative.")
        
        # Add general recommendation if none were generated
        if not recommendations:
            recommendations.append("The data appears to follow a normal distribution. Standard parametric methods can be applied.")
        
        return recommendations
    
    def resample_data(self, target_hz):
        """
        Resample the data to a new frequency
        
        Args:
            target_hz (int): Target frequency in Hz
        """
        if self.data is None:
            print("No data loaded")
            return
        
        # Calculate number of samples for resampling
        n_samples = int(len(self.timestamps) * target_hz / self.sample_rate)
        
        # Resample using scipy's resample function
        resampled_values = signal.resample(self.values, n_samples)
        
        # Create new timestamp array
        new_duration = self.timestamps[-1]  # Same duration as original
        resampled_timestamps = np.linspace(0, new_duration, n_samples)
        
        # Update data
        self.data = pd.DataFrame({
            'timestamp': resampled_timestamps,
            'value': resampled_values
        })
        
        print(f"Resampled data from {self.sample_rate}Hz to {target_hz}Hz")
        
    def filter_data(self, lowcut=None, highcut=None, order=5):
        """
        Apply a bandpass filter to the data
        
        Args:
            lowcut (float): Low cutoff frequency
            highcut (float): High cutoff frequency
            order (int): Filter order
        """
        if self.data is None:
            print("No data loaded")
            return
        
        nyquist = 0.5 * self.sample_rate
        
        if lowcut and highcut:
            # Bandpass filter
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = signal.butter(order, [low, high], btype='band')
        elif lowcut:
            # Highpass filter
            low = lowcut / nyquist
            b, a = signal.butter(order, low, btype='high')
        elif highcut:
            # Lowpass filter
            high = highcut / nyquist
            b, a = signal.butter(order, high, btype='low')
        else:
            print("No filter parameters specified")
            return
        
        # Apply filter
        filtered_values = signal.filtfilt(b, a, self.values)
        
        # Update data
        self.data['value'] = filtered_values
        
        if lowcut and highcut:
            print(f"Applied bandpass filter ({lowcut}-{highcut} Hz)")
        elif lowcut:
            print(f"Applied highpass filter ({lowcut} Hz)")
        elif highcut:
            print(f"Applied lowpass filter ({highcut} Hz)")
    
    def plot_data(self, ax=None, **plot_kwargs):
        """
        Plot the data using matplotlib
        
        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            plot_kwargs: Additional keyword arguments for the plot
            
        Returns:
            matplotlib.axes.Axes: The axes with the plot
        """
        if self.data is None:
            print("No data loaded")
            return None
        
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        # Extract title from kwargs if it exists (it's not a valid plot parameter)
        custom_title = None
        if 'title' in plot_kwargs:
            custom_title = plot_kwargs.pop('title')
        
        # Set default plot options
        plot_options = {
            'color': 'blue',
            'linewidth': 1.0,
            'alpha': 0.8,
            'label': self.metadata.get('sensor_name', 'Data'),
        }
        
        # Update with user-provided options
        plot_options.update(plot_kwargs)
        
        # Plot data with specified options
        ax.plot(self.data['timestamp'], self.data['value'], **plot_options)
        
        # Set title
        if custom_title:
            ax.set_title(custom_title)
        else:
            sensor_name = self.metadata.get('sensor_name', 'Unknown')
            ax.set_title(f"{sensor_name} Data")
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Value')
        ax.grid(True)
        ax.legend()
        
        return ax
    
    def save_to_csv(self, output_path=None):
        """
        Save the data to a CSV file
        
        Args:
            output_path (str, optional): Path to save the CSV file. 
                                         If None, uses the original filename with .csv extension
                                         
        Returns:
            str: Path to the saved file
        """
        if self.data is None:
            print("No data loaded")
            return None
        
        if output_path is None:
            # Use the original filename with .csv extension
            base_name = os.path.splitext(self.file_path)[0]
            output_path = f"{base_name}.csv"
        
        self.data.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        
        return output_path

    @staticmethod
    def perform_post_hoc_analysis(dataset1, dataset2, test_type):
        """
        Perform post-hoc analysis when significant differences are found between datasets.
        
        Args:
            dataset1 (pd.DataFrame or np.ndarray): First dataset
            dataset2 (pd.DataFrame or np.ndarray): Second dataset
            test_type (str): Type of test used for the initial comparison
            
        Returns:
            dict: Post-hoc analysis results
        """
        # Ensure datasets are numpy arrays
        if isinstance(dataset1, pd.DataFrame):
            if 'value' in dataset1.columns:
                data1 = dataset1['value'].values
            else:
                data1 = dataset1.iloc[:, 0].values
        elif isinstance(dataset1, (list, tuple)):
            data1 = np.array(dataset1)
        else:
            data1 = dataset1
            
        if isinstance(dataset2, pd.DataFrame):
            if 'value' in dataset2.columns:
                data2 = dataset2['value'].values
            else:
                data2 = dataset2.iloc[:, 0].values
        elif isinstance(dataset2, (list, tuple)):
            data2 = np.array(dataset2)
        else:
            data2 = dataset2
        
        results = {}
        
        # Check for normality - important for deciding which further tests to recommend
        try:
            shapiro1 = stats.shapiro(data1[:5000] if len(data1) > 5000 else data1)
            shapiro2 = stats.shapiro(data2[:5000] if len(data2) > 5000 else data2)
            
            results['normality'] = {
                'dataset1': {
                    'shapiro_stat': shapiro1[0],
                    'shapiro_p': shapiro1[1],
                    'is_normal': shapiro1[1] > 0.05
                },
                'dataset2': {
                    'shapiro_stat': shapiro2[0],
                    'shapiro_p': shapiro2[1],
                    'is_normal': shapiro2[1] > 0.05
                }
            }
        except Exception:
            # Sometimes shapiro can fail for certain distributions
            results['normality'] = {
                'dataset1': {'is_normal': False},
                'dataset2': {'is_normal': False}
            }
        
        # Levene's test for homogeneity of variance (important for parametric tests)
        try:
            levene_test = stats.levene(data1, data2)
            results['homogeneity'] = {
                'levene_stat': levene_test[0],
                'levene_p': levene_test[1],
                'equal_variance': levene_test[1] > 0.05
            }
        except Exception:
            results['homogeneity'] = {'equal_variance': False}
        
        # Perform bootstrap analysis to get confidence intervals for the difference
        n_bootstrap = 1000
        bootstrap_diffs = []
        
        # Use random samples of the same size for each bootstrap iteration
        min_size = min(len(data1), len(data2))
        for _ in range(n_bootstrap):
            sample1 = np.random.choice(data1, size=min_size, replace=True)
            sample2 = np.random.choice(data2, size=min_size, replace=True)
            
            if 't_test' in test_type or 'welch' in test_type:
                # For parametric tests, look at mean difference
                diff = np.mean(sample1) - np.mean(sample2)
            else:
                # For non-parametric tests, look at median difference
                diff = np.median(sample1) - np.median(sample2)
                
            bootstrap_diffs.append(diff)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        results['bootstrap'] = {
            'diff_ci_lower': ci_lower,
            'diff_ci_upper': ci_upper,
            'bootstrap_samples': n_bootstrap
        }
        
        # Calculate additional effect sizes
        # Common Language Effect Size (probability of superiority)
        cles = 0
        for x in data1:
            for y in data2:
                if x > y:
                    cles += 1
                elif x == y:
                    cles += 0.5
        cles = cles / (len(data1) * len(data2))
        
        results['additional_effect_sizes'] = {
            'cles': cles,
            'cles_interpretation': "Probability that a randomly selected value from dataset 1 exceeds a randomly selected value from dataset 2"
        }
        
        # Calculate recommended follow-up tests based on the data characteristics
        recommendations = []
        
        both_normal = results['normality']['dataset1'].get('is_normal', False) and results['normality']['dataset2'].get('is_normal', False)
        equal_var = results['homogeneity'].get('equal_variance', False)
        
        if both_normal and equal_var:
            if 'mann_whitney' in test_type or 'ks_2samp' in test_type:
                recommendations.append("Consider using an Independent t-test since both datasets appear normally distributed with equal variances.")
        elif both_normal and not equal_var:
            if 'mann_whitney' in test_type or 'ks_2samp' in test_type or 't_test' in test_type:
                recommendations.append("Consider using Welch's t-test since both datasets appear normally distributed but with unequal variances.")
        elif not both_normal:
            if 't_test' in test_type or 'welch_t_test' in test_type:
                recommendations.append("Consider using a non-parametric test (Mann-Whitney U or KS test) since one or both datasets do not appear normally distributed.")
        
        # Additional specific recommendations
        if len(data1) == len(data2) and 'mann_whitney' in test_type:
            recommendations.append("For datasets of equal length, consider using the Wilcoxon signed-rank test if the samples are paired.")
            
        if np.abs(np.mean(data1) - np.mean(data2)) > 3 * (np.std(data1) + np.std(data2)):
            recommendations.append("The difference between datasets is very large. Consider checking for outliers that might be skewing the results.")
        
        results['recommendations'] = recommendations
        
        return results

    @staticmethod
    def compare_multiple_datasets(datasets, test_type='anova'):
        """
        Perform statistical comparison between multiple datasets (up to 15).
        
        Args:
            datasets (list): List of dictionaries with dataset information:
                             {'name': str, 'data': pd.DataFrame, 'start_date': str}
            test_type (str): Type of statistical test to perform:
                - 'anova': One-way ANOVA for normally distributed data (default)
                - 'kruskal': Kruskal-Wallis H-test for non-parametric data
                - 'friedman': Friedman test for paired non-parametric data
                - 'welch_anova': Welch's ANOVA for normally distributed data with unequal variances
                - 'rm_anova': Repeated measures ANOVA for normally distributed paired data
                - 'aligned_ranks': Aligned Ranks Transform ANOVA for non-normal data in factorial designs
                
        Returns:
            dict: Results of the statistical test
        """
        # Validate number of datasets
        if not datasets:
            return {"error": "No datasets provided"}
            
        if len(datasets) > 15:
            return {"error": "Maximum 15 datasets can be compared at once"}
            
        if len(datasets) < 2:
            return {"error": "At least 2 datasets are required for comparison"}
            
        # Initialize post-hoc results to None
        post_hoc_results = None
        
        # Extract data values from datasets
        data_values = []
        data_names = []
        data_counts = []
        data_means = []
        data_medians = []
        data_stds = []
        data_mins = []
        data_maxs = []
        
        for dataset_info in datasets:
            dataset = dataset_info['data']
            name = dataset_info['name']
            data_names.append(name)
            
            # Extract values
            if isinstance(dataset, pd.DataFrame):
                if 'value' in dataset.columns:
                    values = dataset['value'].values
                else:
                    values = dataset.iloc[:, 0].values
            elif isinstance(dataset, (list, tuple)):
                values = np.array(dataset)
            else:
                values = dataset
                
            data_values.append(values)
            
            # Calculate basic statistics
            data_counts.append(len(values))
            data_means.append(np.mean(values))
            data_medians.append(np.median(values))
            data_stds.append(np.std(values))
            data_mins.append(np.min(values))
            data_maxs.append(np.max(values))
        
        # Determine if we need to use a non-parametric test
        # Check normality for each dataset
        normality_results = []
        all_normal = True
        
        for i, values in enumerate(data_values):
            try:
                # Use Shapiro-Wilk test for normality if dataset is not too large
                if len(values) <= 5000:
                    stat, p = stats.shapiro(values)
                    is_normal = p > 0.05
                else:
                    # For large datasets, use D'Agostino and Pearson's test
                    stat, p = stats.normaltest(values)
                    is_normal = p > 0.05
                
                all_normal = all_normal and is_normal
                
                normality_results.append({
                    'dataset': data_names[i],
                    'statistic': stat,
                    'p_value': p,
                    'is_normal': is_normal
                })
            except Exception as e:
                all_normal = False
                normality_results.append({
                    'dataset': data_names[i],
                    'error': str(e),
                    'is_normal': False
                })
        
        # Check homogeneity of variance using Levene's test
        try:
            levene_stat, levene_p = stats.levene(*data_values)
            equal_variance = levene_p > 0.05
        except Exception:
            equal_variance = False
            
        # Check equal sample sizes
        equal_sample_sizes = len(set(data_counts)) == 1
        
        # Auto-select appropriate test if needed
        original_test_type = test_type
        if test_type == 'auto':
            if all_normal:
                if equal_variance:
                    test_type = 'anova'
                else:
                    test_type = 'welch_anova'
            else:
                if equal_sample_sizes:
                    test_type = 'friedman'
                else:
                    test_type = 'kruskal'
                    
            print(f"Auto-selected test: {test_type}")
        
        # Perform appropriate statistical test
        posthoc_results = None
        
        if test_type == 'anova':
            # One-way ANOVA
            try:
                f_stat, p_value = stats.f_oneway(*data_values)
                test_name = "One-way ANOVA"
                test_description = "Test for equality of means across multiple groups"
                null_hypothesis = "All group means are equal"
                
                # Calculate effect size (Eta-squared)
                total_ss = sum(np.var(values) * (len(values) - 1) for values in data_values)
                between_ss = sum(len(values) * (np.mean(values) - np.mean(np.concatenate(data_values)))**2 for values in data_values)
                effect_size = between_ss / total_ss if total_ss > 0 else 0
                effect_size_name = "Eta-squared"
                
                # Automatically run post-hoc tests if result is significant
                if p_value < 0.05:
                    post_hoc_results = HexoskinWavLoader._run_posthoc_tests(
                        data_values, data_names, 'anova', alpha=0.05
                    )
                    
            except Exception as e:
                return {"error": f"ANOVA failed: {str(e)}"}
                
        elif test_type == 'welch_anova':
            # Welch's ANOVA (for unequal variances)
            try:
                # Calculate group statistics
                group_stats = []
                for values in data_values:
                    n = len(values)
                    mean = np.mean(values)
                    var = np.var(values, ddof=1)
                    group_stats.append((n, mean, var))
                
                # Calculate weights
                weights = [n/var for n, _, var in group_stats]
                
                # Calculate weighted mean
                weighted_sum = sum(w * mean for w, (_, mean, _) in zip(weights, group_stats))
                total_weight = sum(weights)
                weighted_mean = weighted_sum / total_weight if total_weight > 0 else 0
                
                # Calculate F statistic
                numerator = sum(w * (mean - weighted_mean)**2 for w, (_, mean, _) in zip(weights, group_stats))
                k = len(data_values)  # number of groups
                
                # Calculate degrees of freedom adjustment
                lambda_vals = []
                for n, _, var in group_stats:
                    weight = n / var
                    lambda_vals.append((1 - (weight / total_weight))**2 / (n - 1))
                
                denominator_df = 1 / (3 * sum(lambda_vals) / (k**2 - 1))
                
                # Welch's F statistic
                f_stat = numerator / (k - 1)
                
                # P-value
                p_value = 1 - stats.f.cdf(f_stat, k - 1, denominator_df)
                
                test_name = "Welch's ANOVA"
                test_description = "Test for equality of means across multiple groups with unequal variances"
                null_hypothesis = "All group means are equal"
                
                # Calculate approximate effect size
                effect_size = f_stat / (f_stat + denominator_df)
                effect_size_name = "Approximate omega-squared"
                
                # Post-hoc tests
                if p_value < 0.05:
                    post_hoc_results = HexoskinWavLoader._run_posthoc_tests(
                        data_values, data_names, 'welch_anova', alpha=0.05
                    )
                    
            except Exception as e:
                return {"error": f"Welch's ANOVA failed: {str(e)}"}
                
        elif test_type == 'kruskal':
            # Kruskal-Wallis H-test
            try:
                h_stat, p_value = stats.kruskal(*data_values)
                test_name = "Kruskal-Wallis H-test"
                test_description = "Non-parametric test for equality of medians across multiple groups"
                null_hypothesis = "All group distributions are equal"
                
                # Calculate epsilon-squared effect size
                n = sum(data_counts)
                effect_size = (h_stat - (len(data_values) - 1)) / (n - len(data_values))
                effect_size_name = "Epsilon-squared"
                
                # Post-hoc tests
                if p_value < 0.05:
                    post_hoc_results = HexoskinWavLoader._run_posthoc_tests(
                        data_values, data_names, 'kruskal', alpha=0.05
                    )
                    
            except Exception as e:
                return {"error": f"Kruskal-Wallis test failed: {str(e)}"}
                
        elif test_type == 'friedman':
            # Check if all datasets have the same length
            if not equal_sample_sizes:
                return {"error": "Friedman test requires all datasets to have the same length"}
            
            try:
                # Friedman test
                # Reshape data for Friedman test
                k = len(data_values)  # number of groups
                n = data_counts[0]    # number of samples per group (they're all equal)
                
                # Create a matrix where each column is a group and each row is a subject
                data_matrix = np.zeros((n, k))
                for i in range(k):
                    data_matrix[:, i] = data_values[i][:n]
                
                chi2_stat, p_value = stats.friedmanchisquare(*[data_matrix[:, i] for i in range(k)])
                test_name = "Friedman Test"
                test_description = "Non-parametric test for repeated measures designs"
                null_hypothesis = "All treatment effects are equal"
                
                # Calculate Kendall's W (effect size)
                effect_size = chi2_stat / (n * (k - 1))
                effect_size_name = "Kendall's W"
                
                # Post-hoc tests
                if p_value < 0.05:
                    post_hoc_results = HexoskinWavLoader._run_posthoc_tests(
                        data_values, data_names, 'friedman', alpha=0.05, n_blocks=n
                    )
                    
            except Exception as e:
                return {"error": f"Friedman test failed: {str(e)}"}
                
        elif test_type == 'rm_anova':
            # Repeated measures ANOVA
            if not equal_sample_sizes:
                return {"error": "Repeated measures ANOVA requires all datasets to have the same length"}
                
            try:
                # Prepare data
                k = len(data_values)  # number of groups/treatments
                n = data_counts[0]    # number of subjects
                
                # Create a matrix where each column is a group and each row is a subject
                data_matrix = np.zeros((n, k))
                for i in range(k):
                    data_matrix[:, i] = data_values[i][:n]
                
                # Calculate grand mean
                grand_mean = np.mean(data_matrix)
                
                # Calculate subject means
                subject_means = np.mean(data_matrix, axis=1)
                
                # Calculate treatment means
                treatment_means = np.mean(data_matrix, axis=0)
                
                # Calculate SS Total
                ss_total = np.sum((data_matrix - grand_mean)**2)
                
                # Calculate SS Subjects (between subjects)
                ss_subjects = np.sum((subject_means - grand_mean)**2) * k
                
                # Calculate SS Treatments (between treatments)
                ss_treatments = np.sum((treatment_means - grand_mean)**2) * n
                
                # Calculate SS Error
                ss_error = ss_total - ss_subjects - ss_treatments
                
                # Calculate degrees of freedom
                df_subjects = n - 1
                df_treatments = k - 1
                df_error = (n - 1) * (k - 1)
                df_total = n * k - 1
                
                # Calculate mean squares
                ms_treatments = ss_treatments / df_treatments
                ms_error = ss_error / df_error
                
                # Calculate F statistic
                f_stat = ms_treatments / ms_error
                
                # Calculate p-value
                p_value = 1 - stats.f.cdf(f_stat, df_treatments, df_error)
                
                test_name = "Repeated Measures ANOVA"
                test_description = "Parametric test for repeated measures designs"
                null_hypothesis = "All treatment effects are equal"
                
                # Calculate effect size (partial eta-squared)
                effect_size = ss_treatments / (ss_treatments + ss_error)
                effect_size_name = "Partial eta-squared"
                
                # Post-hoc tests
                if p_value < 0.05:
                    post_hoc_results = HexoskinWavLoader._run_posthoc_tests(
                        data_values, data_names, 'rm_anova', alpha=0.05
                    )
                    
            except Exception as e:
                return {"error": f"Repeated measures ANOVA failed: {str(e)}"}
                
        elif test_type == 'aligned_ranks':
            # Aligned Ranks Transform ANOVA (non-parametric alternative to RM-ANOVA)
            if not equal_sample_sizes:
                return {"error": "Aligned Ranks Transform ANOVA requires all datasets to have the same length"}
                
            try:
                # Prepare data
                k = len(data_values)  # number of groups/treatments
                n = data_counts[0]    # number of subjects
                
                # Create a matrix where each column is a group and each row is a subject
                data_matrix = np.zeros((n, k))
                for i in range(k):
                    data_matrix[:, i] = data_values[i][:n]
                
                # Calculate cell means
                cell_means = np.mean(data_matrix, axis=0)
                
                # Calculate subject means
                subject_means = np.mean(data_matrix, axis=1).reshape(-1, 1)
                
                # Calculate grand mean
                grand_mean = np.mean(data_matrix)
                
                # Calculate aligned data (subtract subject and condition effects, keep only interaction)
                aligned_data = data_matrix - subject_means - cell_means + grand_mean
                
                # Rank the aligned data (across all cells)
                flat_aligned = aligned_data.flatten()
                ranks = stats.rankdata(flat_aligned)
                ranked_data = ranks.reshape(n, k)
                
                # Run standard RM-ANOVA on the ranks
                # Calculate treatment mean ranks
                treatment_mean_ranks = np.mean(ranked_data, axis=0)
                
                # Calculate SS Treatments
                ss_treatments = np.sum((treatment_mean_ranks - np.mean(ranked_data))**2) * n
                
                # Calculate SS Total
                ss_total = np.sum((ranked_data - np.mean(ranked_data))**2)
                
                # Calculate SS Subjects
                subject_mean_ranks = np.mean(ranked_data, axis=1)
                ss_subjects = np.sum((subject_mean_ranks - np.mean(ranked_data))**2) * k
                
                # Calculate SS Error
                ss_error = ss_total - ss_subjects - ss_treatments
                
                # Calculate degrees of freedom
                df_treatments = k - 1
                df_error = (n - 1) * (k - 1)
                
                # Calculate mean squares
                ms_treatments = ss_treatments / df_treatments
                ms_error = ss_error / df_error
                
                # Calculate F statistic
                f_stat = ms_treatments / ms_error
                
                # Calculate p-value
                p_value = 1 - stats.f.cdf(f_stat, df_treatments, df_error)
                
                test_name = "Aligned Ranks Transform ANOVA"
                test_description = "Non-parametric test for repeated measures designs"
                null_hypothesis = "All treatment effects are equal"
                
                # Calculate effect size (partial eta-squared on ranks)
                effect_size = ss_treatments / (ss_treatments + ss_error)
                effect_size_name = "Partial eta-squared on ranks"
                
                # Post-hoc tests
                if p_value < 0.05:
                    post_hoc_results = HexoskinWavLoader._run_posthoc_tests(
                        data_values, data_names, 'aligned_ranks', alpha=0.05
                    )
                    
            except Exception as e:
                return {"error": f"Aligned Ranks Transform ANOVA failed: {str(e)}"}
                
        else:
            return {"error": f"Unknown test type: {test_type}"}
        
        # Determine if null hypothesis should be rejected
        alpha = 0.05
        reject_null = p_value < alpha
        
        # Interpret effect size
        if test_type in ['anova', 'welch_anova', 'rm_anova', 'aligned_ranks']:
            if effect_size < 0.01:
                effect_interpretation = "Negligible effect"
            elif effect_size < 0.06:
                effect_interpretation = "Small effect"
            elif effect_size < 0.14:
                effect_interpretation = "Medium effect"
            else:
                effect_interpretation = "Large effect"
        elif test_type == 'kruskal':
            if effect_size < 0.01:
                effect_interpretation = "Negligible effect"
            elif effect_size < 0.04:
                effect_interpretation = "Small effect"
            elif effect_size < 0.16:
                effect_interpretation = "Medium effect"
            else:
                effect_interpretation = "Large effect"
        elif test_type == 'friedman':
            if effect_size < 0.1:
                effect_interpretation = "Small effect"
            elif effect_size < 0.3:
                effect_interpretation = "Medium effect"
            else:
                effect_interpretation = "Large effect"
                
        # Generate summary interpretation
        if reject_null:
            if post_hoc_results and 'significant_pairs' in post_hoc_results and post_hoc_results['significant_pairs']:
                sig_pairs_text = ", ".join([f"{p[0]} vs {p[1]}" for p in post_hoc_results['significant_pairs'][:3]])
                if len(post_hoc_results['significant_pairs']) > 3:
                    sig_pairs_text += f", and {len(post_hoc_results['significant_pairs']) - 3} more"
                    
                interpretation = (f"There is a statistically significant difference among the datasets "
                                f"({test_name}, p = {p_value:.4f}, {effect_size_name} = {effect_size:.3f}). "
                                f"Post-hoc tests show significant differences between: {sig_pairs_text}.")
            else:
                interpretation = (f"There is a statistically significant difference among the datasets "
                                f"({test_name}, p = {p_value:.4f}, {effect_size_name} = {effect_size:.3f}). "
                                f"However, post-hoc tests did not identify specific significant pairs.")
        else:
            interpretation = (f"There is not a statistically significant difference among the datasets "
                            f"({test_name}, p = {p_value:.4f}, {effect_size_name} = {effect_size:.3f}).")
        
        # Return results
        return {
            'test_name': test_name,
            'test_description': test_description,
            'null_hypothesis': null_hypothesis,
            'statistic': f_stat if test_type in ['anova', 'welch_anova', 'rm_anova', 'aligned_ranks'] else 
                        h_stat if test_type == 'kruskal' else chi2_stat,
            'p_value': p_value,
            'alpha': alpha,
            'reject_null': reject_null,
            'effect_size': effect_size,
            'effect_size_name': effect_size_name,
            'effect_interpretation': effect_interpretation,
            'normality_results': normality_results,
            'equal_variance': equal_variance,
            'original_test_type': original_test_type,
            'test_type_switched': original_test_type != test_type,
            'descriptive_stats': {
                'names': data_names,
                'counts': data_counts,
                'means': data_means,
                'medians': data_medians,
                'stds': data_stds,
                'mins': data_mins,
                'maxs': data_maxs
            },
            'post_hoc_results': post_hoc_results,
            'interpretation': interpretation
        }

    @staticmethod
    def _run_posthoc_tests(data_values, data_names, test_type, alpha=0.05, n_blocks=None):
        """
        Run appropriate post-hoc tests based on the primary test type.
        
        Args:
            data_values (list): List of numpy arrays containing the data for each group
            data_names (list): Names of the groups
            test_type (str): Primary test that was significant
            alpha (float): Significance level
            n_blocks (int, optional): Number of blocks/subjects for repeated measures
            
        Returns:
            dict: Results of the post-hoc tests
        """
        if len(data_values) < 2:
            return None
            
        # Dictionary to store results
        posthoc_results = {}
        
        # ANOVA post-hoc: Tukey's HSD
        if test_type == 'anova':
            try:
                # Try to import tukey_hsd (SciPy 1.7+)
                try:
                    from scipy.stats import tukey_hsd
                    has_tukey_hsd = True
                except ImportError:
                    has_tukey_hsd = False
                
                if has_tukey_hsd:
                    # SciPy 1.7+ way - use the built-in tukey_hsd function
                    posthoc = tukey_hsd(*data_values)
                
                    posthoc_results = {
                        'test': 'Tukey HSD',
                        'description': 'Post-hoc test for pairwise comparisons after significant ANOVA',
                        'pairwise_p_values': [],
                        'significant_pairs': []
                    }
                
                    # Create pairwise comparisons
                    num_comparisons = len(data_values) * (len(data_values) - 1) // 2
                    
                    for i in range(len(data_values)):
                        for j in range(i+1, len(data_values)):
                            # Get rejection status from the results
                            p_val = float(posthoc.pvalue[i, j])
                        
                            # Add to pairwise results
                            posthoc_results['pairwise_p_values'].append({
                                'group1': data_names[i],
                                'group2': data_names[j],
                                'p_value': p_val,
                                'significant': p_val < alpha
                            })
                        
                            # Add to significant pairs if significant
                            if p_val < alpha:
                                posthoc_results['significant_pairs'].append(
                                    (data_names[i], data_names[j], p_val)
                                )
                else:
                    # For older SciPy versions, use pairwise t-tests with Bonferroni correction
                    posthoc_results = {
                        'test': 'Pairwise t-tests with Bonferroni correction',
                        'description': 'Post-hoc test for pairwise comparisons after significant ANOVA',
                        'pairwise_p_values': [],
                        'significant_pairs': []
                    }
                    
                    # Number of comparisons (for Bonferroni correction)
                    num_comparisons = len(data_values) * (len(data_values) - 1) // 2
                    
                    # Perform pairwise t-tests
                    for i in range(len(data_values)):
                        for j in range(i+1, len(data_values)):
                            # t-test between groups i and j
                            stat, p_val = stats.ttest_ind(data_values[i], data_values[j], equal_var=True)
                            
                            # Apply Bonferroni correction
                            p_val_adj = min(p_val * num_comparisons, 1.0)
                            
                            # Add to pairwise results
                            posthoc_results['pairwise_p_values'].append({
                                'group1': data_names[i],
                                'group2': data_names[j],
                                'p_value': p_val,
                                'p_value_adjusted': p_val_adj,
                                'significant': p_val_adj < alpha
                            })
                            
                            # Add to significant pairs if significant
                            if p_val_adj < alpha:
                                posthoc_results['significant_pairs'].append(
                                    (data_names[i], data_names[j], p_val_adj)
                            )
            except Exception as e:
                posthoc_results = {
                    'test': 'Tukey HSD',
                    'error': str(e)
                }
                
        # Welch's ANOVA post-hoc: Games-Howell
        elif test_type == 'welch_anova':
            try:
                posthoc_results = {
                    'test': 'Games-Howell',
                    'description': 'Post-hoc test for pairwise comparisons after Welch\'s ANOVA (for unequal variances)',
                    'pairwise_p_values': [],
                    'significant_pairs': []
                }
                
                # Calculate Games-Howell p-values manually
                for i in range(len(data_values)):
                    for j in range(i+1, len(data_values)):
                        # Group statistics
                        n1, n2 = len(data_values[i]), len(data_values[j])
                        mean1, mean2 = np.mean(data_values[i]), np.mean(data_values[j])
                        var1, var2 = np.var(data_values[i], ddof=1), np.var(data_values[j], ddof=1)
                        
                        # Mean difference
                        mean_diff = mean1 - mean2
                        
                        # Pooled variance
                        pooled_var = var1/n1 + var2/n2
                        
                        # Calculate degrees of freedom using Welch-Satterthwaite equation
                        df = (pooled_var**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
                        
                        # Calculate test statistic
                        t_stat = mean_diff / np.sqrt(pooled_var) if pooled_var > 0 else 0
                        
                        # Calculate p-value
                        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                        
                        # Calculate effect size (Hedges' g, which corrects for small sample bias)
                        # First calculate Cohen's d
                        pooled_std = np.sqrt((var1 + var2) / 2)
                        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                        
                        # Apply correction factor for Hedges' g
                        correction = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
                        hedges_g = cohens_d * correction
                        
                        posthoc_results['pairwise_p_values'].append({
                            'group1': data_names[i],
                            'group2': data_names[j],
                            'p_value': p_val,
                            'significant': p_val < alpha,
                            'mean_diff': mean_diff,
                            'effect_size': hedges_g,
                            'effect_size_type': "Hedges' g"
                        })
                        
                        if p_val < alpha:
                            posthoc_results['significant_pairs'].append(
                                (data_names[i], data_names[j], p_val)
                            )
            except Exception as e:
                posthoc_results = {
                    'test': 'Games-Howell',
                    'error': str(e)
                }
                
        # Kruskal-Wallis post-hoc: Dunn's test with Bonferroni correction
        elif test_type == 'kruskal':
            try:
                posthoc_results = {
                    'test': 'Dunn Test',
                    'description': 'Post-hoc test for pairwise comparisons after significant Kruskal-Wallis test',
                    'pairwise_p_values': [],
                    'significant_pairs': []
                }
                
                # Calculate mean ranks for each group
                all_data = np.concatenate(data_values)
                all_ranks = stats.rankdata(all_data)
                
                start_idx = 0
                mean_ranks = []
                for data in data_values:
                    end_idx = start_idx + len(data)
                    group_ranks = all_ranks[start_idx:end_idx]
                    mean_ranks.append(np.mean(group_ranks))
                    start_idx = end_idx
                
                # Calculate Bonferroni adjusted alpha
                num_comparisons = len(data_values) * (len(data_values) - 1) // 2
                adjusted_alpha = alpha / num_comparisons
                
                # Calculate pairwise comparisons
                for i in range(len(data_values)):
                    for j in range(i+1, len(data_values)):
                        # Calculate z-statistic and p-value
                        n_i, n_j = len(data_values[i]), len(data_values[j])
                        
                        # Calculate z-statistic
                        z = (mean_ranks[i] - mean_ranks[j]) / np.sqrt(
                            (len(all_data) * (len(all_data) + 1) / 12) * (1/n_i + 1/n_j)
                        )
                        
                        # Two-tailed test
                        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                        
                        # Apply Bonferroni correction
                        p_val_adj = min(p_val * num_comparisons, 1.0)
                        
                        # Calculate effect size (r = Z/sqrt(N))
                        effect_size = abs(z) / np.sqrt(n_i + n_j)
                        
                        posthoc_results['pairwise_p_values'].append({
                            'group1': data_names[i],
                            'group2': data_names[j],
                            'p_value': p_val,
                            'p_value_adjusted': p_val_adj,
                            'significant': p_val_adj < alpha,
                            'mean_rank_diff': mean_ranks[i] - mean_ranks[j],
                            'effect_size': effect_size,
                            'effect_size_type': 'r (rank biserial correlation)'
                        })
                        
                        if p_val_adj < alpha:
                            posthoc_results['significant_pairs'].append(
                                (data_names[i], data_names[j], p_val_adj)
                            )
            except Exception as e:
                posthoc_results = {
                    'test': 'Dunn Test',
                    'error': str(e)
                }
                
        # Friedman post-hoc: Nemenyi test
        elif test_type == 'friedman':
            try:
                if n_blocks is None:
                    raise ValueError("n_blocks parameter required for Friedman post-hoc tests")
                    
                posthoc_results = {
                    'test': 'Nemenyi Test',
                    'description': 'Post-hoc test for pairwise comparisons after significant Friedman test',
                    'pairwise_p_values': [],
                    'significant_pairs': []
                }
                
                # Reshape data for Friedman test
                k = len(data_values)  # number of groups
                n = n_blocks         # number of subjects/blocks
                
                # Create a matrix where each column is a group and each row is a subject
                data_matrix = np.zeros((n, k))
                for i in range(k):
                    data_matrix[:, i] = data_values[i][:n]
                
                # Calculate ranks within each row
                ranked_data = np.zeros_like(data_matrix)
                for i in range(n):
                    ranked_data[i, :] = stats.rankdata(data_matrix[i, :])
                
                # Calculate mean ranks for each group
                mean_ranks = np.mean(ranked_data, axis=0)
                
                # Critical value for Nemenyi test
                critical_diff = np.sqrt((k * (k + 1)) / (6 * n)) * stats.norm.ppf(1 - alpha / (k * (k - 1)))
                
                # Calculate pairwise comparisons
                for i in range(k):
                    for j in range(i+1, k):
                        rank_diff = abs(mean_ranks[i] - mean_ranks[j])
                        significant = rank_diff > critical_diff
                        
                        # Calculate approximate p-value
                        z = rank_diff / np.sqrt((k * (k + 1)) / (6 * n))
                        p_val = 2 * (1 - stats.norm.cdf(z))
                        
                        # Apply Bonferroni correction
                        num_comparisons = k * (k - 1) // 2
                        p_val_adj = min(p_val * num_comparisons, 1.0)
                        
                        # Calculate effect size (r = Z/sqrt(N))
                        effect_size = z / np.sqrt(2 * n)
                        
                        posthoc_results['pairwise_p_values'].append({
                            'group1': data_names[i],
                            'group2': data_names[j],
                            'p_value': p_val,
                            'p_value_adjusted': p_val_adj,
                            'significant': significant,
                            'mean_rank_diff': rank_diff,
                            'critical_diff': critical_diff,
                            'effect_size': effect_size,
                            'effect_size_type': 'r (correlation)'
                        })
                        
                        if significant:
                            posthoc_results['significant_pairs'].append(
                                (data_names[i], data_names[j], p_val_adj)
                            )
            except Exception as e:
                posthoc_results = {
                    'test': 'Nemenyi Test',
                    'error': str(e)
                }
                
        # RM-ANOVA post-hoc: Paired t-tests with Bonferroni correction
        elif test_type == 'rm_anova':
            try:
                if n_blocks is None:
                    # Get minimum length across all datasets
                    n_blocks = min(len(data) for data in data_values)
                
                posthoc_results = {
                    'test': 'Paired t-tests with Bonferroni correction',
                    'description': 'Post-hoc test for pairwise comparisons after significant RM-ANOVA',
                    'pairwise_p_values': [],
                    'significant_pairs': []
                }
                
                # Calculate Bonferroni adjusted alpha
                num_comparisons = len(data_values) * (len(data_values) - 1) // 2
                adjusted_alpha = alpha / num_comparisons
                
                # Calculate pairwise comparisons
                for i in range(len(data_values)):
                    for j in range(i+1, len(data_values)):
                        # Paired t-test
                        t_stat, p_val = stats.ttest_rel(
                            data_values[i][:n_blocks], 
                            data_values[j][:n_blocks]
                        )
                        
                        # Calculate mean difference
                        mean_diff = np.mean(data_values[i][:n_blocks] - data_values[j][:n_blocks])
                        
                        # Calculate effect size (Cohen's d for paired samples)
                        diff = data_values[i][:n_blocks] - data_values[j][:n_blocks]
                        cohens_d = mean_diff / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
                        
                        # Apply Bonferroni correction
                        p_val_adj = min(p_val * num_comparisons, 1.0)
                        
                        posthoc_results['pairwise_p_values'].append({
                            'group1': data_names[i],
                            'group2': data_names[j],
                            'p_value': p_val,
                            'p_value_adjusted': p_val_adj,
                            'significant': p_val_adj < alpha,
                            'mean_diff': mean_diff,
                            'effect_size': cohens_d,
                            'effect_size_type': "Cohen's d for paired samples"
                        })
                        
                        if p_val_adj < alpha:
                            posthoc_results['significant_pairs'].append(
                                (data_names[i], data_names[j], p_val_adj)
                            )
            except Exception as e:
                posthoc_results = {
                    'test': 'Paired t-tests',
                    'error': str(e)
                }
                
        # Aligned Ranks post-hoc: Wilcoxon signed-rank tests with Bonferroni correction
        elif test_type == 'aligned_ranks':
            try:
                if n_blocks is None:
                    # Get minimum length across all datasets
                    n_blocks = min(len(data) for data in data_values)
                
                posthoc_results = {
                    'test': 'Wilcoxon signed-rank tests with Bonferroni correction',
                    'description': 'Non-parametric post-hoc test for pairwise comparisons after Aligned Ranks Transform ANOVA',
                    'pairwise_p_values': [],
                    'significant_pairs': []
                }
                
                # Calculate Bonferroni adjusted alpha
                num_comparisons = len(data_values) * (len(data_values) - 1) // 2
                adjusted_alpha = alpha / num_comparisons
                
                # Calculate pairwise comparisons
                for i in range(len(data_values)):
                    for j in range(i+1, len(data_values)):
                        try:
                            # Wilcoxon signed-rank test
                            stat, p_val = stats.wilcoxon(
                                data_values[i][:n_blocks], 
                                data_values[j][:n_blocks]
                            )
                            
                            # Calculate median difference
                            median_diff = np.median(data_values[i][:n_blocks]) - np.median(data_values[j][:n_blocks])
                            
                            # Calculate effect size (r = Z/sqrt(N))
                            # Convert Wilcoxon statistic to Z-score
                            n = len(data_values[i][:n_blocks])
                            z = (stat - (n * (n + 1) / 4)) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
                            effect_size = abs(z) / np.sqrt(2 * n)
                            
                            # Apply Bonferroni correction
                            p_val_adj = min(p_val * num_comparisons, 1.0)
                            
                            posthoc_results['pairwise_p_values'].append({
                                'group1': data_names[i],
                                'group2': data_names[j],
                                'p_value': p_val,
                                'p_value_adjusted': p_val_adj,
                                'significant': p_val_adj < alpha,
                                'median_diff': median_diff,
                                'effect_size': effect_size,
                                'effect_size_type': "r (effect size)"
                            })
                            
                            if p_val_adj < alpha:
                                posthoc_results['significant_pairs'].append(
                                    (data_names[i], data_names[j], p_val_adj)
                                )
                        except Exception as e:
                            # If Wilcoxon fails, fall back to Mann-Whitney U
                            posthoc_results['pairwise_p_values'].append({
                                'group1': data_names[i],
                                'group2': data_names[j],
                                'error': str(e)
                            })
            except Exception as e:
                posthoc_results = {
                    'test': 'Wilcoxon signed-rank tests',
                    'error': str(e)
                }
                
        # Add multiple testing correction methods
        posthoc_results['multiple_testing_correction'] = {
            'method': 'Bonferroni',
            'description': 'Controls family-wise error rate by adjusting the significance level',
            'original_alpha': alpha
        }
        
        # Add FDR (Benjamini-Hochberg) correction
        try:
            if 'pairwise_p_values' in posthoc_results and posthoc_results['pairwise_p_values']:
                # Extract p-values
                p_values = [item['p_value'] for item in posthoc_results['pairwise_p_values']]
                
                # Sort p-values
                sorted_indices = np.argsort(p_values)
                sorted_p_values = np.array([p_values[i] for i in sorted_indices])
                
                # Calculate FDR corrected p-values
                n = len(p_values)
                fdr_p_values = np.zeros_like(sorted_p_values)
                
                # Apply Benjamini-Hochberg procedure
                for i in range(n-1, -1, -1):
                    fdr_p_values[i] = min(sorted_p_values[i] * n / (i + 1), 1.0)
                    if i < n-1:
                        fdr_p_values[i] = min(fdr_p_values[i], fdr_p_values[i+1])
                
                # Reorder FDR p-values to match original order
                fdr_p_values_reordered = np.zeros_like(fdr_p_values)
                for i, idx in enumerate(sorted_indices):
                    fdr_p_values_reordered[idx] = fdr_p_values[i]
                
                # Add FDR p-values and significance to the results
                for i, p_val_dict in enumerate(posthoc_results['pairwise_p_values']):
                    p_val_dict['fdr_p_value'] = fdr_p_values_reordered[i]
                    p_val_dict['fdr_significant'] = fdr_p_values_reordered[i] < alpha
                
                # Count significant pairs with FDR correction
                fdr_significant_pairs = []
                for i, p_val_dict in enumerate(posthoc_results['pairwise_p_values']):
                    if p_val_dict.get('fdr_significant', False):
                        group1 = p_val_dict['group1']
                        group2 = p_val_dict['group2']
                        p_val = p_val_dict['fdr_p_value']
                        fdr_significant_pairs.append((group1, group2, p_val))
                
                posthoc_results['fdr_significant_pairs'] = fdr_significant_pairs
                
                # Add FDR information
                posthoc_results['multiple_testing_correction']['fdr_method'] = 'Benjamini-Hochberg'
                posthoc_results['multiple_testing_correction']['fdr_description'] = 'Controls false discovery rate'
        except Exception as e:
            posthoc_results['multiple_testing_correction']['fdr_error'] = str(e)
        
        return posthoc_results


class HexoskinWavApp(tk.Tk):
    """GUI application for loading and analyzing Hexoskin WAV files"""
    
    def __init__(self):
        super().__init__()
        
        self.title("Hexoskin WAV File Analyzer - Made by Diego Malpica")
        self.geometry("1200x800")
        
        # Set application icon if available
        try:
            if platform.system() == "Windows":
                self.iconbitmap("icon.ico")
            else:
                logo = tk.PhotoImage(file="icon.png")
                self.iconphoto(True, logo)
        except:
            pass  # Icon not found, continue without it
        
        # Initialize style object first
        self.style = ttk.Style()
        
        # Apply modern theme
        self._set_theme()
        
        self.loader = HexoskinWavLoader()
        self.loaded_files = []
        self.selected_files_for_comparison = []
        
        # Configure app style
        self.style.configure('TLabel', font=('Segoe UI', 10))
        self.style.configure('TButton', font=('Segoe UI', 10))
        self.style.configure('TFrame', background=self._get_bg_color())
        self.style.configure('TLabelframe', background=self._get_bg_color())
        self.style.configure('TLabelframe.Label', font=('Segoe UI', 10, 'bold'))
        self.style.configure('TNotebook.Tab', font=('Segoe UI', 10))
        
        # Create a custom style for primary buttons
        self.style.configure('Primary.TButton', font=('Segoe UI', 10, 'bold'))
        
        # Create a custom style for section headers
        self.style.configure('Header.TLabel', font=('Segoe UI', 11, 'bold'))
        
        self._create_widgets()
        
        # Center window on screen
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
    
    def _set_theme(self):
        """Set the application theme based on the platform"""
        # Try to use a modern theme if available
        available_themes = self.style.theme_names()
        
        if 'clam' in available_themes:
            self.style.theme_use('clam')
        elif 'vista' in available_themes and platform.system() == 'Windows':
            self.style.theme_use('vista')
        elif 'aqua' in available_themes and platform.system() == 'Darwin':
            self.style.theme_use('aqua')
        
        # Configure colors based on theme
        self.bg_color = self._get_bg_color()
        self.fg_color = self._get_fg_color()
    
    def _get_bg_color(self):
        """Get background color based on theme"""
        return self.style.lookup('TFrame', 'background') or '#f0f0f0'
    
    def _get_fg_color(self):
        """Get foreground color based on theme"""
        return self.style.lookup('TLabel', 'foreground') or '#000000'
    
    def _create_widgets(self):
        """Create the widgets for the GUI"""
        # Main frame with padding
        main_frame = ttk.Frame(self, padding="10 10 10 10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a PanedWindow to allow resizable panels
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Left panel (controls)
        left_frame = ttk.Frame(paned_window, width=300)
        
        # Right panel (plots and stats)
        right_frame = ttk.Frame(paned_window)
        
        paned_window.add(left_frame, weight=1)
        paned_window.add(right_frame, weight=3)
        
        # ---------- Left Panel Components ----------
        # App title and version
        header_frame = ttk.Frame(left_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(header_frame, text="Hexoskin WAV Analyzer", 
                font=('Segoe UI', 14, 'bold')).pack(side=tk.TOP)
        ttk.Label(header_frame, text="Version 0.0.3", 
                font=('Segoe UI', 9)).pack(side=tk.TOP)
        
        # File operations section
        file_section = ttk.LabelFrame(left_frame, text="File Operations")
        file_section.pack(fill=tk.X, pady=5)
        
        # Button frame with grid layout for file operations
        file_btn_frame = ttk.Frame(file_section)
        file_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add file open button
        load_btn = ttk.Button(file_btn_frame, text="Open File", 
                            command=lambda: self._load_file(), style='Primary.TButton')
        load_btn.grid(column=0, row=0, sticky=tk.EW, padx=2, pady=2)
        
        # Add batch open button
        batch_btn = ttk.Button(file_btn_frame, text="Batch Open", 
                             command=lambda: self._batch_load_files())
        batch_btn.grid(column=1, row=0, sticky=tk.EW, padx=2, pady=2)
        
        # Add save CSV button
        save_btn = ttk.Button(file_btn_frame, text="Save CSV", 
                            command=lambda: self._save_to_csv())
        save_btn.grid(column=0, row=1, sticky=tk.EW, padx=2, pady=2)
        
        # Add export graph button
        export_btn = ttk.Button(file_btn_frame, text="Export Graph", 
                              command=lambda: self._export_graph())
        export_btn.grid(column=1, row=1, sticky=tk.EW, padx=2, pady=2)
        
        # Configure grid to expand buttons
        file_btn_frame.columnconfigure(0, weight=1)
        file_btn_frame.columnconfigure(1, weight=1)
        
        # Loaded files frame
        files_frame = ttk.LabelFrame(left_frame, text="Loaded Files")
        files_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Add a scrollbar to the listbox
        files_list_frame = ttk.Frame(files_frame)
        files_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        files_scrollbar = ttk.Scrollbar(files_list_frame)
        files_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Listbox for loaded files with modern styling
        self.files_listbox = tk.Listbox(
            files_list_frame, 
            font=('Segoe UI', 10),
            selectmode=tk.EXTENDED,
            activestyle='dotbox',
            selectbackground='#0078D7',
            selectforeground='white',
            exportselection=False
        )
        self.files_listbox.pack(fill=tk.BOTH, expand=True)
        self.files_listbox.config(yscrollcommand=files_scrollbar.set)
        files_scrollbar.config(command=self.files_listbox.yview)
        self.files_listbox.bind('<<ListboxSelect>>', self._on_file_select)
        
        # Comparison button for selected files
        compare_btn = ttk.Button(files_frame, text="Compare Selected Files", 
                               command=lambda: self._compare_selected_files())
        compare_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Statistics button for a single file
        stats_btn = ttk.Button(files_frame, text="Show Statistics", 
                             command=lambda: self._show_statistics())
        stats_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Unload button for selected files
        unload_btn = ttk.Button(files_frame, text="Unload Selected Files", 
                              command=lambda: self._unload_files())
        unload_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Processing frame
        processing_frame = ttk.LabelFrame(left_frame, text="Processing")
        processing_frame.pack(fill=tk.X, pady=10)
        
        # Resample controls
        resample_frame = ttk.Frame(processing_frame)
        resample_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(resample_frame, text="Resample (Hz):").pack(side=tk.LEFT)
        self.resample_var = tk.StringVar(value="100")
        resample_entry = ttk.Entry(resample_frame, textvariable=self.resample_var, width=8)
        resample_entry.pack(side=tk.LEFT, padx=5)
        
        resample_btn = ttk.Button(resample_frame, text="Apply", 
                                command=lambda: self._resample_data(), width=8)
        resample_btn.pack(side=tk.LEFT, padx=5)
        
        # Filter controls
        filter_frame = ttk.Frame(processing_frame)
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(filter_frame, text="Filter (Hz):").pack(side=tk.LEFT)
        
        filter_controls = ttk.Frame(filter_frame)
        filter_controls.pack(side=tk.LEFT)
        
        ttk.Label(filter_controls, text="Low:").pack(side=tk.LEFT)
        self.lowcut_var = tk.StringVar()
        lowcut_entry = ttk.Entry(filter_controls, textvariable=self.lowcut_var, width=5)
        lowcut_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(filter_controls, text="High:").pack(side=tk.LEFT, padx=(10, 0))
        self.highcut_var = tk.StringVar()
        highcut_entry = ttk.Entry(filter_controls, textvariable=self.highcut_var, width=5)
        highcut_entry.pack(side=tk.LEFT, padx=2)
        
        filter_btn = ttk.Button(processing_frame, text="Apply Filter", 
                              command=lambda: self._filter_data())
        filter_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Plot customization controls
        plot_config_frame = ttk.LabelFrame(left_frame, text="Plot Customization")
        plot_config_frame.pack(fill=tk.X, pady=10)
        
        # Plot color
        color_frame = ttk.Frame(plot_config_frame)
        color_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(color_frame, text="Line Color:").pack(side=tk.LEFT)
        self.color_var = tk.StringVar(value="blue")
        color_combo = ttk.Combobox(color_frame, textvariable=self.color_var, width=10, 
                                 state="readonly")
        color_combo['values'] = ('blue', 'red', 'green', 'orange', 'purple', 'black')
        color_combo.pack(side=tk.LEFT, padx=5)
        
        # Line width
        width_frame = ttk.Frame(plot_config_frame)
        width_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(width_frame, text="Line Width:").pack(side=tk.LEFT)
        self.width_var = tk.StringVar(value="1.0")
        width_entry = ttk.Entry(width_frame, textvariable=self.width_var, width=5)
        width_entry.pack(side=tk.LEFT, padx=5)
        
        # Time unit selection
        time_unit_frame = ttk.Frame(plot_config_frame)
        time_unit_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(time_unit_frame, text="Time Unit:").pack(side=tk.LEFT)
        self.time_unit_var = tk.StringVar(value="seconds")
        time_unit_combo = ttk.Combobox(time_unit_frame, textvariable=self.time_unit_var, 
                                     width=10, state="readonly")
        time_unit_combo['values'] = ('seconds', 'minutes', 'hours', 'days')
        time_unit_combo.pack(side=tk.LEFT, padx=5)
        
        # Custom title
        title_frame = ttk.Frame(plot_config_frame)
        title_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(title_frame, text="Title:").pack(side=tk.LEFT)
        self.title_var = tk.StringVar()
        title_entry = ttk.Entry(title_frame, textvariable=self.title_var)
        title_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Apply customization button
        customize_btn = ttk.Button(plot_config_frame, text="Apply Customization", 
                                 command=lambda: self._update_plot())
        customize_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # View control buttons
        view_frame = ttk.LabelFrame(left_frame, text="View Controls")
        view_frame.pack(fill=tk.X, pady=10)
        
        # Auto-fit button
        fit_btn = ttk.Button(view_frame, text="Fit to Window", 
                           command=lambda: self._fit_to_window())
        fit_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Zoom controls
        zoom_frame = ttk.Frame(view_frame)
        zoom_frame.pack(fill=tk.X, padx=5, pady=5)
        
        zoom_in_btn = ttk.Button(zoom_frame, text="Zoom In", command=lambda: self._zoom_in())
        zoom_in_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        
        zoom_out_btn = ttk.Button(zoom_frame, text="Zoom Out", command=lambda: self._zoom_out())
        zoom_out_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(2, 0))
        
        # Reset view button
        reset_view_btn = ttk.Button(view_frame, text="Reset View", 
                                  command=lambda: self._reset_view())
        reset_view_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Stats button
        stats_btn = ttk.Button(left_frame, text="Calculate Statistics", 
                             command=self._show_statistics)
        stats_btn.pack(fill=tk.X, pady=5)
        
        # Metadata display
        self.metadata_frame = ttk.LabelFrame(left_frame, text="Metadata")
        self.metadata_frame.pack(fill=tk.X, pady=10)
        
        # Add scrollbar to metadata text
        metadata_text_frame = ttk.Frame(self.metadata_frame)
        metadata_text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        metadata_scrollbar = ttk.Scrollbar(metadata_text_frame)
        metadata_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.metadata_text = tk.Text(metadata_text_frame, height=6, width=30, 
                                   font=('Consolas', 9), wrap=tk.WORD)
        self.metadata_text.pack(fill=tk.BOTH, expand=True)
        self.metadata_text.config(yscrollcommand=metadata_scrollbar.set)
        metadata_scrollbar.config(command=self.metadata_text.yview)
        
        # ---------- Right Panel Components ----------
        self.right_notebook = ttk.Notebook(right_frame)
        self.right_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Plot tab
        plot_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(plot_frame, text="Plot")
        
        # Create figure and canvas for matplotlib with improved styling
        plt.style.use('seaborn-v0_8-whitegrid')  # Modern style for plots
        self.figure = Figure(figsize=(5, 4), dpi=100, constrained_layout=True)
        self.figure.patch.set_facecolor('#F8F8F8')  # Light background
        
        self.plot_canvas = FigureCanvasTkAgg(self.figure, plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar for matplotlib
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        self.plot_toolbar = NavigationToolbar2Tk(self.plot_canvas, toolbar_frame)
        self.plot_toolbar.update()
        
        # Add a single subplot
        self.plot_ax = self.figure.add_subplot(111)
        self.plot_ax.set_title("No data loaded")
        self.plot_ax.set_xlabel("Time (seconds)")
        self.plot_ax.set_ylabel("Value")
        self.plot_canvas.draw()
        
        # Comparison tab
        comparison_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(comparison_frame, text="Comparison")
        
        # Create a PanedWindow for comparison tab
        comparison_paned = ttk.PanedWindow(comparison_frame, orient=tk.VERTICAL)
        comparison_paned.pack(fill=tk.BOTH, expand=True)
        
        # Upper frame for controls
        upper_frame = ttk.Frame(comparison_paned)
        comparison_paned.add(upper_frame, weight=1)
        
        # Test selection frame
        test_frame = ttk.LabelFrame(upper_frame, text="Statistical Test")
        test_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Test type selection
        test_type_frame = ttk.Frame(test_frame)
        test_type_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(test_type_frame, text="Test Type:").pack(side=tk.LEFT)
        self.test_type_var = tk.StringVar(value="mann_whitney")
        test_type_combo = ttk.Combobox(test_type_frame, textvariable=self.test_type_var, width=20, state="readonly")
        test_type_combo['values'] = (
            'mann_whitney',      # Non-parametric test for independent samples
            'wilcoxon',         # Non-parametric test for paired samples
            'ks_2samp',         # Two-sample Kolmogorov-Smirnov test
            't_test',           # Independent t-test
            'welch_t_test',     # Welch's t-test for unequal variances
            'anova',            # One-way ANOVA
            'kruskal',          # Kruskal-Wallis H-test
            'friedman'          # Friedman test
        )
        test_type_combo.pack(side=tk.LEFT, padx=5)
        
        # Test suggestion assistant button
        suggest_btn = ttk.Button(test_type_frame, text="Test Suggestion Assistant", 
                               command=lambda: self._suggest_test())
        suggest_btn.pack(side=tk.LEFT, padx=5)

        # Comparison plots section
        plots_frame = ttk.LabelFrame(upper_frame, text="Comparison Plots")
        plots_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollable plot area
        plot_scroll = ttk.Frame(plots_frame)
        plot_scroll.pack(fill=tk.BOTH, expand=True)

        self.comparison_canvas_scroll = tk.Canvas(plot_scroll)
        self.comparison_scrollbar = ttk.Scrollbar(plot_scroll, orient=tk.VERTICAL, 
                                                 command=self.comparison_canvas_scroll.yview)
        self.comparison_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.comparison_canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.comparison_canvas_scroll.configure(yscrollcommand=self.comparison_scrollbar.set)
        
        # Plot container
        self.comparison_inner_frame = ttk.Frame(self.comparison_canvas_scroll)
        self.comparison_canvas_window = self.comparison_canvas_scroll.create_window(
            (0, 0), window=self.comparison_inner_frame, anchor="nw", tags="inner_frame"
        )
        
        # Bind scroll events
        self.comparison_inner_frame.bind("<Configure>", self._configure_comparison_scroll_region)
        self.comparison_canvas_scroll.bind("<Configure>", self._configure_comparison_canvas)
        
        # Matplotlib figure
        self.comparison_figure = Figure(figsize=(5, 8), dpi=100, constrained_layout=True)
        self.comparison_figure.patch.set_facecolor('#F8F8F8')
        
        self.comparison_canvas_widget = FigureCanvasTkAgg(self.comparison_figure, self.comparison_inner_frame)
        self.comparison_canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Plot controls
        controls_frame = ttk.Frame(plots_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        reset_view_btn = ttk.Button(controls_frame, text="Reset View/Scrollbar", 
                                      command=lambda: self._fit_comparison_to_window())
        reset_view_btn.pack(side=tk.LEFT, padx=5)

        # Matplotlib toolbar
        toolbar_frame = ttk.Frame(plots_frame)
        toolbar_frame.pack(fill=tk.X)
        self.comp_toolbar = NavigationToolbar2Tk(self.comparison_canvas_widget, toolbar_frame)
        self.comp_toolbar.update()
        
        # Create comparison plots
        self.comp_ax1 = self.comparison_figure.add_subplot(211)  # Time series
        self.comp_ax2 = self.comparison_figure.add_subplot(212)  # Box plot
        
        # Data preprocessing frame
        preprocess_frame = ttk.LabelFrame(upper_frame, text="Data Preprocessing")
        preprocess_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Normalization options
        norm_frame = ttk.Frame(preprocess_frame)
        norm_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.normalize_var = tk.BooleanVar(value=False)
        norm_check = ttk.Checkbutton(norm_frame, text="Normalize Data", variable=self.normalize_var)
        norm_check.pack(side=tk.LEFT)
        
        self.norm_method_var = tk.StringVar(value="z_score")
        norm_method_combo = ttk.Combobox(norm_frame, textvariable=self.norm_method_var, 
                                       width=15, state="readonly")
        norm_method_combo['values'] = ('z_score', 'min_max', 'robust')
        norm_method_combo.pack(side=tk.LEFT, padx=5)
        
        # Alignment options
        align_frame = ttk.Frame(preprocess_frame)
        align_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.align_var = tk.BooleanVar(value=False)
        align_check = ttk.Checkbutton(align_frame, text="Align Datasets", variable=self.align_var)
        align_check.pack(side=tk.LEFT)
        
        ttk.Label(align_frame, text="Target Hz:").pack(side=tk.LEFT, padx=(5, 0))
        self.target_hz_var = tk.StringVar()
        target_hz_entry = ttk.Entry(align_frame, textvariable=self.target_hz_var, width=8)
        target_hz_entry.pack(side=tk.LEFT, padx=5)
        
        # Selected files for comparison
        selected_frame = ttk.LabelFrame(upper_frame, text="Selected Files")
        selected_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create scrollable listbox for selected files
        file_list_frame = ttk.Frame(selected_frame)
        file_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        file_list_scrollbar = ttk.Scrollbar(file_list_frame)
        file_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.comparison_file_listbox = tk.Listbox(file_list_frame, height=5, selectmode=tk.EXTENDED,
                                               yscrollcommand=file_list_scrollbar.set)
        self.comparison_file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_list_scrollbar.config(command=self.comparison_file_listbox.yview)
        
        # Compare button
        compare_btn_frame = ttk.Frame(upper_frame)
        compare_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Run comparison button
        compare_btn = ttk.Button(compare_btn_frame, text="Run Comparison", 
                               command=lambda: self._run_comparison_test_wrapper(), style='Primary.TButton')
        compare_btn.pack(fill=tk.X)
        
        # Progress bar
        progress_frame = ttk.Frame(upper_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                     maximum=100, mode='determinate')
        progress_bar.pack(fill=tk.X, side=tk.TOP)
        
        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.pack(side=tk.TOP)
        
        # Lower frame for results
        lower_frame = ttk.Frame(comparison_paned)
        comparison_paned.add(lower_frame, weight=2)
        
        # Results text widget with scrollbar
        results_frame = ttk.LabelFrame(lower_frame, text="Comparison Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar to results
        results_scroll = ttk.Scrollbar(results_frame)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.comparison_text = tk.Text(results_frame, wrap=tk.WORD, width=50, height=20,
                                     yscrollcommand=results_scroll.set)
        self.comparison_text.pack(fill=tk.BOTH, expand=True)
        results_scroll.config(command=self.comparison_text.yview)
        
        # Export results button
        export_btn = ttk.Button(lower_frame, text="Export Results", 
                              command=lambda: self._export_comparison_results())
        export_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # About tab
        about_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(about_frame, text="About")
        
        # Create scrollable text widget for About information
        about_text_frame = ttk.Frame(about_frame)
        about_text_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        about_scrollbar = ttk.Scrollbar(about_text_frame)
        about_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        about_text = tk.Text(about_text_frame, wrap=tk.WORD, font=('Segoe UI', 11), 
                           padx=10, pady=10, relief=tk.FLAT, bg=self._get_bg_color())
        about_text.pack(fill=tk.BOTH, expand=True)
        about_text.config(yscrollcommand=about_scrollbar.set)
        about_scrollbar.config(command=about_text.yview)
        
        # Disable editing
        about_text.config(state=tk.NORMAL)
        
        # Add content
        about_text.tag_configure("title", font=('Segoe UI', 16, 'bold'), justify='center')
        about_text.tag_configure("heading", font=('Segoe UI', 12, 'bold'), justify='center')
        about_text.tag_configure("normal", font=('Segoe UI', 11), justify='left')
        about_text.tag_configure("center", justify='center')
        about_text.tag_configure("author", font=('Segoe UI', 14, 'bold'), foreground='#0066cc', justify='center')
        about_text.tag_configure("subtitle", font=('Segoe UI', 12, 'italic'), justify='center')
        about_text.tag_configure("bullet", font=('Segoe UI', 11), lmargin1=20, lmargin2=30)
        
        about_text.insert(tk.END, "Hexoskin WAV File Analyzer\n", "title")
        about_text.insert(tk.END, "Version 0.0.3\n\n", "center")
        
        about_text.insert(tk.END, "Created by\n", "center")
        about_text.insert(tk.END, "Diego Malpica, MD\n", "author")
        about_text.insert(tk.END, "Aerospace Medicine & Physiological Research\n\n", "subtitle")
        
        about_text.insert(tk.END, "Developed for the Valquiria Space Analog Simulation\n\n", "center")
        
        about_text.insert(tk.END, "About This Tool\n", "heading")
        about_text.insert(tk.END, "This advanced application enables researchers to analyze physiological data "
                        "from Hexoskin devices with precision and ease. Designed specifically for "
                        "aerospace medicine applications, it bridges the gap between complex physiological "
                        "monitoring and actionable insights.\n\n", "normal")
        
        about_text.insert(tk.END, "Key Capabilities\n", "heading")
        features = [
            "Interactive visualization of physiological time series data",
            "Advanced signal processing with customizable filters",
            "Comprehensive statistical analysis with multiple normality tests",
            "Advanced descriptive statistics with distribution analysis",
            "Non-parametric and parametric comparison between multiple datasets (up to 15)",
            "Multiple post-hoc analysis methods with effect size calculations",
            "Support for repeated measures designs and factorial analysis",
            "Intelligent data alignment and normalization algorithms",
            "Flexible export options for research publications"
        ]
        
        for feature in features:
            about_text.insert(tk.END, f"• {feature}\n", "bullet")
        
        about_text.insert(tk.END, "\nNew in Version 0.0.3\n", "heading")
        updates = [
            "Fixed issue with notebook widget initialization to properly display comparisons",
            "Improved error handling in post-hoc analysis for multiple dataset comparisons",
            "Fixed post-hoc analysis display in statistical comparison results",
            "Enhanced consistency in variable naming throughout code",
            "Enhanced post-hoc analysis display with detailed results",
            "Improved statistical test suggestions with data-driven recommendations",
            "Added clear explanations when post-hoc analysis is not performed",
            "Enhanced multiple testing correction with FDR (False Discovery Rate)",
            "Improved effect size interpretations for all statistical tests",
            "Added comprehensive pairwise comparison tables",
            "Enhanced visualization of statistical results",
            "Fixed various indentation errors in code"
        ]
        
        for update in updates:
            about_text.insert(tk.END, f"• {update}\n", "bullet")
        
        about_text.insert(tk.END, "\nCollaborations & Support\n", "heading")
        about_text.insert(tk.END, "Centro de Telemedicina de Colombia\nWomen AeroSTEAM\n\n", "center")
        
        about_text.insert(tk.END, "License\n", "heading")
        about_text.insert(tk.END, "This is an open-source project available for academic and research use.\n", "normal")
        
        about_text.insert(tk.END, "\n© 2025 Diego Malpica, MD. All rights reserved.\n", "center")
        
        # Make text read-only
        about_text.config(state=tk.DISABLED)
    
    def _load_file(self):
        """Load a single WAV file"""
        file_path = filedialog.askopenfilename(
            title="Select Hexoskin WAV File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if file_path:
            self._process_file(file_path)
    
    def _batch_load_files(self):
        """Load multiple WAV files"""
        file_paths = filedialog.askopenfilenames(
            title="Select Hexoskin WAV Files",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        for file_path in file_paths:
            self._process_file(file_path)
    
    def _process_file(self, file_path):
        """Process a WAV file and add it to the list"""
        # Create a new loader instance for each file
        loader = HexoskinWavLoader()
        success = loader.load_wav_file(file_path)
        
        if success:
            # Add to our list of loaded files
            file_name = os.path.basename(file_path)
            self.loaded_files.append({
                'name': file_name,
                'path': file_path,
                'loader': loader
            })
            
            # Update the listbox
            self.files_listbox.insert(tk.END, file_name)
            
            # Select the newly added file
            self.files_listbox.selection_clear(0, tk.END)
            self.files_listbox.selection_set(tk.END)
            self.files_listbox.activate(tk.END)
            self._on_file_select(None)
    
    def _on_file_select(self, event):
        """Handle file selection in the listbox"""
        selection = self.files_listbox.curselection()
        
        # Store selected files for comparison
        self.selected_files_for_comparison = []
        for index in selection:
            self.selected_files_for_comparison.append(self.loaded_files[index])
        
        # Update the comparison file listbox if it exists
        if hasattr(self, 'comparison_file_listbox'):
            self.comparison_file_listbox.delete(0, tk.END)
            for file_info in self.selected_files_for_comparison:
                self.comparison_file_listbox.insert(tk.END, file_info['name'])
        
        # For the plot tab, use the first selected file
        if selection:
            index = selection[0]
            file_info = self.loaded_files[index]
            
            # Update metadata display
            metadata = file_info['loader'].get_metadata()
            metadata_str = '\n'.join([f"{k}: {v}" for k, v in metadata.items()])
            self.metadata_text.delete(1.0, tk.END)
            self.metadata_text.insert(tk.END, metadata_str)
            
            # Reset custom title to use file name
            sensor_name = metadata.get('sensor_name', 'Data')
            start_date = metadata.get('start_date', '')
            
            if start_date:
                self.title_var.set(f"{sensor_name} - {start_date}")
            else:
                self.title_var.set(sensor_name)
            
            # Update plot
            self._update_plot()
            
            # Run statistics analysis automatically
            # self._show_statistics()  # Commented out until method is implemented
    
    def _update_plot(self):
        """Update the plot with current customization settings"""
        selection = self.files_listbox.curselection()
        
        if selection:
            index = selection[0]
            file_info = self.loaded_files[index]
            
            # Get customization options
            try:
                line_width = float(self.width_var.get())
            except ValueError:
                line_width = 1.0
                
            color = self.color_var.get()
            title = self.title_var.get()
            time_unit = self.time_unit_var.get()
            
            # Clear the plot
            self.plot_ax.clear()
            
            # Plot with custom options
            plot_options = {
                'color': color,
                'linewidth': line_width,
            }
            
            # Get the data
            data = file_info['loader'].get_data().copy()
            
            # Handle the x-axis based on available data and time unit
            has_real_timestamps = 'datetime' in data.columns
            
            if has_real_timestamps:
                # Use real timestamps if available
                if time_unit == 'seconds':
                    # Use seconds from the start timestamp
                    x_data = data['abs_timestamp'] - data['abs_timestamp'].min()
                    xlabel = 'Time (seconds from start)'
                elif time_unit == 'minutes':
                    # Convert to minutes from the start
                    x_data = (data['abs_timestamp'] - data['abs_timestamp'].min()) / 60.0
                    xlabel = 'Time (minutes from start)'
                elif time_unit == 'hours':
                    # Convert to hours from the start
                    x_data = (data['abs_timestamp'] - data['abs_timestamp'].min()) / 3600.0
                    xlabel = 'Time (hours from start)'
                elif time_unit == 'days':
                    # Use actual dates for the x-axis
                    x_data = data['datetime']
                    xlabel = 'Date'
            else:
                # Use relative timestamps
                if time_unit == 'minutes':
                    x_data = data['timestamp'] / 60.0
                    xlabel = 'Time (minutes)'
                elif time_unit == 'hours':
                    x_data = data['timestamp'] / 3600.0
                    xlabel = 'Time (hours)'
                elif time_unit == 'days':
                    x_data = data['timestamp'] / 86400.0
                    xlabel = 'Time (days)'
                else:  # seconds (default)
                    x_data = data['timestamp']
                    xlabel = 'Time (seconds)'
            
            # Plot the data with appropriate x-axis
            self.plot_ax.plot(x_data, data['value'], **plot_options)
            
            # Set title
            if title:
                self.plot_ax.set_title(title)
            else:
                sensor_name = file_info['loader'].metadata.get('sensor_name', 'Unknown')
                start_date = file_info['loader'].metadata.get('start_date', None)
                
                if start_date:
                    self.plot_ax.set_title(f"{sensor_name} - {start_date}")
                else:
                    self.plot_ax.set_title(f"{sensor_name} Data")
            
            # Format x-axis based on time unit
            if has_real_timestamps and time_unit == 'days':
                # Format the date axis nicely
                self.figure.autofmt_xdate()
            
            # Set labels
            self.plot_ax.set_xlabel(xlabel)
            self.plot_ax.set_ylabel('Value')
            self.plot_ax.grid(True)
            
            # Get sensor name if not already defined
            if 'sensor_name' not in locals():
                sensor_name = file_info['loader'].metadata.get('sensor_name', 'Unknown')
            
            # Add legend with more information if available
            if has_real_timestamps:
                device = file_info['loader'].metadata.get('devices', 'Unknown')
                self.plot_ax.legend([f"{sensor_name} - Device: {device}"])
            else:
                self.plot_ax.legend([sensor_name])
            
            self.figure.tight_layout()
            self.plot_canvas.draw()
            
            # Store the current data range for zoom operations
            self.x_limits = self.plot_ax.get_xlim()
            self.y_limits = self.plot_ax.get_ylim()
    
    def _fit_to_window(self):
        """Auto-fit the data to the current window"""
        if not hasattr(self, 'plot_ax') or not self.files_listbox.curselection():
            return
            
        # Get the current data
        index = self.files_listbox.curselection()[0]
        file_info = self.loaded_files[index]
        data = file_info['loader'].get_data().copy()
        
        # Get time unit
        time_unit = self.time_unit_var.get()
        
        # Handle different axis data types
        has_real_timestamps = 'datetime' in data.columns
        
        if has_real_timestamps:
            # Use real timestamps if available
            if time_unit == 'seconds':
                x_data = data['abs_timestamp'] - data['abs_timestamp'].min()
            elif time_unit == 'minutes':
                x_data = (data['abs_timestamp'] - data['abs_timestamp'].min()) / 60.0
            elif time_unit == 'hours':
                x_data = (data['abs_timestamp'] - data['abs_timestamp'].min()) / 3600.0
            elif time_unit == 'days':
                x_data = data['datetime']
                # For datetime objects, we need special handling
                self.plot_ax.set_xlim(x_data.min(), x_data.max())
                y_margin = (data['value'].max() - data['value'].min()) * 0.05
                self.plot_ax.set_ylim(data['value'].min() - y_margin, data['value'].max() + y_margin)
                self.plot_canvas.draw()
                return
        else:
            # Use relative timestamps
            if time_unit == 'minutes':
                x_data = data['timestamp'] / 60.0
            elif time_unit == 'hours':
                x_data = data['timestamp'] / 3600.0
            elif time_unit == 'days':
                x_data = data['timestamp'] / 86400.0
            else:  # seconds (default)
                x_data = data['timestamp']
        
        # Set the limits with a small margin for numerical data
        x_min, x_max = x_data.min(), x_data.max()
        x_margin = (x_max - x_min) * 0.05
        y_margin = (data['value'].max() - data['value'].min()) * 0.05
        
        self.plot_ax.set_xlim(x_min - x_margin, x_max + x_margin)
        self.plot_ax.set_ylim(data['value'].min() - y_margin, data['value'].max() + y_margin)
        
        # Update the canvas
        self.plot_canvas.draw()
        
        # Store the current limits
        self.x_limits = self.plot_ax.get_xlim()
        self.y_limits = self.plot_ax.get_ylim()
    
    def _zoom_in(self):
        """Zoom in on the plot"""
        if not hasattr(self, 'plot_ax') or not self.files_listbox.curselection():
            return
            
        if not hasattr(self, 'x_limits'):
            self._fit_to_window()
            return
            
        # Get current limits
        x_min, x_max = self.plot_ax.get_xlim()
        y_min, y_max = self.plot_ax.get_ylim()
        
        # Calculate new limits (zoom in by 20%)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        x_range = (x_max - x_min) * 0.8 / 2  # 20% zoom in
        y_range = (y_max - y_min) * 0.8 / 2
        
        # Set new limits
        self.plot_ax.set_xlim(x_center - x_range, x_center + x_range)
        self.plot_ax.set_ylim(y_center - y_range, y_center + y_range)
        
        # Update the canvas
        self.plot_canvas.draw()
        
        # Store the current limits
        self.x_limits = self.plot_ax.get_xlim()
        self.y_limits = self.plot_ax.get_ylim()
    
    def _zoom_out(self):
        """Zoom out on the plot"""
        if not hasattr(self, 'plot_ax') or not self.files_listbox.curselection():
            return
            
        if not hasattr(self, 'x_limits'):
            self._fit_to_window()
            return
            
        # Get current limits
        x_min, x_max = self.plot_ax.get_xlim()
        y_min, y_max = self.plot_ax.get_ylim()
        
        # Calculate new limits (zoom out by 25%)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        x_range = (x_max - x_min) * 1.25 / 2  # 25% zoom out
        y_range = (y_max - y_min) * 1.25 / 2
        
        # Set new limits
        self.plot_ax.set_xlim(x_center - x_range, x_center + x_range)
        self.plot_ax.set_ylim(y_center - y_range, y_center + y_range)
        
        # Update the canvas
        self.plot_canvas.draw()
        
        # Store the current limits
        self.x_limits = self.plot_ax.get_xlim()
        self.y_limits = self.plot_ax.get_ylim()
    
    def _reset_view(self):
        """Reset the view to the original limits"""
        if not hasattr(self, 'plot_ax') or not self.files_listbox.curselection():
            return
            
        # Call fit to window to reset the view
        self._fit_to_window()
    
    def _compare_selected_files(self):
        """Compare the selected files"""
        if len(self.selected_files_for_comparison) < 2:
            messagebox.showinfo("Information", "Please select at least two files to compare")
            return
        
        # Limit to 15 files maximum
        if len(self.selected_files_for_comparison) > 15:
            messagebox.showinfo("Information", "Maximum 15 files can be compared at once. Using the first 15 selected files.")
            self.selected_files_for_comparison = self.selected_files_for_comparison[:15]
            
            # Update the comparison file listbox
            if hasattr(self, 'comparison_file_listbox'):
                self.comparison_file_listbox.delete(0, tk.END)
                for file_info in self.selected_files_for_comparison:
                    self.comparison_file_listbox.insert(tk.END, file_info['name'])
        
        # Switch to the comparison tab
        self.right_notebook.select(1)  # Select the comparison tab (index 1)
        
        # Update progress bar
        self.progress_var.set(0)
        self.progress_label.config(text="Preparing data for comparison...")
        self.update_idletasks()  # Force UI update
        
        # Clear the comparison plots
        self.comparison_figure.clear()
        
        # Adjust figure size based on number of files to compare (for scrolling)
        num_files = len(self.selected_files_for_comparison)
        figure_height = max(8, 4 + num_files * 1.5)  # Base height + extra height per file
        self.comparison_figure.set_size_inches(5, figure_height)
        
        # Recreate subplots with adjusted heights
        self.comp_ax1 = self.comparison_figure.add_subplot(211)  # Time series
        self.comp_ax2 = self.comparison_figure.add_subplot(212)  # Box plot
        
        # Process all selected files
        processed_data = []
        
        # Apply preprocessing to all datasets
        for i, file_info in enumerate(self.selected_files_for_comparison):
            # Update progress
            progress = (i / len(self.selected_files_for_comparison)) * 40  # 40% of progress for data prep
            self.progress_var.set(progress)
            self.progress_label.config(text=f"Processing {file_info['name']}...")
            self.update_idletasks()
            
            loader = file_info['loader']
            data = loader.get_data().copy()
            
            # Apply normalization if enabled
            if self.normalize_var.get():
                data = HexoskinWavLoader.normalize_dataset(
                    data, 
                    method=self.norm_method_var.get()
                )
            
            # Store processed data
            processed_data.append({
                'name': loader.metadata.get('sensor_name', f'File {i+1}'),
                'data': data,
                'start_date': loader.metadata.get('start_date', None)
            })
        
        # Apply alignment if enabled and there are exactly two datasets
        if self.align_var.get() and len(processed_data) == 2:
            self.progress_label.config(text="Aligning datasets...")
            self.progress_var.set(45)  # 45% progress
            self.update_idletasks()
            
            target_hz = None
            if self.target_hz_var.get():
                try:
                    target_hz = float(self.target_hz_var.get())
                except ValueError:
                    pass
                    
            aligned_data1, aligned_data2 = HexoskinWavLoader.align_datasets(
                processed_data[0]['data'],
                processed_data[1]['data'],
                target_hz=target_hz
            )
            
            # Update processed data with aligned data
            processed_data[0]['data'] = aligned_data1
            processed_data[1]['data'] = aligned_data2
            
            # Add note if datasets were aligned
            if len(aligned_data1) > 0:
                self.comp_ax1.set_title(f'Time Series Comparison (Aligned at {target_hz or "auto"} Hz)')
            else:
                messagebox.showinfo("Warning", "Datasets could not be aligned - no overlapping time range")
        
        # Update progress
        self.progress_var.set(50)  # 50% progress
        self.progress_label.config(text="Plotting data...")
        self.update_idletasks()
        
        # Get the time unit
        time_unit = self.time_unit_var.get()
                
        # Plot time series for all datasets
        for i, file_data in enumerate(processed_data):
            data = file_data['data'].copy()
            name = file_data['name']
            start_date = file_data['start_date']
            color = plt.cm.tab10(i % 10)  # Use tab10 colormap for distinct colors
            
            # Add start date to label if available
            if start_date:
                label = f"{name} - {start_date}"
            else:
                label = name
            
            # Handle the x-axis based on available data and time unit
            has_real_timestamps = 'datetime' in data.columns
            
            if has_real_timestamps:
                # Use real timestamps if available
                if time_unit == 'seconds':
                    # Use seconds from the start timestamp
                    x_data = data['abs_timestamp'] - data['abs_timestamp'].min()
                    xlabel = 'Time (seconds from start)'
                elif time_unit == 'minutes':
                    # Convert to minutes from the start
                    x_data = (data['abs_timestamp'] - data['abs_timestamp'].min()) / 60.0
                    xlabel = 'Time (minutes from start)'
                elif time_unit == 'hours':
                    # Convert to hours from the start
                    x_data = (data['abs_timestamp'] - data['abs_timestamp'].min()) / 3600.0
                    xlabel = 'Time (hours from start)'
                elif time_unit == 'days':
                    # Use actual dates for the x-axis
                    x_data = data['datetime']
                    xlabel = 'Date'
            else:
                # Use relative timestamps
                if time_unit == 'minutes':
                    x_data = data['timestamp'] / 60.0
                    xlabel = 'Time (minutes)'
                elif time_unit == 'hours':
                    x_data = data['timestamp'] / 3600.0
                    xlabel = 'Time (hours)'
                elif time_unit == 'days':
                    x_data = data['timestamp'] / 86400.0
                    xlabel = 'Time (days)'
                else:  # seconds (default)
                    x_data = data['timestamp']
                    xlabel = 'Time (seconds)'
            
            # Plot time series with reduced opacity for better visibility with many lines
            opacity = max(0.2, min(0.8, 1.0 - (len(processed_data) - 2) * 0.1))
            self.comp_ax1.plot(x_data, data['value'], 
                             color=color, 
                             label=label,
                             alpha=opacity)
        
        # Set time series plot properties
        if not self.comp_ax1.get_title():  # If title not already set by alignment
            self.comp_ax1.set_title('Time Series Comparison')
        
        # Format x-axis if using real dates
        if time_unit == 'days' and any('datetime' in data_item['data'].columns for data_item in processed_data):
            self.comparison_figure.autofmt_xdate()
            
        self.comp_ax1.set_xlabel(xlabel)
        self.comp_ax1.set_ylabel('Value')
        self.comp_ax1.grid(True)
        
        # Add legend with smaller font for many items
        legend_font_size = max(6, min(10, 10 - (len(processed_data) - 5) * 0.5))
        if len(processed_data) > 10:
            # For many datasets, place legend outside the plot
            self.comp_ax1.legend(fontsize=legend_font_size, loc='upper left', bbox_to_anchor=(1, 1))
        else:
            self.comp_ax1.legend(fontsize=legend_font_size)
        
        # Update progress
        self.progress_var.set(70)  # 70% progress
        self.progress_label.config(text="Creating boxplot...")
        self.update_idletasks()
        
        # Create box plot data
        data_for_boxplot = []
        labels_for_boxplot = []
        
        for file_data in processed_data:
            data_for_boxplot.append(file_data['data']['value'])
            
            # For many files, shorten the labels
            if len(processed_data) > 5:
                name = file_data['name']
                if len(name) > 15:  # Truncate long names
                    name = name[:12] + "..."
                labels_for_boxplot.append(name)
            else:
                if file_data.get('start_date'):
                    labels_for_boxplot.append(f"{file_data['name']}\n{file_data['start_date']}")
                else:
                    labels_for_boxplot.append(file_data['name'])
        
        # Create boxplot with appropriate sizing for the number of groups
        boxplot_width = max(0.3, min(0.8, 0.9 - (len(processed_data) - 2) * 0.05))
        self.comp_ax2.boxplot(data_for_boxplot, labels=labels_for_boxplot, widths=boxplot_width)
        
        # If many files, rotate labels for better fit
        if len(processed_data) > 5:
            self.comp_ax2.set_xticklabels(labels_for_boxplot, rotation=45, ha='right')
        
        norm_status = " (Normalized)" if self.normalize_var.get() else ""
        self.comp_ax2.set_title(f'Distribution Comparison{norm_status}')
        self.comp_ax2.set_ylabel('Value')
        self.comp_ax2.grid(True)
        
        # Update the figure
        self.comparison_figure.tight_layout()
        self.comparison_canvas_widget.draw()
        
        # Reset scrollbar position to top and update scroll region
        self.comparison_canvas_scroll.yview_moveto(0)
        self.comparison_canvas_scroll.update_idletasks()
        self.comparison_canvas_scroll.configure(scrollregion=self.comparison_canvas_scroll.bbox("all"))
        
        # Update progress
        self.progress_var.set(90)  # 90% progress
        self.progress_label.config(text="Running statistical tests...")
        self.update_idletasks()
        
        # Run the appropriate comparison test
        if len(self.selected_files_for_comparison) == 2:
            # For two datasets, use the two-group comparison
            # self._run_comparison_test()  # Commented out until method is implemented
            messagebox.showinfo("Information", "The comparison test functionality is not implemented yet.")
        else:
            # For multiple datasets, use the multi-group comparison
            self._run_multi_comparison_test(processed_data)
        
        # Finish progress
        self.progress_var.set(100)
        self.progress_label.config(text="Comparison complete")
        
        # Schedule resetting the progress bar after 2 seconds
        self.after(2000, lambda: (self.progress_var.set(0), self.progress_label.config(text="")))
    
    def _run_multi_comparison_test(self, processed_data):
        """Run statistical test to compare multiple files"""
        # Get the test type
        test_type = self.test_type_var.get()
        
        # For multiple groups, we should use appropriate tests
        if test_type in ['mann_whitney', 'wilcoxon', 'ks_2samp', 't_test', 'welch_t_test']:
            # If user selected a two-group test, switch to an appropriate multi-group test
            if test_type in ['t_test', 'welch_t_test']:
                test_type = 'anova'  # Parametric test for multiple groups
            else:
                test_type = 'kruskal'  # Non-parametric test for multiple groups
            
            messagebox.showinfo("Information", 
                               f"Switched to {test_type} test which is appropriate for comparing multiple groups.")
        
        try:
            # Run the comparison
            result = HexoskinWavLoader.compare_multiple_datasets(processed_data, test_type=test_type)
            
            # Ensure the comparison_text widget exists and is accessible
            try:
                # Try to access the comparison_text widget
                self.comparison_text.delete(1.0, tk.END)
            except (AttributeError, tk.TclError):
                # If we can't access it, create a new Text widget
                if hasattr(self, 'comparison_frame') and self.comparison_frame:
                    # Clear existing comparison frame contents
                    for widget in self.comparison_frame.winfo_children():
                        widget.destroy()
                    
                    # Create a new Text widget inside the comparison frame
                    stats_frame = ttk.Frame(self.comparison_frame)
                    stats_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    
                    stats_scrollbar = ttk.Scrollbar(stats_frame)
                    stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                    
                    self.comparison_text = tk.Text(
                        stats_frame,
                        font=('Segoe UI', 10),
                        wrap=tk.WORD,
                        height=15,
                        yscrollcommand=stats_scrollbar.set
                    )
                    self.comparison_text.pack(fill=tk.BOTH, expand=True)
                    stats_scrollbar.config(command=self.comparison_text.yview)
                else:
                    # Can't do anything without the comparison frame
                    messagebox.showinfo("Error", "The comparison view is not properly initialized")
                    return
            
            # Header
            self.comparison_text.insert(tk.END, f"Comparison of {len(processed_data)} Files\n")
            self.comparison_text.insert(tk.END, "=" * 50 + "\n\n")
            
            # Preprocessing information
            self.comparison_text.insert(tk.END, "PREPROCESSING\n")
            self.comparison_text.insert(tk.END, "-" * 50 + "\n")
            
            if self.align_var.get() and len(processed_data) == 2:
                target_hz_str = self.target_hz_var.get() or "auto"
                self.comparison_text.insert(tk.END, f"Alignment: Enabled for 2 datasets (Target Hz: {target_hz_str})\n")
            else:
                if len(processed_data) > 2:
                    self.comparison_text.insert(tk.END, "Alignment: Not applicable for multiple files\n")
                else:
                    self.comparison_text.insert(tk.END, "Alignment: Disabled\n")
            
            if self.normalize_var.get():
                self.comparison_text.insert(tk.END, f"Normalization: {self.norm_method_var.get()}\n")
            else:
                self.comparison_text.insert(tk.END, "Normalization: Disabled\n")
            
            self.comparison_text.insert(tk.END, "\n")
            
            # Test information
            self.comparison_text.insert(tk.END, "TEST INFORMATION\n")
            self.comparison_text.insert(tk.END, "-" * 50 + "\n")
            self.comparison_text.insert(tk.END, f"Test: {result['test_name']}\n")
            self.comparison_text.insert(tk.END, f"{result['test_description']}\n")
            self.comparison_text.insert(tk.END, f"Null hypothesis: {result['null_hypothesis']}\n\n")
            
            # Test results
            self.comparison_text.insert(tk.END, "TEST RESULTS\n")
            self.comparison_text.insert(tk.END, "-" * 50 + "\n")
            self.comparison_text.insert(tk.END, f"Statistic: {result['statistic']:.6f}\n")
            self.comparison_text.insert(tk.END, f"p-value: {result['p_value']:.6f}\n")
            self.comparison_text.insert(tk.END, f"Alpha: {result['alpha']}\n")
            self.comparison_text.insert(tk.END, f"Reject null hypothesis: {'Yes' if result['reject_null'] else 'No'}\n\n")
            
            # Effect size
            self.comparison_text.insert(tk.END, "EFFECT SIZE\n")
            self.comparison_text.insert(tk.END, "-" * 50 + "\n")
            self.comparison_text.insert(tk.END, f"{result['effect_size_name']}: {result['effect_size']:.6f}\n")
            self.comparison_text.insert(tk.END, f"Interpretation: {result['effect_interpretation']}\n\n")
            
            # Descriptive statistics
            self.comparison_text.insert(tk.END, "DESCRIPTIVE STATISTICS\n")
            self.comparison_text.insert(tk.END, "-" * 50 + "\n")
            
            # Create a table-like format for multiple files
            desc_stats = result['descriptive_stats']
            num_files = len(desc_stats['names'])
            
            # Calculate column widths
            name_width = max(15, max(len(name) for name in desc_stats['names']))
            col_width = max(name_width, 15)  # At least 15 characters wide
            
            # Create header
            header = f"{'Statistic':15}"
            for i, name in enumerate(desc_stats['names']):
                header += f" {name[:col_width]:>{col_width}}"
            self.comparison_text.insert(tk.END, header + "\n")
            self.comparison_text.insert(tk.END, "-" * (15 + col_width * num_files) + "\n")
            
            # Add statistics rows
            stats_to_display = [
                ('Count', 'counts', '0f'),
                ('Mean', 'means', '.4f'),
                ('Median', 'medians', '.4f'),
                ('Std Dev', 'stds', '.4f'),
                ('Min', 'mins', '.4f'),
                ('Max', 'maxs', '.4f')
            ]
            
            for label, key, format_str in stats_to_display:
                row = f"{label:15}"
                for i in range(num_files):
                    value = desc_stats[key][i]
                    row += f" {value:{format_str}:>{col_width}}"
                self.comparison_text.insert(tk.END, row + "\n")
            
            self.comparison_text.insert(tk.END, "\n")
            
            # Interpretation
            self.comparison_text.insert(tk.END, "INTERPRETATION\n")
            self.comparison_text.insert(tk.END, "-" * 50 + "\n")
            self.comparison_text.insert(tk.END, f"{result['interpretation']}\n")
            
            # Display post-hoc analysis for significant results
            if result['reject_null']:
                if 'post_hoc_results' in result and result['post_hoc_results']:
                    post_hoc = result['post_hoc_results']
                
                    self.comparison_text.insert(tk.END, "\nPOST-HOC ANALYSIS\n")
                    self.comparison_text.insert(tk.END, "-" * 50 + "\n")
                
                    # Display test information
                    self.comparison_text.insert(tk.END, f"Test: {post_hoc['test']}\n")
                    self.comparison_text.insert(tk.END, f"Description: {post_hoc['description']}\n\n")
                    
                    # Display pairwise comparisons in a table format
                    if 'pairwise_p_values' in post_hoc:
                        self.comparison_text.insert(tk.END, "Pairwise Comparisons:\n")
                        
                        # Calculate column widths for table
                        group_width = max(15, max(len(pair['group1']) for pair in post_hoc['pairwise_p_values']))
                        group_width = max(group_width, max(len(pair['group2']) for pair in post_hoc['pairwise_p_values']))
                        
                        # Create header
                        header = f"{'Group 1':{group_width}} {'Group 2':{group_width}} {'p-value':10} {'Adj. p-value':12} {'Effect Size':12} {'Significant':10}\n"
                        self.comparison_text.insert(tk.END, header)
                        self.comparison_text.insert(tk.END, "-" * (group_width * 2 + 44) + "\n")
                        
                        # Add rows
                        for pair in post_hoc['pairwise_p_values']:
                            p_val = pair['p_value']
                            p_adj = pair.get('p_value_adjusted', p_val)  # Use original p-value if no adjustment
                            effect = pair.get('effect_size', '')
                            
                            row = (f"{pair['group1']:{group_width}} "
                                  f"{pair['group2']:{group_width}} "
                                  f"{p_val:10.4f} "
                                  f"{p_adj:12.4f} "
                                  f"{effect:12.4f} "
                                  f"{'Yes' if pair['significant'] else 'No':10}\n")
                            self.comparison_text.insert(tk.END, row)
                        
                        self.comparison_text.insert(tk.END, "\n")
                        
                        # Display FDR results if available
                        if 'fdr_significant_pairs' in post_hoc:
                            self.comparison_text.insert(tk.END, "FDR-Corrected Results:\n")
                            self.comparison_text.insert(tk.END, f"Number of significant pairs after FDR correction: {len(post_hoc['fdr_significant_pairs'])}\n\n")
                    
                    # Display multiple testing correction information
                    if 'multiple_testing_correction' in post_hoc:
                        mtc = post_hoc['multiple_testing_correction']
                        self.comparison_text.insert(tk.END, "Multiple Testing Correction:\n")
                        self.comparison_text.insert(tk.END, f"Method: {mtc['method']}\n")
                        self.comparison_text.insert(tk.END, f"Description: {mtc['description']}\n")
                        if 'fdr_method' in mtc:
                            self.comparison_text.insert(tk.END, f"Additional correction: {mtc['fdr_method']} ({mtc['fdr_description']})\n")
                        self.comparison_text.insert(tk.END, "\n")
                    
                    # Display effect size interpretation if available
                    if any('effect_size_type' in pair for pair in post_hoc['pairwise_p_values']):
                        self.comparison_text.insert(tk.END, "Effect Size Interpretation:\n")
                        for pair in post_hoc['pairwise_p_values']:
                            if 'effect_size' in pair and 'effect_size_type' in pair:
                                effect_size = abs(pair['effect_size'])
                                interpretation = ""
                                if pair['effect_size_type'] == "Cohen's d" or pair['effect_size_type'] == "Hedges' g":
                                    if effect_size < 0.2:
                                        interpretation = "negligible effect"
                                    elif effect_size < 0.5:
                                        interpretation = "small effect"
                                    elif effect_size < 0.8:
                                        interpretation = "medium effect"
                                    else:
                                        interpretation = "large effect"
                                elif 'correlation' in pair['effect_size_type'].lower():
                                    if effect_size < 0.1:
                                        interpretation = "negligible effect"
                                    elif effect_size < 0.3:
                                        interpretation = "small effect"
                                    elif effect_size < 0.5:
                                        interpretation = "medium effect"
                                    else:
                                        interpretation = "large effect"
                                
                                if interpretation:
                                    self.comparison_text.insert(tk.END, 
                                        f"{pair['group1']} vs {pair['group2']}: {pair['effect_size_type']} = {pair['effect_size']:.4f} ({interpretation})\n")
                        
                        self.comparison_text.insert(tk.END, "\n")
                else:
                    self.comparison_text.insert(tk.END, "\nPOST-HOC ANALYSIS\n")
                    self.comparison_text.insert(tk.END, "-" * 50 + "\n")
                    if 'post_hoc_results' not in result:
                        self.comparison_text.insert(tk.END, "Post-hoc analysis was not performed. This might be due to:\n")
                        self.comparison_text.insert(tk.END, "1. The test type does not support post-hoc analysis\n")
                        self.comparison_text.insert(tk.END, "2. An error occurred during post-hoc analysis\n")
                    elif not result['post_hoc_results']:
                        self.comparison_text.insert(tk.END, "Post-hoc analysis was attempted but returned no results. This might be due to:\n")
                        self.comparison_text.insert(tk.END, "1. Insufficient data for pairwise comparisons\n")
                        self.comparison_text.insert(tk.END, "2. No significant pairwise differences were found\n")
                    self.comparison_text.insert(tk.END, "\nConsider checking the data or using a different test type if post-hoc analysis is needed.\n\n")
            else:
                self.comparison_text.insert(tk.END, "\nPOST-HOC ANALYSIS\n")
                self.comparison_text.insert(tk.END, "-" * 50 + "\n")
                self.comparison_text.insert(tk.END, "Post-hoc analysis was not performed because the main test did not show significant differences.\n")
                self.comparison_text.insert(tk.END, "Post-hoc tests are only meaningful when the main test indicates significant differences between groups.\n\n")
            
            # Recommendations
            self.comparison_text.insert(tk.END, "\nRECOMMENDATIONS\n")
            self.comparison_text.insert(tk.END, "-" * 50 + "\n")
            
            if result['reject_null']:
                self.comparison_text.insert(tk.END, "Since there is a statistically significant difference, you may want to:\n")
                self.comparison_text.insert(tk.END, "1. Investigate the factors that might be causing these differences\n")
                self.comparison_text.insert(tk.END, "2. Consider normalizing the data if comparing across different conditions\n")
                self.comparison_text.insert(tk.END, "3. Check for potential outliers or data collection issues\n")
                self.comparison_text.insert(tk.END, "4. If applicable, conduct follow-up tests to identify specific regions of difference\n")
            else:
                self.comparison_text.insert(tk.END, "Since there is no statistically significant difference, you may want to:\n")
                self.comparison_text.insert(tk.END, "1. Consider combining the datasets for increased statistical power\n")
                self.comparison_text.insert(tk.END, "2. Check if this result aligns with your expectations or hypotheses\n")
                self.comparison_text.insert(tk.END, "3. Consider if the sample size is adequate to detect meaningful differences\n")
                self.comparison_text.insert(tk.END, "4. If applicable, explore subgroups within the data where differences might exist\n")
            
            # Complete progress
            self.progress_var.set(100)
            self.progress_label.config(text="Analysis complete")
            
            # Update the scroll region to make sure all content is accessible
            self.comparison_canvas_scroll.update_idletasks()
            self.comparison_canvas_scroll.configure(scrollregion=self.comparison_canvas_scroll.bbox("all"))
        
        except Exception as e:
            self.comparison_text.delete(1.0, tk.END)
            self.comparison_text.insert(tk.END, f"Error performing comparison: {str(e)}\n\n")
            traceback.print_exc()
            
            if test_type == 'friedman':
                self.comparison_text.insert(tk.END, "Note: Friedman test requires all datasets to have the same length.\n")
                self.comparison_text.insert(tk.END, "Consider using Kruskal-Wallis or ANOVA tests instead.\n")
    
    def _configure_comparison_scroll_region(self, event):
        """Configure scroll region for the comparison plot"""
        self.comparison_canvas_scroll.configure(scrollregion=self.comparison_canvas_scroll.bbox("all"))
    
    def _configure_comparison_canvas(self, event):
        """Configure canvas for the comparison plot"""
        # Update the width of the canvas window to fill the canvas
        canvas_width = event.width
        self.comparison_canvas_scroll.itemconfig(self.comparison_canvas_window, width=canvas_width)
    
    def _fit_comparison_to_window(self):
        """Fit the comparison plots to the window and update the scrollbar"""
        # Reset scrollbar to top
        self.comparison_canvas_scroll.yview_moveto(0)
        # Update the scroll region
        self.comparison_canvas_scroll.update_idletasks()
        self.comparison_canvas_scroll.configure(scrollregion=self.comparison_canvas_scroll.bbox("all"))
        
        # Adjust the figure tight layout
        self.comparison_figure.tight_layout()
        self.comparison_canvas_widget.draw()
    
    def _export_comparison_results(self):
        """Export comparison results to a CSV file"""
        if not hasattr(self, 'last_comparison_results') or not self.last_comparison_results:
            messagebox.showinfo("No Results", "There are no comparison results to export.")
            return
            
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Comparison Results"
        )
        
        if not file_path:
            return  # User canceled
            
        try:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(["Comparison Results", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow([])
                
                # Write test information
                writer.writerow(["Test Type", self.test_type_var.get()])
                if hasattr(self, 'compared_files') and self.compared_files:
                    writer.writerow(["Compared Files"])
                    for file_name in self.compared_files:
                        writer.writerow(["", os.path.basename(file_name)])
                writer.writerow([])
                
                # Write test results
                writer.writerow(["Statistical Results"])
                for key, value in self.last_comparison_results.items():
                    if isinstance(value, dict):
                        writer.writerow([key])
                        for subkey, subvalue in value.items():
                            writer.writerow(["", subkey, subvalue])
                    else:
                        writer.writerow([key, value])
                
                # If we have p-values, add significance interpretation
                if 'p_value' in self.last_comparison_results:
                    p_value = self.last_comparison_results['p_value']
                    if isinstance(p_value, (int, float)):
                        writer.writerow([])
                        writer.writerow(["Significance"])
                        if p_value < 0.001:
                            writer.writerow(["", "Highly significant (p < 0.001)"])
                        elif p_value < 0.01:
                            writer.writerow(["", "Very significant (p < 0.01)"])
                        elif p_value < 0.05:
                            writer.writerow(["", "Significant (p < 0.05)"])
                        else:
                            writer.writerow(["", "Not significant (p ≥ 0.05)"])
            
            messagebox.showinfo("Export Successful", f"Results exported to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
            
    def _save_to_csv(self):
        """Save the currently selected data to a CSV file"""
        if not self.selected_item or not self.selected_file_info:
            messagebox.showinfo("Information", "Please select a file to save")
            return
            
        # Get the selected loader
        loader = self.selected_file_info.get('loader')
        if not loader or not hasattr(loader, 'get_data'):
            messagebox.showerror("Error", "Selected file data is not available")
            return
            
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Data to CSV"
        )
        
        if not file_path:
            return  # User canceled
            
        try:
            # Get the data
            data = loader.get_data()
            
            # Save to CSV
            data.to_csv(file_path, index=False)
            
            messagebox.showinfo("Export Successful", f"Data saved to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save data: {str(e)}")
            
    def _export_graph(self):
        """Export the current graph as an image file"""
        # Check if there's data to export
        if not hasattr(self, 'plot_ax') or not self.plot_ax:
            messagebox.showinfo("Information", "No graph to export")
            return
            
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ],
            title="Export Graph as Image"
        )
        
        if not file_path:
            return  # User canceled
            
        try:
            # Save the figure
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            
            messagebox.showinfo("Export Successful", f"Graph exported to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export graph: {str(e)}")
            
    def _unload_files(self):
        """Remove selected files from the list"""
        if not hasattr(self, 'file_tree') or not self.file_tree:
            return
            
        # Get selected items
        selected = self.file_tree.selection()
        if not selected:
            messagebox.showinfo("Information", "Please select files to unload")
            return
            
        # Confirm with user
        num_selected = len(selected)
        if messagebox.askyesno("Confirm", f"Unload {num_selected} selected file(s)?"):
            # Remove from file tree and data structures
            for item_id in selected:
                # Remove from tree
                self.file_tree.delete(item_id)
                
                # Remove from file_info dictionary if it exists
                if hasattr(self, 'file_info') and item_id in self.file_info:
                    del self.file_info[item_id]
                    
                # Remove from selected files for comparison if present
                if hasattr(self, 'selected_files_for_comparison'):
                    self.selected_files_for_comparison = [
                        f for f in self.selected_files_for_comparison 
                        if f.get('item_id') != item_id
                    ]
            
            # Clear the current plot if the selected file was unloaded
            if hasattr(self, 'selected_item') and self.selected_item in selected:
                # Clear selection
                self.selected_item = None
                self.selected_file_info = None
                
                # Reset plot
                if hasattr(self, 'plot_ax'):
                    self.plot_ax.clear()
                    self.plot_ax.set_title("No data loaded")
                    self.plot_ax.set_xlabel("Time (seconds)")
                    self.plot_ax.set_ylabel("Value")
                    self.plot_canvas.draw()
                
                # Clear stats
                if hasattr(self, 'stats_text'):
                    self.stats_text.delete(1.0, tk.END)
                
            messagebox.showinfo("Success", f"Unloaded {num_selected} file(s)")
            
    def _resample_data(self):
        """Resample the selected signal to a different sampling rate"""
        if not self.selected_item or not self.selected_file_info:
            messagebox.showinfo("Information", "Please select a file to resample")
            return
            
        # Get the selected loader
        loader = self.selected_file_info.get('loader')
        if not loader or not hasattr(loader, 'get_data'):
            messagebox.showerror("Error", "Selected file data is not available")
            return
            
        # Get current sample rate
        current_hz = None
        if hasattr(loader, 'sample_rate') and loader.sample_rate:
            current_hz = loader.sample_rate
            
        # Ask user for target Hz
        target_hz = None
        target_hz_str = tk.simpledialog.askstring(
            "Resample Data", 
            f"Enter target sample rate in Hz{' (Current: ' + str(current_hz) + ' Hz)' if current_hz else ''}:",
            initialvalue="256"
        )
        
        if not target_hz_str:
            return  # User canceled
            
        try:
            target_hz = float(target_hz_str)
            if target_hz <= 0:
                raise ValueError("Sample rate must be positive")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid sample rate: {str(e)}")
            return
            
        try:
            # Show progress
            self.progress_var.set(20)
            self.progress_label.config(text="Resampling data...")
            self.update_idletasks()
            
            # Get the data
            data = loader.get_data()
            
            # Update progress
            self.progress_var.set(50)
            self.update_idletasks()
            
            # Resample the data
            resampled_data = loader.resample_data(target_hz)
            
            # Update progress
            self.progress_var.set(80)
            self.update_idletasks()
            
            # Update the plot
            self._update_plot()
            
            # Update progress
            self.progress_var.set(100)
            self.progress_label.config(text=f"Resampled to {target_hz} Hz")
            self.update_idletasks()
            
            # Reset progress bar after 2 seconds
            self.after(2000, lambda: (self.progress_var.set(0), self.progress_label.config(text="")))
            
            messagebox.showinfo("Success", f"Data resampled to {target_hz} Hz")
            
        except Exception as e:
            self.progress_var.set(0)
            self.progress_label.config(text="")
            messagebox.showerror("Resampling Error", f"Failed to resample data: {str(e)}")

    def _show_statistics(self):
        """Display comprehensive statistics for the selected file"""
        selected_indices = self.files_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("Information", "Please select a file first")
            return
            
        # Use only the first selected file
        selected_index = selected_indices[0]
        selected_path = self.files_listbox.get(selected_index)
        
        # Find the corresponding file info from loaded_files
        file_info = None
        for file_data in self.loaded_files:
            if file_data['name'] == selected_path:
                file_info = file_data
                break
        
        if not file_info or not file_info.get('loader'):
            messagebox.showerror("Error", "File data not found. Please reload the file.")
            return
            
        loader = file_info['loader']
        stats = loader.get_descriptive_stats()
        
        if not stats:
            messagebox.showinfo("Information", "No statistics available for this file")
            return
            
        # Create statistics window
        stats_window = tk.Toplevel(self)
        stats_window.title(f"Statistical Analysis: {selected_path}")
        stats_window.geometry("900x700")
        stats_window.minsize(800, 600)
        
        # Set up the main container with padding
        main_frame = ttk.Frame(stats_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 1. Basic Statistics Tab
        basic_tab = ttk.Frame(notebook)
        notebook.add(basic_tab, text="Descriptive Statistics")
        
        # Basic stats table
        basic_frame = ttk.LabelFrame(basic_tab, text="Descriptive Statistics Summary")
        basic_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a treeview for basic statistics
        basic_tree = ttk.Treeview(basic_frame, columns=("Statistic", "Value",), show="headings")
        basic_tree.heading("Statistic", text="Statistic")
        basic_tree.heading("Value", text="Value")
        basic_tree.column("Statistic", width=200)
        basic_tree.column("Value", width=150)
        basic_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add a scrollbar to the treeview
        scrollbar = ttk.Scrollbar(basic_frame, orient=tk.VERTICAL, command=basic_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        basic_tree.configure(yscrollcommand=scrollbar.set)
        
        # Populate basic statistics
        basic_stats = stats.get('basic', {})
        for stat_name, stat_value in basic_stats.items():
            # Format number to 4 decimal places if it's a float
            if isinstance(stat_value, float):
                formatted_value = f"{stat_value:.4f}"
            else:
                formatted_value = str(stat_value)
                
            # Convert snake_case to Title Case for display
            display_name = stat_name.replace('_', ' ').title()
            basic_tree.insert("", tk.END, values=(display_name, formatted_value))
            
        # Add additional statistics
        additional_stats = stats.get('additional', {})
        for stat_name, stat_value in additional_stats.items():
            if stat_name == 'percentiles' or isinstance(stat_value, (dict, list)):
                continue  # Skip complex values for this simple table
                
            # Format number to 4 decimal places if it's a float
            if isinstance(stat_value, float):
                formatted_value = f"{stat_value:.4f}"
            else:
                formatted_value = str(stat_value)
                
            # Convert snake_case to Title Case for display
            display_name = stat_name.replace('_', ' ').title()
            basic_tree.insert("", tk.END, values=(display_name, formatted_value))
        
        # Add a scientific report frame
        scientific_report_frame = ttk.LabelFrame(basic_tab, text="Scientific Report")
        scientific_report_frame.pack(fill=tk.X, pady=10, padx=5)
        
        report_text = tk.Text(scientific_report_frame, wrap=tk.WORD, height=8, font=('Segoe UI', 10))
        report_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a formal scientific description of the descriptive statistics
        sample_size = basic_stats.get('count', 0)
        mean_value = basic_stats.get('mean', 0)
        std_value = basic_stats.get('std', 0)
        min_value = basic_stats.get('min', 0)
        max_value = basic_stats.get('max', 0)
        median_value = basic_stats.get('50%', 0)
        
        skewness = additional_stats.get('skewness', 0)
        kurtosis = additional_stats.get('kurtosis', 0)
        cv = additional_stats.get('coefficient_of_variation', 0)
        
        scientific_description = f"""The analysis of the physiological data sample (n = {sample_size}) revealed the following results:

The mean value was {mean_value:.2f} ± {std_value:.2f} (mean ± SD), with a range of {min_value:.2f} to {max_value:.2f}. The median value was {median_value:.2f}, indicating a {'positively' if skewness > 0 else 'negatively'} skewed distribution (skewness = {skewness:.3f}). The coefficient of variation was {cv:.2f}, suggesting {'high' if cv > 0.3 else 'moderate' if cv > 0.1 else 'low'} variability in the dataset.

The distribution exhibited a kurtosis value of {kurtosis:.3f}, indicating a {'leptokurtic (heavy-tailed)' if kurtosis > 0 else 'platykurtic (light-tailed)'} distribution compared to a normal distribution.
"""
        
        report_text.insert(tk.END, scientific_description)
        report_text.config(state=tk.DISABLED)
        
        # 2. Normality Tests Tab
        normality_tab = ttk.Frame(notebook)
        notebook.add(normality_tab, text="Normality Analysis")
        
        # Create a frame for normality tests
        normality_frame = ttk.LabelFrame(normality_tab, text="Normality Test Results")
        normality_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a text widget with scrollbar for displaying normality test results
        normality_text = tk.Text(normality_frame, wrap=tk.WORD, font=('Segoe UI', 10))
        normality_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        normality_sb = ttk.Scrollbar(normality_frame, orient=tk.VERTICAL, command=normality_text.yview)
        normality_sb.pack(side=tk.RIGHT, fill=tk.Y)
        normality_text.configure(yscrollcommand=normality_sb.set)
        
        # Populate normality test results
        normality_results = stats.get('normality', {})
        
        # Overall assessment
        overall = normality_results.get('overall_assessment', {})
        assessment = overall.get('assessment', "No assessment available")
        
        normality_text.insert(tk.END, "NORMALITY ASSESSMENT\n\n", "heading")
        normality_text.insert(tk.END, f"{assessment}\n\n")
        
        # Add recommendations
        recommendations = overall.get('recommendation', [])
        if recommendations:
            normality_text.insert(tk.END, "RECOMMENDATIONS\n\n", "heading")
            for i, rec in enumerate(recommendations, 1):
                normality_text.insert(tk.END, f"{i}. {rec}\n")
            normality_text.insert(tk.END, "\n")
        
        # Individual test results in formal scientific format
        normality_text.insert(tk.END, "STATISTICAL TESTS OF NORMALITY\n\n", "heading")
        
        test_names = [
            'shapiro_wilk', 'dagostino_k2', 'kolmogorov_smirnov', 
            'anderson_darling', 'jarque_bera'
        ]
        
        for test_name in test_names:
            test_result = normality_results.get(test_name)
            if not test_result:
                continue
                
            if 'error' in test_result:
                continue
                
            # Format the test name for display
            display_name = test_name.replace('_', ' ').title()
            statistic = test_result.get('statistic', 0)
            p_value = test_result.get('p_value', 1)
            normal = test_result.get('normal', False)
            
            # Get the degrees of freedom if available
            df = test_result.get('df', '')
            df_str = f"({df})" if df else ""
            
            # Format according to scientific standards
            if p_value < 0.001:
                p_value_str = "p < 0.001"
            else:
                p_value_str = f"p = {p_value:.3f}"
                
            # Create scientific report for this test
            if test_name == 'shapiro_wilk':
                test_stat_name = "W"
            elif test_name == 'dagostino_k2':
                test_stat_name = "K²"
            elif test_name == 'kolmogorov_smirnov':
                test_stat_name = "D"
            elif test_name == 'anderson_darling':
                test_stat_name = "A²"
            elif test_name == 'jarque_bera':
                test_stat_name = "JB"
            else:
                test_stat_name = "Statistic"
                
            scientific_report = f"{display_name}: {test_stat_name}{df_str} = {statistic:.3f}, {p_value_str}"
            
            # Add interpretation
            interpretation = f"The data {'appears to be normally distributed' if normal else 'deviates significantly from a normal distribution'} according to this test."
            
            normality_text.insert(tk.END, f"{scientific_report}\n")
            normality_text.insert(tk.END, f"{interpretation}\n\n")
        
        # Add distribution shape information in formal scientific format
        shape_info = normality_results.get('distribution_shape', {})
        if shape_info and 'error' not in shape_info:
            normality_text.insert(tk.END, "DISTRIBUTION SHAPE ANALYSIS\n\n", "heading")
            
            skewness = shape_info.get('skewness', 0)
            kurtosis = shape_info.get('kurtosis', 0)
            skew_interp = shape_info.get('skewness_interpretation', '')
            kurt_interp = shape_info.get('kurtosis_interpretation', '')
            
            normality_text.insert(tk.END, f"The distribution exhibited a skewness value of {skewness:.3f}, indicating {skew_interp.lower()}. ")
            normality_text.insert(tk.END, f"The kurtosis value of {kurtosis:.3f} suggests {kurt_interp.lower()}.\n\n")
            
            if abs(skewness) > 1 or abs(kurtosis) > 1:
                normality_text.insert(tk.END, "These values suggest a substantial deviation from normality, which may affect the validity of parametric statistical tests.\n\n")
            else:
                normality_text.insert(tk.END, "These values suggest a mild deviation from normality, which may not substantially affect the validity of parametric statistical tests.\n\n")
        
        # Configure text tags
        normality_text.tag_configure("heading", font=('Segoe UI', 11, 'bold'))
        
        # 3. QQ Plot Tab (if data available)
        qq_plot_data = normality_results.get('qq_plot_data')
        if qq_plot_data and 'error' not in qq_plot_data:
            qq_tab = ttk.Frame(notebook)
            notebook.add(qq_tab, text="QQ Plot Analysis")
            
            # Create a frame for the QQ plot
            qq_frame = ttk.Frame(qq_tab)
            qq_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Create a matplotlib figure
            fig = Figure(figsize=(6, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            # Plot QQ plot
            theoretical = qq_plot_data['theoretical_quantiles']
            observed = qq_plot_data['observed_quantiles']
            
            # Plot the QQ line
            ax.scatter(theoretical, observed, color='blue', alpha=0.6)
            
            # Add reference line
            min_val = min(theoretical)
            max_val = max(theoretical)
            slope = qq_plot_data['slope']
            intercept = qq_plot_data['intercept']
            
            ax.plot([min_val, max_val], 
                   [min_val * slope + intercept, max_val * slope + intercept], 
                   color='red', linestyle='--')
            
            ax.set_title('Quantile-Quantile Plot')
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel('Sample Quantiles')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add interpretation text below the plot
            qq_interpretation_frame = ttk.LabelFrame(qq_tab, text="QQ Plot Interpretation")
            qq_interpretation_frame.pack(fill=tk.X, padx=5, pady=5)
            
            qq_interp_text = tk.Text(qq_interpretation_frame, wrap=tk.WORD, height=6, font=('Segoe UI', 10))
            qq_interp_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Create formal scientific interpretation of QQ plot
            r_squared = qq_plot_data.get('r_squared', 0)
            
            qq_interpretation = f"""The Quantile-Quantile (Q-Q) plot examines the distribution of the data against the theoretical quantiles of a normal distribution. The plot reveals a correlation of R² = {r_squared:.3f} between the sample and theoretical quantiles.

{'Points that closely follow the reference line indicate alignment with a normal distribution, suggesting the data approximates normality.' if r_squared > 0.95 else 'Deviations from the reference line, particularly at the tails of the distribution, suggest departures from normality. This confirms the results of the formal normality tests.'}

{'The plot reveals no substantial deviations from normality.' if r_squared > 0.95 else 'The plot reveals deviations from normality that warrant consideration when selecting statistical methods for further analysis.'}
"""
            
            qq_interp_text.insert(tk.END, qq_interpretation)
            qq_interp_text.config(state=tk.DISABLED)
            
            # Embed the plot in the tkinter window
            canvas = FigureCanvasTkAgg(fig, master=qq_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar_frame = ttk.Frame(qq_frame)
            toolbar_frame.pack(fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
        
        # 4. Results and Recommendations Tab
        results_tab = ttk.Frame(notebook)
        notebook.add(results_tab, text="Results & Recommendations")
        
        results_frame = ttk.Frame(results_tab, padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create headers
        ttk.Label(results_frame, text="Statistical Analysis Results", 
                 font=('Segoe UI', 14, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        # Create a text widget for formal scientific results
        results_text = tk.Text(results_frame, wrap=tk.WORD, height=20, font=('Segoe UI', 10))
        results_text.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar to results text
        results_sb = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=results_text.yview)
        results_sb.pack(side=tk.RIGHT, fill=tk.Y)
        results_text.configure(yscrollcommand=results_sb.set)
        
        # Test normality of the data to provide appropriate recommendations
        is_normal = False
        normality_tests = normality_results.get('overall_assessment', {})
        if normality_tests:
            is_normal = normality_tests.get('is_normal', False)
        
        # Generate a comprehensive scientific summary
        full_summary = f"""STATISTICAL ANALYSIS REPORT

Sample Identification: {selected_path}
Sample Size: n = {sample_size}

DESCRIPTIVE STATISTICS
Mean ± SD: {mean_value:.2f} ± {std_value:.2f}
Median (IQR): {median_value:.2f} ({basic_stats.get('25%', 0):.2f} - {basic_stats.get('75%', 0):.2f})
Range: {min_value:.2f} to {max_value:.2f}
Coefficient of Variation: {cv:.3f}

DISTRIBUTION CHARACTERISTICS
Skewness: {skewness:.3f} ({'positive' if skewness > 0 else 'negative'} skew)
Kurtosis: {kurtosis:.3f} ({'leptokurtic' if kurtosis > 0 else 'platykurtic'})
Normality Assessment: The data distribution {'appears to be normal' if is_normal else 'deviates from normality'}

STATISTICAL INFERENCE
The analysis of this physiological dataset reveals {'normal distribution characteristics' if is_normal else 'non-normal distribution characteristics'}, which has important implications for subsequent statistical analyses. {'Parametric statistical methods are appropriate for further analysis.' if is_normal else 'Non-parametric statistical methods are recommended for further analysis.'}

RECOMMENDATIONS
1. {'Parametric tests such as t-tests and ANOVA are suitable for comparing this dataset with others.' if is_normal else 'Non-parametric tests such as Mann-Whitney U and Wilcoxon signed-rank tests are recommended for comparing this dataset with others.'}
2. {'For correlation analyses, Pearson correlation coefficients are appropriate.' if is_normal else 'For correlation analyses, Spearman or Kendall correlation coefficients are recommended.'}
3. {'Standard error of the mean (SEM) and confidence intervals can be reliably calculated.' if is_normal else 'Bootstrap methods for calculating confidence intervals should be considered.'}
4. {'Data transformation is not necessary for statistical analysis.' if is_normal else 'Data transformation (e.g., log, square root) might improve normality for certain analyses.'}

CONCLUSION
This dataset {'demonstrates characteristics consistent with a normal distribution and is suitable for parametric statistical analyses.' if is_normal else 'demonstrates deviations from normality, suggesting that non-parametric approaches or appropriate transformations should be considered for valid statistical inference.'}
"""
        
        results_text.insert(tk.END, full_summary)
        
        # Add export button
        export_frame = ttk.Frame(main_frame)
        export_frame.pack(fill=tk.X, pady=10)
        
        export_btn = ttk.Button(export_frame, text="Export Results", 
                              command=lambda: self._export_statistics(f"{selected_path}_stats_report.txt", full_summary))
        export_btn.pack(side=tk.RIGHT)
        
        # Make results text read-only
        results_text.config(state=tk.DISABLED)

    def _export_statistics(self, filename, content):
        """Export statistics to a text file"""
        # Ask for a save location
        save_path = filedialog.asksaveasfilename(
            title="Save Statistics Report",
            initialfile=filename,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not save_path:
            return  # User cancelled
            
        try:
            # If content is a string, write it directly as text
            if isinstance(content, str):
                with open(save_path, 'w', newline='') as f:
                    f.write(content)
            # If content is a dictionary (old stats format), write as CSV
            elif isinstance(content, dict):
                with open(save_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Statistic', 'Value'])
                
                    # Write basic statistics
                    basic_stats = content.get('basic', {})
                    if basic_stats:
                        writer.writerow(['=== BASIC STATISTICS ===', ''])
                        for key, value in basic_stats.items():
                            writer.writerow([key, value])
                
                    # Write additional statistics
                    additional_stats = content.get('additional', {})
                    if additional_stats:
                        writer.writerow(['=== ADDITIONAL STATISTICS ===', ''])
                        for key, value in additional_stats.items():
                            if isinstance(value, (dict, list)):
                                continue  # Skip complex values
                            writer.writerow([key, value])
                    
                    # Write percentiles
                    percentiles = additional_stats.get('percentiles', {})
                    if percentiles:
                        writer.writerow(['=== PERCENTILES ===', ''])
                        for key, value in percentiles.items():
                            writer.writerow([f"Percentile {key}", value])
                
                    # Write normality test results
                    normality = content.get('normality', {})
                    if normality:
                        writer.writerow(['=== NORMALITY TESTS ===', ''])
                        for test_name, test_result in normality.items():
                            if test_name == 'overall_assessment' or test_name == 'distribution_shape':
                                continue
                        
                            if isinstance(test_result, dict) and 'error' not in test_result:
                                writer.writerow([f"{test_name.replace('_', ' ').title()} Statistic", test_result.get('statistic', '')])
                                writer.writerow([f"{test_name.replace('_', ' ').title()} p-value", test_result.get('p_value', '')])
                                writer.writerow([f"{test_name.replace('_', ' ').title()} Normal", test_result.get('normal', '')])
                
                    # Write overall assessment
                    overall = normality.get('overall_assessment', {})
                    if overall:
                        writer.writerow(['=== OVERALL ASSESSMENT ===', ''])
                        writer.writerow(['assessment', overall.get('assessment', '')])
                        
                        # Write recommendations
                        recommendations = overall.get('recommendation', [])
                        for i, rec in enumerate(recommendations, 1):
                            writer.writerow([f'recommendation_{i}', rec])
            else:
                raise ValueError("Unsupported content type for export")
            
            messagebox.showinfo("Success", f"Statistics exported to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export statistics: {str(e)}")

    def _run_comparison_test_wrapper(self):
        """Wrapper to handle running the appropriate comparison test"""
        if not self.selected_files_for_comparison:
            messagebox.showinfo("Information", "Please select files to compare")
            return

        # Get the selected files' data
        processed_data = []
        for file_info in self.selected_files_for_comparison:
            loader = file_info['loader']
            data = loader.get_data().copy()
            
            # Apply normalization if enabled
            if self.normalize_var.get():
                data = HexoskinWavLoader.normalize_dataset(
                    data, 
                    method=self.norm_method_var.get()
                )
            
            # Store processed data
            processed_data.append({
                'name': loader.metadata.get('sensor_name', f'File {len(processed_data)+1}'),
                'data': data,
                'start_date': loader.metadata.get('start_date', None)
            })

        # Apply alignment if enabled and there are exactly two datasets
        if self.align_var.get() and len(processed_data) == 2:
            target_hz = None
            if self.target_hz_var.get():
                try:
                    target_hz = float(self.target_hz_var.get())
                except ValueError:
                    pass
                    
            aligned_data1, aligned_data2 = HexoskinWavLoader.align_datasets(
                processed_data[0]['data'],
                processed_data[1]['data'],
                target_hz=target_hz
            )
            
            # Update processed data with aligned data
            processed_data[0]['data'] = aligned_data1
            processed_data[1]['data'] = aligned_data2

        # Run the appropriate comparison test
        if len(processed_data) == 2:
            self._run_comparison_test(processed_data)
        else:
            self._run_multi_comparison_test(processed_data)

    def _run_comparison_test(self, processed_data):
        """Run statistical comparison between two datasets"""
        # Get the test type
        test_type = self.test_type_var.get()
        
        # Get the two datasets
        dataset1 = processed_data[0]['data']
        dataset2 = processed_data[1]['data']
        
        try:
            # Run the comparison
            comparison_results = HexoskinWavLoader.compare_datasets(
                dataset1, dataset2, test_type=test_type
            )
            
            # Clear previous results
            self.comparison_text.delete(1.0, tk.END)
            
            # Create a formal scientific report
            name1 = processed_data[0]['name']
            name2 = processed_data[1]['name']
            
            # Header
            self.comparison_text.insert(tk.END, f"Statistical Comparison: {name1} vs {name2}\n")
            self.comparison_text.insert(tk.END, "=" * 50 + "\n\n")
            
            # Test Information
            self.comparison_text.insert(tk.END, "TEST INFORMATION\n")
            self.comparison_text.insert(tk.END, "-" * 50 + "\n")
            self.comparison_text.insert(tk.END, f"Test: {comparison_results['test_name']}\n")
            self.comparison_text.insert(tk.END, f"Description: {comparison_results['test_description']}\n")
            self.comparison_text.insert(tk.END, f"Null Hypothesis: {comparison_results['null_hypothesis']}\n\n")
            
            # Results
            self.comparison_text.insert(tk.END, "RESULTS\n")
            self.comparison_text.insert(tk.END, "-" * 50 + "\n")
            self.comparison_text.insert(tk.END, f"Test Statistic: {comparison_results['statistic']:.4f}\n")
            self.comparison_text.insert(tk.END, f"P-value: {comparison_results['p_value']:.4f}\n")
            self.comparison_text.insert(tk.END, f"Effect Size ({comparison_results['effect_size_name']}): {comparison_results['effect_size']:.4f}\n")
            self.comparison_text.insert(tk.END, f"Effect Interpretation: {comparison_results['effect_interpretation']}\n\n")
            
            # Interpretation
            self.comparison_text.insert(tk.END, "INTERPRETATION\n")
            self.comparison_text.insert(tk.END, "-" * 50 + "\n")
            self.comparison_text.insert(tk.END, f"{comparison_results['interpretation']}\n\n")
            
            # Post-hoc results if available
            if comparison_results.get('post_hoc_results'):
                post_hoc = comparison_results['post_hoc_results']
                self.comparison_text.insert(tk.END, "POST-HOC ANALYSIS\n")
                self.comparison_text.insert(tk.END, "-" * 50 + "\n")
                
                if 'pairwise_p_values' in post_hoc:
                    for pair in post_hoc['pairwise_p_values']:
                        self.comparison_text.insert(tk.END, 
                            f"{pair['group1']} vs {pair['group2']}: p = {pair['p_value']:.4f} "
                            f"({'significant' if pair['significant'] else 'not significant'})\n"
                        )
                
                self.comparison_text.insert(tk.END, "\n")
            
            # Store results for potential export
            self.last_comparison_results = comparison_results
            
        except Exception as e:
            self.comparison_text.delete(1.0, tk.END)
            self.comparison_text.insert(tk.END, f"Error performing comparison: {str(e)}\n")
            traceback.print_exc()

    def _suggest_test(self):
        """Suggest appropriate statistical test based on data characteristics"""
        if not self.selected_files_for_comparison:
            messagebox.showinfo("Information", "Please select files to compare first")
            return

        # Create a new window for the assistant
        assistant_window = tk.Toplevel(self)
        assistant_window.title("Test Suggestion Assistant")
        assistant_window.geometry("600x700")
        assistant_window.minsize(500, 600)

        # Main frame with padding
        main_frame = ttk.Frame(assistant_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(main_frame, text="Statistical Test Suggestion Assistant", 
                 font=('Segoe UI', 14, 'bold')).pack(pady=(0, 10))

        # Create a text widget for displaying the analysis
        analysis_text = tk.Text(main_frame, wrap=tk.WORD, font=('Segoe UI', 10))
        analysis_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=analysis_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        analysis_text.config(yscrollcommand=scrollbar.set)

        # Analyze the data
        num_datasets = len(self.selected_files_for_comparison)
        
        # Header
        analysis_text.insert(tk.END, "DATA ANALYSIS\n", "heading")
        analysis_text.insert(tk.END, "=" * 50 + "\n\n")
        
        # Number of datasets
        analysis_text.insert(tk.END, f"Number of datasets selected: {num_datasets}\n\n")

        if num_datasets < 2:
            analysis_text.insert(tk.END, "Please select at least two datasets for comparison.\n")
            analysis_text.tag_configure("heading", font=('Segoe UI', 11, 'bold'))
            return

        # Process the data
        processed_data = []
        normality_results = []
        sample_sizes = []
        variances = []

        for file_info in self.selected_files_for_comparison:
            loader = file_info['loader']
            data = loader.get_data()['value'].values
            
            # Check normality
            if len(data) > 5000:
                # For large datasets, use D'Agostino's K^2 test
                stat, p_value = stats.normaltest(data)
            else:
                # For smaller datasets, use Shapiro-Wilk test
                stat, p_value = stats.shapiro(data)
            
            is_normal = p_value > 0.05
            normality_results.append(is_normal)
            sample_sizes.append(len(data))
            variances.append(np.var(data, ddof=1))

        # Check if sample sizes are equal
        equal_sample_sizes = len(set(sample_sizes)) == 1

        # Check homogeneity of variance using Levene's test
        if num_datasets == 2:
            data1 = self.selected_files_for_comparison[0]['loader'].get_data()['value'].values
            data2 = self.selected_files_for_comparison[1]['loader'].get_data()['value'].values
            _, levene_p = stats.levene(data1, data2)
            equal_variances = levene_p > 0.05
        else:
            # For multiple datasets
            datasets = [f['loader'].get_data()['value'].values for f in self.selected_files_for_comparison]
            _, levene_p = stats.levene(*datasets)
            equal_variances = levene_p > 0.05

        # Display data characteristics
        analysis_text.insert(tk.END, "DATA CHARACTERISTICS\n", "heading")
        analysis_text.insert(tk.END, "-" * 50 + "\n\n")

        # Normality
        analysis_text.insert(tk.END, "Normality Test Results:\n")
        for i, (is_normal, size) in enumerate(zip(normality_results, sample_sizes)):
            name = self.selected_files_for_comparison[i]['name']
            analysis_text.insert(tk.END, f"Dataset {i+1} ({name}):\n")
            analysis_text.insert(tk.END, f"• Sample size: {size}\n")
            analysis_text.insert(tk.END, f"• Distribution: {'Normal' if is_normal else 'Non-normal'}\n")
        analysis_text.insert(tk.END, "\n")

        # Sample sizes
        analysis_text.insert(tk.END, "Sample Sizes:\n")
        analysis_text.insert(tk.END, f"• {'Equal' if equal_sample_sizes else 'Unequal'} sample sizes\n")
        if not equal_sample_sizes:
            analysis_text.insert(tk.END, "• Sizes: " + ", ".join(str(s) for s in sample_sizes) + "\n")
        analysis_text.insert(tk.END, "\n")

        # Variance homogeneity
        analysis_text.insert(tk.END, "Variance Homogeneity:\n")
        analysis_text.insert(tk.END, f"• {'Equal' if equal_variances else 'Unequal'} variances (Levene's test p={levene_p:.4f})\n\n")

        # Test suggestions
        analysis_text.insert(tk.END, "RECOMMENDED TESTS\n", "heading")
        analysis_text.insert(tk.END, "-" * 50 + "\n\n")

        if num_datasets == 2:
            # Two-group comparison
            all_normal = all(normality_results)
            
            if all_normal:
                if equal_variances:
                    analysis_text.insert(tk.END, "Primary Recommendation:\n")
                    analysis_text.insert(tk.END, "• Independent t-test\n")
                    analysis_text.insert(tk.END, "  Reason: Data is normally distributed with equal variances\n\n")
                else:
                    analysis_text.insert(tk.END, "Primary Recommendation:\n")
                    analysis_text.insert(tk.END, "• Welch's t-test\n")
                    analysis_text.insert(tk.END, "  Reason: Data is normally distributed but variances are unequal\n\n")
            else:
                analysis_text.insert(tk.END, "Primary Recommendation:\n")
                analysis_text.insert(tk.END, "• Mann-Whitney U test\n")
                analysis_text.insert(tk.END, "  Reason: Data is not normally distributed\n\n")

            if equal_sample_sizes:
                analysis_text.insert(tk.END, "Alternative Options:\n")
                analysis_text.insert(tk.END, "• Wilcoxon signed-rank test (if samples are paired)\n")
                analysis_text.insert(tk.END, "• Kolmogorov-Smirnov test (to compare entire distributions)\n")
        else:
            # Multiple group comparison
            all_normal = all(normality_results)
            
            if all_normal:
                if equal_variances:
                    analysis_text.insert(tk.END, "Primary Recommendation:\n")
                    analysis_text.insert(tk.END, "• One-way ANOVA\n")
                    analysis_text.insert(tk.END, "  Reason: Data is normally distributed with equal variances\n\n")
                else:
                    analysis_text.insert(tk.END, "Primary Recommendation:\n")
                    analysis_text.insert(tk.END, "• Welch's ANOVA\n")
                    analysis_text.insert(tk.END, "  Reason: Data is normally distributed but variances are unequal\n\n")
            else:
                if equal_sample_sizes:
                    analysis_text.insert(tk.END, "Primary Recommendation:\n")
                    analysis_text.insert(tk.END, "• Friedman test\n")
                    analysis_text.insert(tk.END, "  Reason: Data is not normally distributed but sample sizes are equal\n\n")
                else:
                    analysis_text.insert(tk.END, "Primary Recommendation:\n")
                    analysis_text.insert(tk.END, "• Kruskal-Wallis H-test\n")
                    analysis_text.insert(tk.END, "  Reason: Data is not normally distributed and sample sizes are unequal\n\n")

        # Additional considerations
        analysis_text.insert(tk.END, "ADDITIONAL CONSIDERATIONS\n", "heading")
        analysis_text.insert(tk.END, "-" * 50 + "\n\n")

        if not all_normal:
            analysis_text.insert(tk.END, "Data Transformation Options:\n")
            analysis_text.insert(tk.END, "• Consider log transformation for right-skewed data\n")
            analysis_text.insert(tk.END, "• Consider square root transformation for count data\n")
            analysis_text.insert(tk.END, "• Consider Box-Cox transformation for complex distributions\n\n")

        if not equal_variances:
            analysis_text.insert(tk.END, "Variance Considerations:\n")
            analysis_text.insert(tk.END, "• Use tests that don't assume equal variances\n")
            analysis_text.insert(tk.END, "• Consider normalizing or standardizing the data\n\n")

        # Apply text tags
        analysis_text.tag_configure("heading", font=('Segoe UI', 11, 'bold'))

        # Make text read-only
        analysis_text.config(state=tk.DISABLED)

        # Add buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=(10, 0))

        # Add Apply Suggestion button
        def apply_suggestion():
            if num_datasets == 2:
                if all_normal:
                    if equal_variances:
                        self.test_type_var.set("t_test")
                    else:
                        self.test_type_var.set("welch_t_test")
                else:
                    self.test_type_var.set("mann_whitney")
            else:
                if all_normal:
                    if equal_variances:
                        self.test_type_var.set("anova")
                    else:
                        self.test_type_var.set("welch_anova")
                else:
                    if equal_sample_sizes:
                        self.test_type_var.set("friedman")
                    else:
                        self.test_type_var.set("kruskal")
            assistant_window.destroy()

        apply_btn = ttk.Button(buttons_frame, text="Apply Suggestion", command=apply_suggestion)
        apply_btn.pack(side=tk.LEFT, padx=5)

        # Add Close button
        close_btn = ttk.Button(buttons_frame, text="Close", command=assistant_window.destroy)
        close_btn.pack(side=tk.RIGHT, padx=5)

        # Center the window
        assistant_window.transient(self)
        assistant_window.grab_set()
        self.wait_window(assistant_window)


def main():
    """Run the application"""
    app = HexoskinWavApp()
    app.mainloop()


if __name__ == "__main__":
    main() 