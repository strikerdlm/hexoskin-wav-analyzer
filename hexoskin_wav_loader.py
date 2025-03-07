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
from scipy import stats, signal
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
        
        # Find mode
        mode_result = stats.mode(data_values)
        mode_value = mode_result.mode[0]
        mode_count = mode_result.count[0]
        
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
        interp_func1 = scipy.interpolate.interp1d(
            dataset1['timestamp'], 
            dataset1['value'],
            bounds_error=False,
            fill_value="extrapolate"
        )
        
        interp_func2 = scipy.interpolate.interp1d(
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
            'std': np.std(data1),
            'min': np.min(data1),
            'max': np.max(data1),
            'count': len(data1)
        }
        
        desc2 = {
            'mean': np.mean(data2),
            'median': np.median(data2),
            'std': np.std(data2),
            'min': np.min(data2),
            'max': np.max(data2),
            'count': len(data2)
        }
        
        # Perform statistical test
        if test_type == 'mann_whitney':
            # Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            test_name = "Mann-Whitney U test"
            test_description = "Non-parametric test to determine if two independent samples are drawn from the same distribution"
            null_hypothesis = "The distributions of both samples are equal"
            
        elif test_type == 'wilcoxon':
            # Wilcoxon signed-rank test
            statistic, p_value = stats.wilcoxon(data1, data2)
            test_name = "Wilcoxon signed-rank test"
            test_description = "Non-parametric test for paired samples to determine if the differences come from a distribution with zero median"
            null_hypothesis = "The differences between paired samples have zero median"
            
        elif test_type == 'ks_2samp':
            # Two-sample Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(data1, data2)
            test_name = "Two-sample Kolmogorov-Smirnov test"
            test_description = "Non-parametric test to determine if two independent samples are drawn from the same continuous distribution"
            null_hypothesis = "The two samples come from the same distribution"
            
        elif test_type == 't_test':
            # Independent t-test
            statistic, p_value = stats.ttest_ind(data1, data2, equal_var=True)
            test_name = "Independent t-test"
            test_description = "Parametric test to determine if two independent samples have different means"
            null_hypothesis = "The means of the two samples are equal"
            
        elif test_type == 'welch_t_test':
            # Welch's t-test
            statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False)
            test_name = "Welch's t-test"
            test_description = "Parametric test to determine if two independent samples have different means (not assuming equal variances)"
            null_hypothesis = "The means of the two samples are equal"
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Determine if null hypothesis should be rejected
        alpha = 0.05
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
        elif test_type == 't_test':
            # Calculate Cohen's d (effect size for t-test)
            mean_diff = np.mean(data1) - np.mean(data2)
            pooled_std = np.sqrt((np.std(data1, ddof=1)**2 + np.std(data2, ddof=1)**2) / 2)
            effect_size = mean_diff / pooled_std
            effect_size_name = "Cohen's d"
        elif test_type == 'welch_t_test':
            # Calculate Cohen's d (effect size for Welch's t-test)
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
            'interpretation': f"There is{'' if reject_null else ' not'} a statistically significant difference "
                            + f"between the two datasets ({test_name}, p = {p_value:.4f}).",
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
                    posthoc_results = HexoskinWavLoader._run_posthoc_tests(
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
                    posthoc_results = HexoskinWavLoader._run_posthoc_tests(
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
                    posthoc_results = HexoskinWavLoader._run_posthoc_tests(
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
                    posthoc_results = HexoskinWavLoader._run_posthoc_tests(
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
                    posthoc_results = HexoskinWavLoader._run_posthoc_tests(
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
                    posthoc_results = HexoskinWavLoader._run_posthoc_tests(
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
            if posthoc_results and 'significant_pairs' in posthoc_results and posthoc_results['significant_pairs']:
                sig_pairs_text = ", ".join([f"{p[0]} vs {p[1]}" for p in posthoc_results['significant_pairs'][:3]])
                if len(posthoc_results['significant_pairs']) > 3:
                    sig_pairs_text += f", and {len(posthoc_results['significant_pairs']) - 3} more"
                    
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
            'posthoc_results': posthoc_results,
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
                # Tukey's HSD test
                from scipy.stats import tukey_hsd
                
                # Run Tukey's HSD
                posthoc = tukey_hsd(*data_values)
                
                posthoc_results = {
                    'test': 'Tukey HSD',
                    'description': 'Post-hoc test for pairwise comparisons after significant ANOVA',
                    'pairwise_p_values': [],
                    'significant_pairs': []
                }
                
                # Create pairwise comparisons
                for i in range(len(data_values)):
                    for j in range(i+1, len(data_values)):
                        p_val = posthoc.pvalue[i, j]
                        mean_diff = np.mean(data_values[i]) - np.mean(data_values[j])
                        
                        # Calculate Cohen's d effect size
                        pooled_std = np.sqrt((np.std(data_values[i], ddof=1)**2 + np.std(data_values[j], ddof=1)**2) / 2)
                        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                        
                        posthoc_results['pairwise_p_values'].append({
                            'group1': data_names[i],
                            'group2': data_names[j],
                            'p_value': p_val,
                            'significant': p_val < alpha,
                            'mean_diff': mean_diff,
                            'effect_size': cohens_d,
                            'effect_size_type': "Cohen's d"
                        })
                        
                        if p_val < alpha:
                            posthoc_results['significant_pairs'].append(
                                (data_names[i], data_names[j], p_val)
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
        ttk.Label(header_frame, text="Version 0.0.2", 
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
        right_notebook = ttk.Notebook(right_frame)
        right_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Plot tab
        plot_frame = ttk.Frame(right_notebook)
        right_notebook.add(plot_frame, text="Plot")
        
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
        
        # Statistics tab
        self.stats_frame = ttk.Frame(right_notebook)
        right_notebook.add(self.stats_frame, text="Statistics")
        
        # Add scrollbar to stats text
        stats_text_frame = ttk.Frame(self.stats_frame)
        stats_text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        stats_scrollbar = ttk.Scrollbar(stats_text_frame)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.stats_text = tk.Text(stats_text_frame, font=('Consolas', 10), wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        self.stats_text.config(yscrollcommand=stats_scrollbar.set)
        stats_scrollbar.config(command=self.stats_text.yview)
        
        # Comparison tab
        self.comparison_frame = ttk.Frame(right_notebook)
        right_notebook.add(self.comparison_frame, text="Comparison")
        
        # Create PanedWindow for comparison tab
        comparison_paned = ttk.PanedWindow(self.comparison_frame, orient=tk.VERTICAL)
        comparison_paned.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable comparison plot frame
        comparison_plot_frame = ttk.Frame(comparison_paned)
        
        # Add vertical scrollbar to plot area
        comparison_scroll_frame = ttk.Frame(comparison_plot_frame)
        comparison_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        self.comparison_canvas_scroll = tk.Canvas(comparison_scroll_frame)
        self.comparison_scrollbar = ttk.Scrollbar(comparison_scroll_frame, orient=tk.VERTICAL, 
                                                 command=self.comparison_canvas_scroll.yview)
        self.comparison_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.comparison_canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.comparison_canvas_scroll.configure(yscrollcommand=self.comparison_scrollbar.set)
        
        # Create inner frame for the plot
        self.comparison_inner_frame = ttk.Frame(self.comparison_canvas_scroll)
        self.comparison_canvas_window = self.comparison_canvas_scroll.create_window(
            (0, 0), window=self.comparison_inner_frame, anchor="nw", tags="inner_frame"
        )
        
        # Bind events for proper scrolling
        self.comparison_inner_frame.bind("<Configure>", self._configure_comparison_scroll_region)
        self.comparison_canvas_scroll.bind("<Configure>", self._configure_comparison_canvas)
        
        # Create figure and canvas for comparison plots
        self.comparison_figure = Figure(figsize=(5, 8), dpi=100, constrained_layout=True)
        self.comparison_figure.patch.set_facecolor('#F8F8F8')  # Light background
        
        self.comparison_canvas_widget = FigureCanvasTkAgg(self.comparison_figure, self.comparison_inner_frame)
        self.comparison_canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add view control buttons frame
        comp_view_controls = ttk.Frame(comparison_plot_frame)
        comp_view_controls.pack(fill=tk.X, padx=5, pady=5)
        
        # Add reset view button
        comp_reset_view_btn = ttk.Button(comp_view_controls, text="Reset View/Scrollbar", 
                                      command=lambda: self._fit_comparison_to_window())
        comp_reset_view_btn.pack(side=tk.LEFT, padx=5)
        
        # Add toolbar for comparison matplotlib
        comp_toolbar_frame = ttk.Frame(comparison_plot_frame)
        comp_toolbar_frame.pack(fill=tk.X)
        self.comp_toolbar = NavigationToolbar2Tk(self.comparison_canvas_widget, comp_toolbar_frame)
        self.comp_toolbar.update()
        
        # Create subplots for comparison
        self.comp_ax1 = self.comparison_figure.add_subplot(211)  # Time series
        self.comp_ax2 = self.comparison_figure.add_subplot(212)  # Box plot
        
        # Comparison stats frame
        comparison_stats_frame = ttk.Frame(comparison_paned)
        
        # Data preparation options
        prep_options_frame = ttk.LabelFrame(comparison_stats_frame, text="Data Preparation")
        prep_options_frame.pack(fill=tk.X, pady=5)
        
        # Data alignment options
        align_frame = ttk.Frame(prep_options_frame)
        align_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.align_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(align_frame, text="Align Datasets", variable=self.align_var).pack(side=tk.LEFT)
        
        ttk.Label(align_frame, text="Target Hz:").pack(side=tk.LEFT, padx=(10, 0))
        self.target_hz_var = tk.StringVar(value="")
        target_hz_entry = ttk.Entry(align_frame, textvariable=self.target_hz_var, width=8)
        target_hz_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(align_frame, text="(leave empty for auto)").pack(side=tk.LEFT)
        
        # Normalization options
        norm_frame = ttk.Frame(prep_options_frame)
        norm_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.normalize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(norm_frame, text="Normalize Data", variable=self.normalize_var).pack(side=tk.LEFT)
        
        ttk.Label(norm_frame, text="Method:").pack(side=tk.LEFT, padx=(10, 0))
        self.norm_method_var = tk.StringVar(value="min_max")
        norm_method_combo = ttk.Combobox(norm_frame, textvariable=self.norm_method_var, 
                                       width=10, state="readonly")
        norm_method_combo['values'] = ('min_max', 'z_score', 'robust')
        norm_method_combo.pack(side=tk.LEFT, padx=5)
        
        # View controls for comparison
        comp_view_frame = ttk.Frame(prep_options_frame)
        comp_view_frame.pack(fill=tk.X, padx=5, pady=5)
        
        comp_fit_btn = ttk.Button(comp_view_frame, text="Fit Comparison to Window", 
                                command=lambda: self._fit_comparison_to_window())
        comp_fit_btn.pack(fill=tk.X, pady=2)
        
        # Comparison test options
        test_options_frame = ttk.LabelFrame(comparison_stats_frame, text="Statistical Test")
        test_options_frame.pack(fill=tk.X, pady=5)
        
        # Add test type selector
        test_frame = ttk.Frame(test_options_frame)
        test_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(test_frame, text="Test Type:").pack(side=tk.LEFT)
        self.test_type_var = tk.StringVar(value="mann_whitney")
        self.test_dropdown = ttk.Combobox(test_frame, textvariable=self.test_type_var)
        self.test_dropdown['values'] = ("mann_whitney", "t_test", "wilcoxon", "ks_test")
        self.test_dropdown.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # Add a variable for the test description
        self.test_description_var = tk.StringVar()
        
        # Define the update_test_description function
        def update_test_description(*args):
            test = self.test_type_var.get()
            descriptions = {
                'mann_whitney': "Non-parametric, two independent groups",
                'wilcoxon': "Non-parametric, paired samples",
                'ks_test': "Non-parametric, compares distributions",
                't_test': "Parametric, two independent groups with equal variance"
            }
            self.test_description_var.set(descriptions.get(test, ""))
        
        # Add the trace to update the description when the test type changes
        self.test_type_var.trace_add("write", update_test_description)
        
        # Add test description label
        self.test_description_var.set("Non-parametric, two independent groups")
        
        # Test suggestion assistant button
        suggest_test_btn = ttk.Button(test_options_frame, text="Test Suggestion Assistant", 
                                   command=lambda: messagebox.showinfo("Information", "Test suggestion functionality is not implemented yet."))
        suggest_test_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Progress bar frame
        progress_frame = ttk.Frame(test_options_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            orient=tk.HORIZONTAL, 
            length=100, 
            mode='determinate',
            variable=self.progress_var
        )
        self.progress_bar.pack(fill=tk.X, expand=True)
        
        self.progress_label = ttk.Label(progress_frame, text="", font=('Segoe UI', 8))
        self.progress_label.pack(fill=tk.X, pady=(2, 0))
        
        # Radio buttons for test selection - keep for backward compatibility
        # But update labels to show which are for 2 groups vs multiple groups
        test_2group_frame = ttk.LabelFrame(test_options_frame, text="Two Group Tests")
        test_2group_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Radiobutton(test_2group_frame, text="Mann-Whitney U Test", 
                      variable=self.test_type_var, 
                      value="mann_whitney").pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(test_2group_frame, text="Wilcoxon Signed-Rank Test", 
                      variable=self.test_type_var,
                      value="wilcoxon").pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(test_2group_frame, text="Kolmogorov-Smirnov Test", 
                      variable=self.test_type_var,
                      value="ks_2samp").pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(test_2group_frame, text="Independent t-test", 
                      variable=self.test_type_var,
                      value="t_test").pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(test_2group_frame, text="Welch's t-test", 
                      variable=self.test_type_var,
                      value="welch_t_test").pack(anchor=tk.W, padx=5)
        
        # Multiple group tests
        test_multigroup_frame = ttk.LabelFrame(test_options_frame, text="Multiple Group Tests")
        test_multigroup_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Radiobutton(test_multigroup_frame, text="One-way ANOVA (parametric)", 
                      variable=self.test_type_var,
                      value="anova").pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(test_multigroup_frame, text="Kruskal-Wallis H-test (non-parametric)", 
                      variable=self.test_type_var,
                      value="kruskal").pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(test_multigroup_frame, text="Friedman Test (paired non-parametric)", 
                      variable=self.test_type_var,
                      value="friedman").pack(anchor=tk.W, padx=5)
        
        # Run comparison button
        compare_run_btn = ttk.Button(test_options_frame, text="Run Comparison Test", 
                                  command=lambda: self._run_comparison_test(), style='Primary.TButton')
        compare_run_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Comparison results text with scrollbar
        comp_text_frame = ttk.Frame(comparison_stats_frame)
        comp_text_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        comp_scrollbar = ttk.Scrollbar(comp_text_frame)
        comp_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.comparison_text = tk.Text(comp_text_frame, font=('Consolas', 10), wrap=tk.WORD)
        self.comparison_text.pack(fill=tk.BOTH, expand=True)
        self.comparison_text.config(yscrollcommand=comp_scrollbar.set)
        comp_scrollbar.config(command=self.comparison_text.yview)
        
        # Add both frames to the paned window
        comparison_paned.add(comparison_plot_frame, weight=2)
        comparison_paned.add(comparison_stats_frame, weight=1)
        
        # About tab
        about_frame = ttk.Frame(right_notebook)
        right_notebook.add(about_frame, text="About")
        
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
        about_text.insert(tk.END, "Version 0.0.2\n\n", "center")
        
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
        
        about_text.insert(tk.END, "\nNew in Version 0.0.2\n", "heading")
        updates = [
            "Enhanced statistical analysis with comprehensive descriptive statistics",
            "Advanced normality tests including Anderson-Darling and Jarque-Bera",
            "Support for comparing up to 15 datasets simultaneously",
            "New statistical tests: Welch's ANOVA, RM-ANOVA, and Aligned Ranks Transform",
            "Improved post-hoc analysis with multiple correction methods",
            "Interactive statistical visualization with QQ plots and histograms",
            "Export capabilities for all statistical results"
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
            
            # Create a table-like format
            self.comparison_text.insert(tk.END, f"{'Statistic':15} {'File 1':15} {'File 2':15}\n")
            self.comparison_text.insert(tk.END, "-" * 45 + "\n")
            
            desc1 = result['descriptive_stats_1']
            desc2 = result['descriptive_stats_2']
            
            self.comparison_text.insert(tk.END, f"{'Count':15} {desc1['count']:15.0f} {desc2['count']:15.0f}\n")
            self.comparison_text.insert(tk.END, f"{'Mean':15} {desc1['mean']:15.4f} {desc2['mean']:15.4f}\n")
            self.comparison_text.insert(tk.END, f"{'Median':15} {desc1['median']:15.4f} {desc2['median']:15.4f}\n")
            self.comparison_text.insert(tk.END, f"{'Std Dev':15} {desc1['std']:15.4f} {desc2['std']:15.4f}\n")
            self.comparison_text.insert(tk.END, f"{'Min':15} {desc1['min']:15.4f} {desc2['min']:15.4f}\n")
            self.comparison_text.insert(tk.END, f"{'Max':15} {desc1['max']:15.4f} {desc2['max']:15.4f}\n\n")
            
            # Interpretation
            self.comparison_text.insert(tk.END, "INTERPRETATION\n")
            self.comparison_text.insert(tk.END, "-" * 50 + "\n")
            self.comparison_text.insert(tk.END, f"{result['interpretation']}\n")
            
            # Display post-hoc analysis for significant results
            if result['reject_null'] and 'post_hoc_results' in result and result['post_hoc_results']:
                post_hoc = result['post_hoc_results']
                
                self.comparison_text.insert(tk.END, "\nPOST-HOC ANALYSIS\n")
                self.comparison_text.insert(tk.END, "-" * 50 + "\n")
                
                # Display normality test results
                if 'normality' in post_hoc:
                    self.comparison_text.insert(tk.END, "Normality Assessment:\n")
                    norm1 = post_hoc['normality']['dataset1']
                    norm2 = post_hoc['normality']['dataset2']
                    
                    if 'shapiro_p' in norm1:
                        self.comparison_text.insert(tk.END, f"Dataset 1: {'Normal' if norm1['is_normal'] else 'Non-normal'} distribution (Shapiro p={norm1['shapiro_p']:.4f})\n")
                        self.comparison_text.insert(tk.END, f"Dataset 2: {'Normal' if norm2['is_normal'] else 'Non-normal'} distribution (Shapiro p={norm2['shapiro_p']:.4f})\n")
                    else:
                        self.comparison_text.insert(tk.END, f"Dataset 1: {'Normal' if norm1['is_normal'] else 'Non-normal'} distribution\n")
                        self.comparison_text.insert(tk.END, f"Dataset 2: {'Normal' if norm2['is_normal'] else 'Non-normal'} distribution\n")
                
                # Display homogeneity of variance results
                if 'homogeneity' in post_hoc:
                    homo = post_hoc['homogeneity']
                    if 'levene_p' in homo:
                        self.comparison_text.insert(tk.END, f"\nHomogeneity of Variance:\n")
                        self.comparison_text.insert(tk.END, f"Equal variances: {'Yes' if homo['equal_variance'] else 'No'} (Levene's test p={homo['levene_p']:.4f})\n")
                
                # Display bootstrap confidence intervals
                if 'bootstrap' in post_hoc:
                    boot = post_hoc['bootstrap']
                    self.comparison_text.insert(tk.END, f"\nBootstrap Analysis ({boot['bootstrap_samples']} samples):\n")
                    self.comparison_text.insert(tk.END, f"95% CI for the difference: [{boot['diff_ci_lower']:.4f}, {boot['diff_ci_upper']:.4f}]\n")
                    
                    # Interpretation of CI
                    if boot['diff_ci_lower'] > 0 and boot['diff_ci_upper'] > 0:
                        self.comparison_text.insert(tk.END, "Dataset 1 values are consistently higher than Dataset 2 values.\n")
                    elif boot['diff_ci_lower'] < 0 and boot['diff_ci_upper'] < 0:
                        self.comparison_text.insert(tk.END, "Dataset 2 values are consistently higher than Dataset 1 values.\n")
                    else:
                        self.comparison_text.insert(tk.END, "The direction of the difference is not consistent (CI includes zero).\n")
                
                # Display additional effect sizes
                if 'additional_effect_sizes' in post_hoc:
                    add_effects = post_hoc['additional_effect_sizes']
                    self.comparison_text.insert(tk.END, f"\nAdditional Effect Size Measures:\n")
                    
                    if 'cles' in add_effects:
                        cles = add_effects['cles']
                        self.comparison_text.insert(tk.END, f"Common Language Effect Size: {cles:.4f}\n")
                        self.comparison_text.insert(tk.END, f"Interpretation: There is a {cles*100:.1f}% probability that a random value from Dataset 1 exceeds a random value from Dataset 2.\n")
                
                # Display test recommendations
                if 'recommendations' in post_hoc and post_hoc['recommendations']:
                    self.comparison_text.insert(tk.END, f"\nRECOMMENDED FOLLOW-UP ACTIONS\n")
                    self.comparison_text.insert(tk.END, "-" * 50 + "\n")
                    
                    for i, rec in enumerate(post_hoc['recommendations'], 1):
                        self.comparison_text.insert(tk.END, f"{i}. {rec}\n")
            
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
        selected_index = self.file_list.curselection()
        if not selected_index:
            messagebox.showinfo("Information", "Please select a file first")
            return
            
        selected_path = self.file_list.get(selected_index)
        file_info = self.processed_files.get(selected_path)
        
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
        stats_window.title(f"Statistics: {selected_path}")
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
        notebook.add(basic_tab, text="Basic Statistics")
        
        # Basic stats table
        basic_frame = ttk.LabelFrame(basic_tab, text="Descriptive Statistics")
        basic_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a treeview for basic statistics
        basic_tree = ttk.Treeview(basic_frame, columns=("Value",), show="headings")
        basic_tree.heading("Value", text="Value")
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
                
            basic_tree.insert("", tk.END, values=(stat_name, formatted_value))
            
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
                
            basic_tree.insert("", tk.END, values=(stat_name, formatted_value))
        
        # 2. Normality Tests Tab
        normality_tab = ttk.Frame(notebook)
        notebook.add(normality_tab, text="Normality Tests")
        
        # Create a frame for normality tests
        normality_frame = ttk.LabelFrame(normality_tab, text="Normality Test Results")
        normality_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a text widget with scrollbar for displaying normality test results
        normality_text = tk.Text(normality_frame, wrap=tk.WORD, font=('Consolas', 10))
        normality_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        normality_sb = ttk.Scrollbar(normality_frame, orient=tk.VERTICAL, command=normality_text.yview)
        normality_sb.pack(side=tk.RIGHT, fill=tk.Y)
        normality_text.configure(yscrollcommand=normality_sb.set)
        
        # Populate normality test results
        normality_results = stats.get('normality', {})
        
        # Overall assessment
        overall = normality_results.get('overall_assessment', {})
        assessment = overall.get('assessment', "No assessment available")
        
        normality_text.insert(tk.END, f"OVERALL ASSESSMENT:\n{assessment}\n\n")
        
        # Add recommendations
        recommendations = overall.get('recommendation', [])
        if recommendations:
            normality_text.insert(tk.END, "RECOMMENDATIONS:\n")
            for i, rec in enumerate(recommendations, 1):
                normality_text.insert(tk.END, f"{i}. {rec}\n")
            normality_text.insert(tk.END, "\n")
        
        # Individual test results
        normality_text.insert(tk.END, "NORMALITY TESTS:\n\n")
        
        test_names = [
            'shapiro_wilk', 'dagostino_k2', 'kolmogorov_smirnov', 
            'anderson_darling', 'jarque_bera'
        ]
        
        for test_name in test_names:
            test_result = normality_results.get(test_name)
            if not test_result:
                continue
                
            if 'error' in test_result:
                normality_text.insert(tk.END, f"{test_name.replace('_', ' ').title()}: Error - {test_result['error']}\n\n")
                continue
                
            normality_text.insert(tk.END, f"{test_name.replace('_', ' ').title()}:\n")
            normality_text.insert(tk.END, f"  Description: {test_result.get('description', 'No description')}\n")
            
            if 'statistic' in test_result:
                normality_text.insert(tk.END, f"  Statistic: {test_result['statistic']:.4f}\n")
                
            if 'p_value' in test_result:
                normality_text.insert(tk.END, f"  p-value: {test_result['p_value']:.4f}\n")
                
            normal = test_result.get('normal', False)
            normality_text.insert(tk.END, f"  Normal distribution: {'Yes' if normal else 'No'}\n\n")
        
        # Add distribution shape information
        shape_info = normality_results.get('distribution_shape', {})
        if shape_info and 'error' not in shape_info:
            normality_text.insert(tk.END, "DISTRIBUTION SHAPE:\n")
            normality_text.insert(tk.END, f"  Skewness: {shape_info['skewness']:.4f}\n")
            normality_text.insert(tk.END, f"  Skewness interpretation: {shape_info['skewness_interpretation']}\n")
            normality_text.insert(tk.END, f"  Kurtosis: {shape_info['kurtosis']:.4f}\n")
            normality_text.insert(tk.END, f"  Kurtosis interpretation: {shape_info['kurtosis_interpretation']}\n\n")
        
        # 3. QQ Plot Tab (if data available)
        qq_plot_data = normality_results.get('qq_plot_data')
        if qq_plot_data and 'error' not in qq_plot_data:
            qq_tab = ttk.Frame(notebook)
            notebook.add(qq_tab, text="QQ Plot")
            
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
            
            ax.set_title('Q-Q Plot')
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel('Sample Quantiles')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Embed the plot in the tkinter window
            canvas = FigureCanvasTkAgg(fig, master=qq_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar_frame = ttk.Frame(qq_frame)
            toolbar_frame.pack(fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
        
        # 4. Histogram with Normal Distribution Tab
        hist_tab = ttk.Frame(notebook)
        notebook.add(hist_tab, text="Histogram")
        
        # Create a frame for the histogram
        hist_frame = ttk.Frame(hist_tab)
        hist_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a matplotlib figure
        fig = Figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Get the data
        data = loader.data['value']
        
        # Plot histogram
        n, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.6, color='blue')
        
        # Add normal distribution curve
        mu = np.mean(data)
        sigma = np.std(data)
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
        
        ax.set_title('Histogram with Normal Distribution Curve')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Embed the plot in the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=hist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(hist_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        # 5. Export Button
        export_frame = ttk.Frame(main_frame)
        export_frame.pack(fill=tk.X, pady=10)
        
        export_btn = ttk.Button(export_frame, text="Export Statistics to CSV", 
                               command=lambda: self._export_statistics(selected_path, stats))
        export_btn.pack(side=tk.RIGHT, padx=5)
    
    def _export_statistics(self, filename, stats):
        """Export statistics to a CSV file"""
        if not stats:
            messagebox.showerror("Error", "No statistics available to export")
            return
            
        # Ask for save location
        save_path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV Files', '*.csv')],
            initialfile=f"{os.path.basename(filename)}_statistics.csv"
        )
        
        if not save_path:
            return  # User canceled
            
        try:
            with open(save_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Statistic', 'Value'])
                
                # Write basic statistics
                writer.writerow(['=== BASIC STATISTICS ===', ''])
                for stat_name, stat_value in stats.get('basic', {}).items():
                    writer.writerow([stat_name, stat_value])
                
                # Write additional statistics
                writer.writerow(['=== ADDITIONAL STATISTICS ===', ''])
                for stat_name, stat_value in stats.get('additional', {}).items():
                    if stat_name == 'percentiles':
                        for p_name, p_value in stat_value.items():
                            writer.writerow([f"percentile_{p_name}", p_value])
                    elif isinstance(stat_value, (list, dict)):
                        continue  # Skip complex objects
                    else:
                        writer.writerow([stat_name, stat_value])
                
                # Write normality test results
                writer.writerow(['=== NORMALITY TESTS ===', ''])
                for test_name in ['shapiro_wilk', 'dagostino_k2', 'kolmogorov_smirnov', 'anderson_darling', 'jarque_bera']:
                    test_result = stats.get('normality', {}).get(test_name, {})
                    if 'error' in test_result:
                        writer.writerow([test_name, f"Error: {test_result['error']}"])
                        continue
                        
                    if 'statistic' in test_result:
                        writer.writerow([f"{test_name}_statistic", test_result['statistic']])
                    if 'p_value' in test_result:
                        writer.writerow([f"{test_name}_p_value", test_result['p_value']])
                    if 'normal' in test_result:
                        writer.writerow([f"{test_name}_normal", test_result['normal']])
                
                # Write overall assessment
                overall = stats.get('normality', {}).get('overall_assessment', {})
                if overall:
                    writer.writerow(['=== OVERALL ASSESSMENT ===', ''])
                    writer.writerow(['assessment', overall.get('assessment', '')])
                    
                    # Write recommendations
                    recommendations = overall.get('recommendation', [])
                    for i, rec in enumerate(recommendations, 1):
                        writer.writerow([f'recommendation_{i}', rec])
            
            messagebox.showinfo("Success", f"Statistics exported to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export statistics: {str(e)}")


def main():
    """Run the application"""
    app = HexoskinWavApp()
    app.mainloop()


if __name__ == "__main__":
    main() 