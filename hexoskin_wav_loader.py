import os
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import wave
import scipy.signal as signal
import scipy.stats as stats
import scipy.interpolate
import json
import datetime
import platform

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
            pandas.Series: Descriptive statistics
        """
        if self.data is None:
            print("No data loaded")
            return None
        
        return self.data['value'].describe()

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
        
        # Handle datasets of different lengths
        if len(data1) != len(data2) and test_type == 'wilcoxon':
            raise ValueError("Wilcoxon signed-rank test requires datasets of equal length")
        
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
        
        # Interpret effect size
        if abs(effect_size) < 0.3:
            effect_interpretation = "Small effect"
        elif abs(effect_size) < 0.5:
            effect_interpretation = "Medium effect"
        else:
            effect_interpretation = "Large effect"
            
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
            'interpretation': "There is " + ("" if reject_null else "not ") + 
                           "a statistically significant difference between the two datasets " +
                           f"({test_name}, p = {p_value:.4f})."
        }
    
    def test_normality(self):
        """
        Test the normality of the data distribution
        
        Returns:
            dict: Results of normality tests
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
            
        # Shapiro-Wilk test (best for sample sizes < 5000)
        shapiro_test = stats.shapiro(data_sample)
        
        # D'Agostino's K-squared test
        dagostino_test = stats.normaltest(data_sample)
        
        # Kolmogorov-Smirnov test against normal distribution
        ks_test = stats.kstest(
            data_sample, 
            'norm', 
            args=(np.mean(data_sample), np.std(data_sample, ddof=1))
        )
        
        # Calculate skewness and kurtosis
        skewness = stats.skew(data_sample)
        kurtosis = stats.kurtosis(data_sample)
        
        return {
            'shapiro_wilk': {
                'statistic': shapiro_test[0],
                'p_value': shapiro_test[1],
                'normal': shapiro_test[1] > 0.05
            },
            'dagostino_k2': {
                'statistic': dagostino_test[0],
                'p_value': dagostino_test[1],
                'normal': dagostino_test[1] > 0.05
            },
            'kolmogorov_smirnov': {
                'statistic': ks_test[0],
                'p_value': ks_test[1],
                'normal': ks_test[1] > 0.05
            },
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
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
        ttk.Label(header_frame, text="Version 0.0.1", 
                font=('Segoe UI', 9)).pack(side=tk.TOP)
        
        # File operations section
        file_section = ttk.LabelFrame(left_frame, text="File Operations")
        file_section.pack(fill=tk.X, pady=5)
        
        # Button frame with grid layout for file operations
        file_btn_frame = ttk.Frame(file_section)
        file_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Load file button
        load_btn = ttk.Button(file_btn_frame, text="Load WAV File", 
                            command=self._load_file, style='Primary.TButton')
        load_btn.grid(column=0, row=0, sticky=tk.EW, padx=2, pady=2)
        
        # Batch load button
        batch_load_btn = ttk.Button(file_btn_frame, text="Batch Load Files", 
                                  command=self._batch_load_files)
        batch_load_btn.grid(column=1, row=0, sticky=tk.EW, padx=2, pady=2)
        
        # Save to CSV button
        save_csv_btn = ttk.Button(file_btn_frame, text="Save to CSV", 
                                command=self._save_to_csv)
        save_csv_btn.grid(column=0, row=1, sticky=tk.EW, padx=2, pady=2)
        
        # Export graph button
        export_graph_btn = ttk.Button(file_btn_frame, text="Export Graph", 
                                    command=self._export_graph)
        export_graph_btn.grid(column=1, row=1, sticky=tk.EW, padx=2, pady=2)
        
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
                               command=self._compare_selected_files)
        compare_btn.pack(fill=tk.X, padx=5, pady=5)
        
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
                                command=self._resample_data, width=8)
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
                              command=self._filter_data)
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
                                 command=self._update_plot)
        customize_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # View control buttons
        view_frame = ttk.LabelFrame(left_frame, text="View Controls")
        view_frame.pack(fill=tk.X, pady=10)
        
        # Auto-fit button
        fit_btn = ttk.Button(view_frame, text="Fit to Window", 
                           command=self._fit_to_window)
        fit_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Zoom controls
        zoom_frame = ttk.Frame(view_frame)
        zoom_frame.pack(fill=tk.X, padx=5, pady=5)
        
        zoom_in_btn = ttk.Button(zoom_frame, text="Zoom In", command=self._zoom_in)
        zoom_in_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        
        zoom_out_btn = ttk.Button(zoom_frame, text="Zoom Out", command=self._zoom_out)
        zoom_out_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(2, 0))
        
        # Reset view button
        reset_view_btn = ttk.Button(view_frame, text="Reset View", 
                                  command=self._reset_view)
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
        
        # Comparison plot frame
        comparison_plot_frame = ttk.Frame(comparison_paned)
        
        # Create figure and canvas for comparison plots
        self.comparison_figure = Figure(figsize=(5, 8), dpi=100, constrained_layout=True)
        self.comparison_figure.patch.set_facecolor('#F8F8F8')  # Light background
        
        self.comparison_canvas = FigureCanvasTkAgg(self.comparison_figure, comparison_plot_frame)
        self.comparison_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar for comparison matplotlib
        comp_toolbar_frame = ttk.Frame(comparison_plot_frame)
        comp_toolbar_frame.pack(fill=tk.X)
        self.comp_toolbar = NavigationToolbar2Tk(self.comparison_canvas, comp_toolbar_frame)
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
                                command=self._fit_comparison_to_window)
        comp_fit_btn.pack(fill=tk.X, pady=2)
        
        # Comparison test options
        test_options_frame = ttk.LabelFrame(comparison_stats_frame, text="Statistical Test")
        test_options_frame.pack(fill=tk.X, pady=5)
        
        # Radio buttons for test selection
        self.test_type_var = tk.StringVar(value="mann_whitney")
        ttk.Radiobutton(test_options_frame, text="Mann-Whitney U Test", 
                      variable=self.test_type_var, 
                      value="mann_whitney").pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(test_options_frame, text="Wilcoxon Signed-Rank Test", 
                      variable=self.test_type_var,
                      value="wilcoxon").pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(test_options_frame, text="Kolmogorov-Smirnov Test", 
                      variable=self.test_type_var,
                      value="ks_2samp").pack(anchor=tk.W, padx=5)
        
        # Run comparison button
        compare_run_btn = ttk.Button(test_options_frame, text="Run Comparison Test", 
                                  command=self._run_comparison_test, style='Primary.TButton')
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
        about_text.insert(tk.END, "Version 0.0.1\n\n", "center")
        
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
            "Comprehensive statistical analysis with normality testing",
            "Non-parametric comparison between multiple datasets",
            "Intelligent data alignment and normalization algorithms",
            "Flexible export options for research publications"
        ]
        
        for feature in features:
            about_text.insert(tk.END, f"• {feature}\n", "bullet")
        
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
            self._show_statistics()
    
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
        
        # Clear the comparison plots
        self.comp_ax1.clear()
        self.comp_ax2.clear()
        
        # Process all selected files
        processed_data = []
        
        # Apply preprocessing to all datasets
        for i, file_info in enumerate(self.selected_files_for_comparison):
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
            
            # Plot time series
            self.comp_ax1.plot(x_data, data['value'], 
                             color=color, 
                             label=label,
                             alpha=0.8)
        
        # Set time series plot properties
        if not self.comp_ax1.get_title():  # If title not already set by alignment
            self.comp_ax1.set_title('Time Series Comparison')
        
        # Format x-axis if using real dates
        if time_unit == 'days' and any('datetime' in data_item['data'].columns for data_item in processed_data):
            self.comparison_figure.autofmt_xdate()
            
        self.comp_ax1.set_xlabel(xlabel)
        self.comp_ax1.set_ylabel('Value')
        self.comp_ax1.grid(True)
        self.comp_ax1.legend()
        
        # Create box plot data
        data_for_boxplot = []
        labels_for_boxplot = []
        
        for file_data in processed_data:
            data_for_boxplot.append(file_data['data']['value'])
            
            if file_data.get('start_date'):
                labels_for_boxplot.append(f"{file_data['name']}\n{file_data['start_date']}")
            else:
                labels_for_boxplot.append(file_data['name'])
        
        # Create boxplot
        self.comp_ax2.boxplot(data_for_boxplot, labels=labels_for_boxplot)
        norm_status = " (Normalized)" if self.normalize_var.get() else ""
        self.comp_ax2.set_title(f'Distribution Comparison{norm_status}')
        self.comp_ax2.set_ylabel('Value')
        self.comp_ax2.grid(True)
        
        # Update the figure
        self.comparison_figure.tight_layout()
        self.comparison_canvas.draw()
        
        # If only two files are selected, run the comparison test
        if len(self.selected_files_for_comparison) == 2:
            self._run_comparison_test()
    
    def _run_comparison_test(self):
        """Run statistical test to compare selected files"""
        if len(self.selected_files_for_comparison) != 2:
            messagebox.showinfo("Information", "Please select exactly two files for statistical testing")
            return
        
        # Get the two selected files
        file1 = self.selected_files_for_comparison[0]
        file2 = self.selected_files_for_comparison[1]
        
        # Get the test type
        test_type = self.test_type_var.get()
        
        try:
            # Process data with alignment and normalization as needed
            data1 = file1['loader'].get_data().copy()
            data2 = file2['loader'].get_data().copy()
            
            # Apply normalization if enabled
            if self.normalize_var.get():
                data1 = HexoskinWavLoader.normalize_dataset(
                    data1, 
                    method=self.norm_method_var.get()
                )
                data2 = HexoskinWavLoader.normalize_dataset(
                    data2, 
                    method=self.norm_method_var.get()
                )
            
            # Apply alignment if enabled
            if self.align_var.get():
                target_hz = None
                if self.target_hz_var.get():
                    try:
                        target_hz = float(self.target_hz_var.get())
                    except ValueError:
                        pass
                
                data1, data2 = HexoskinWavLoader.align_datasets(data1, data2, target_hz=target_hz)
                
                # Check if alignment was successful
                if len(data1) == 0 or len(data2) == 0:
                    messagebox.showinfo("Warning", "Datasets could not be aligned - no overlapping time range")
                    return
            
            # Run the comparison
            result = HexoskinWavLoader.compare_datasets(data1, data2, test_type=test_type)
            
            # Display the results
            self.comparison_text.delete(1.0, tk.END)
            
            # Header
            self.comparison_text.insert(tk.END, f"Comparison of Files\n")
            self.comparison_text.insert(tk.END, f"1: {file1['name']}\n")
            self.comparison_text.insert(tk.END, f"2: {file2['name']}\n")
            self.comparison_text.insert(tk.END, "=" * 50 + "\n\n")
            
            # Preprocessing information
            self.comparison_text.insert(tk.END, "PREPROCESSING\n")
            self.comparison_text.insert(tk.END, "-" * 50 + "\n")
            
            if self.align_var.get():
                target_hz_str = self.target_hz_var.get() or "auto"
                self.comparison_text.insert(tk.END, f"Alignment: Enabled (Target Hz: {target_hz_str})\n")
                aligned_length = len(data1)
                self.comparison_text.insert(tk.END, f"Aligned dataset length: {aligned_length} points\n")
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
            
            # Recommendations
            self.comparison_text.insert(tk.END, "\nRECOMMENDATIONS\n")
            self.comparison_text.insert(tk.END, "-" * 50 + "\n")
            
            if result['reject_null']:
                self.comparison_text.insert(tk.END, "Since there is a statistically significant difference, you may want to:\n")
                self.comparison_text.insert(tk.END, "1. Investigate the factors that might be causing these differences\n")
                self.comparison_text.insert(tk.END, "2. Consider normalizing the data if comparing across different conditions\n")
                self.comparison_text.insert(tk.END, "3. Check for potential outliers or data collection issues\n")
            else:
                self.comparison_text.insert(tk.END, "Since there is no statistically significant difference, you may want to:\n")
                self.comparison_text.insert(tk.END, "1. Consider combining the datasets for increased statistical power\n")
                self.comparison_text.insert(tk.END, "2. Check if this result aligns with your expectations or hypotheses\n")
                self.comparison_text.insert(tk.END, "3. Consider if the sample size is adequate to detect meaningful differences\n")
            
        except Exception as e:
            self.comparison_text.delete(1.0, tk.END)
            self.comparison_text.insert(tk.END, f"Error performing comparison: {str(e)}\n\n")
            
            if test_type == 'wilcoxon' and not self.align_var.get():
                self.comparison_text.insert(tk.END, "Note: Wilcoxon signed-rank test requires datasets of equal length.\n")
                self.comparison_text.insert(tk.END, "Enable the 'Align Datasets' option to make the lengths equal or use a different test.\n")
    
    def _save_to_csv(self):
        """Save the currently selected file to CSV"""
        selection = self.files_listbox.curselection()
        
        if not selection:
            messagebox.showinfo("Information", "Please select a file first")
            return
        
        index = selection[0]
        file_info = self.loaded_files[index]
        
        output_path = filedialog.asksaveasfilename(
            title="Save to CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=os.path.splitext(file_info['name'])[0] + ".csv"
        )
        
        if output_path:
            file_info['loader'].save_to_csv(output_path)
            messagebox.showinfo("Success", f"Data saved to {output_path}")
    
    def _export_graph(self):
        """Export the current graph to a PNG file"""
        # Check if we're on the comparison tab or the main plot tab
        notebook = self.winfo_children()[0].winfo_children()[0].winfo_children()[1].winfo_children()[0]
        current_tab = notebook.index(notebook.select())
        
        if current_tab == 0:  # Main plot tab
            fig = self.figure
            if not self.files_listbox.curselection():
                messagebox.showinfo("Information", "Please select a file first")
                return
                
            index = self.files_listbox.curselection()[0]
            file_info = self.loaded_files[index]
            default_name = os.path.splitext(file_info['name'])[0] + "_graph.png"
            
        elif current_tab == 2:  # Comparison tab
            fig = self.comparison_figure
            if len(self.selected_files_for_comparison) < 2:
                messagebox.showinfo("Information", "Please select at least two files to compare")
                return
                
            default_name = "comparison_graph.png"
            
        else:
            # Statistics tab doesn't have graphs to export
            messagebox.showinfo("Information", "No graph to export in the current tab")
            return
        
        output_path = filedialog.asksaveasfilename(
            title="Export Graph to PNG",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile=default_name
        )
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Graph exported to {output_path}")
    
    def _show_statistics(self):
        """Calculate and display statistics for the current file"""
        selection = self.files_listbox.curselection()
        
        if not selection:
            messagebox.showinfo("Information", "Please select a file first")
            return
        
        index = selection[0]
        file_info = self.loaded_files[index]
        
        # Clear the stats text
        self.stats_text.delete(1.0, tk.END)
        
        # Add file info header
        self.stats_text.insert(tk.END, f"Statistics for {file_info['name']}\n")
        self.stats_text.insert(tk.END, "=" * 50 + "\n\n")
        
        # Descriptive statistics
        self.stats_text.insert(tk.END, "DESCRIPTIVE STATISTICS\n")
        self.stats_text.insert(tk.END, "-" * 50 + "\n")
        
        desc_stats = file_info['loader'].get_descriptive_stats()
        if desc_stats is not None:
            self.stats_text.insert(tk.END, f"Count:          {desc_stats['count']:.0f}\n")
            self.stats_text.insert(tk.END, f"Mean:           {desc_stats['mean']:.4f}\n")
            self.stats_text.insert(tk.END, f"Standard Dev:   {desc_stats['std']:.4f}\n")
            self.stats_text.insert(tk.END, f"Minimum:        {desc_stats['min']:.4f}\n")
            self.stats_text.insert(tk.END, f"25th Percentile: {desc_stats['25%']:.4f}\n")
            self.stats_text.insert(tk.END, f"Median:         {desc_stats['50%']:.4f}\n")
            self.stats_text.insert(tk.END, f"75th Percentile: {desc_stats['75%']:.4f}\n")
            self.stats_text.insert(tk.END, f"Maximum:        {desc_stats['max']:.4f}\n")
        
        # Normality tests
        self.stats_text.insert(tk.END, "\n\nNORMALITY TESTS\n")
        self.stats_text.insert(tk.END, "-" * 50 + "\n")
        
        norm_tests = file_info['loader'].test_normality()
        if norm_tests is not None:
            # Shapiro-Wilk test
            shapiro = norm_tests['shapiro_wilk']
            self.stats_text.insert(tk.END, "Shapiro-Wilk Test:\n")
            self.stats_text.insert(tk.END, f"  Statistic: {shapiro['statistic']:.6f}\n")
            self.stats_text.insert(tk.END, f"  p-value:   {shapiro['p_value']:.6f}\n")
            self.stats_text.insert(tk.END, f"  Normal:    {'Yes' if shapiro['normal'] else 'No'}\n\n")
            
            # D'Agostino's K-squared test
            dagostino = norm_tests['dagostino_k2']
            self.stats_text.insert(tk.END, "D'Agostino's K^2 Test:\n")
            self.stats_text.insert(tk.END, f"  Statistic: {dagostino['statistic']:.6f}\n")
            self.stats_text.insert(tk.END, f"  p-value:   {dagostino['p_value']:.6f}\n")
            self.stats_text.insert(tk.END, f"  Normal:    {'Yes' if dagostino['normal'] else 'No'}\n\n")
            
            # Kolmogorov-Smirnov test
            ks = norm_tests['kolmogorov_smirnov']
            self.stats_text.insert(tk.END, "Kolmogorov-Smirnov Test:\n")
            self.stats_text.insert(tk.END, f"  Statistic: {ks['statistic']:.6f}\n")
            self.stats_text.insert(tk.END, f"  p-value:   {ks['p_value']:.6f}\n")
            self.stats_text.insert(tk.END, f"  Normal:    {'Yes' if ks['normal'] else 'No'}\n\n")
            
            # Skewness and Kurtosis
            self.stats_text.insert(tk.END, "Distribution Shape:\n")
            self.stats_text.insert(tk.END, f"  Skewness:  {norm_tests['skewness']:.6f}\n")
            self.stats_text.insert(tk.END, f"  Kurtosis:  {norm_tests['kurtosis']:.6f}\n")
        
        # Overall assessment
        self.stats_text.insert(tk.END, "\n\nOVERALL ASSESSMENT\n")
        self.stats_text.insert(tk.END, "-" * 50 + "\n")
        
        if norm_tests is not None:
            # Count how many tests indicate normality
            normal_count = sum([
                1 if norm_tests['shapiro_wilk']['normal'] else 0,
                1 if norm_tests['dagostino_k2']['normal'] else 0,
                1 if norm_tests['kolmogorov_smirnov']['normal'] else 0
            ])
            
            if normal_count >= 2:
                assessment = "The data appears to be normally distributed."
                rec_tests = "Parametric tests (t-test, ANOVA, Pearson correlation) are appropriate."
            else:
                assessment = "The data does not appear to be normally distributed."
                rec_tests = "Non-parametric tests (Mann-Whitney U, Kruskal-Wallis, Spearman correlation) are recommended."
            
            self.stats_text.insert(tk.END, f"{assessment}\n{rec_tests}\n")
    
    def _resample_data(self):
        """Resample the currently selected file"""
        selection = self.files_listbox.curselection()
        
        if not selection:
            messagebox.showinfo("Information", "Please select a file first")
            return
        
        try:
            target_hz = float(self.resample_var.get())
            if target_hz <= 0:
                raise ValueError("Sampling rate must be positive")
                
            index = selection[0]
            file_info = self.loaded_files[index]
            
            file_info['loader'].resample_data(target_hz)
            
            # Update plot
            self._update_plot()
            
            # If statistics tab is visible, update statistics
            if hasattr(self, 'stats_text') and self.stats_text.winfo_viewable():
                self._show_statistics()
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
    
    def _filter_data(self):
        """Apply filter to the currently selected file"""
        selection = self.files_listbox.curselection()
        
        if not selection:
            messagebox.showinfo("Information", "Please select a file first")
            return
        
        try:
            # Get filter parameters
            lowcut = None if not self.lowcut_var.get() else float(self.lowcut_var.get())
            highcut = None if not self.highcut_var.get() else float(self.highcut_var.get())
            
            if lowcut is None and highcut is None:
                messagebox.showinfo("Information", "Please specify at least one filter cutoff frequency")
                return
                
            index = selection[0]
            file_info = self.loaded_files[index]
            
            file_info['loader'].filter_data(lowcut, highcut)
            
            # Update plot
            self._update_plot()
            
            # If statistics tab is visible, update statistics
            if hasattr(self, 'stats_text') and self.stats_text.winfo_viewable():
                self._show_statistics()
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def _fit_comparison_to_window(self):
        """Fit the comparison plot to the current window"""
        if not hasattr(self, 'comp_ax1') or len(self.selected_files_for_comparison) < 1:
            return
            
        # Get the time unit
        time_unit = self.time_unit_var.get()
        
        # Collect x and y data from all selected files
        x_mins, x_maxs = [], []
        y_mins, y_maxs = [], []
        has_datetime = False
        
        for file_info in self.selected_files_for_comparison:
            data = file_info['loader'].get_data().copy()
            
            # Handle the x-axis based on available data and time unit
            has_real_timestamps = 'datetime' in data.columns
            
            if has_real_timestamps:
                if time_unit == 'days':
                    has_datetime = True
                    dt_min = data['datetime'].min()
                    dt_max = data['datetime'].max()
                    x_mins.append(dt_min)
                    x_maxs.append(dt_max)
                else:
                    # For numerical timestamps
                    if time_unit == 'seconds':
                        x_data = data['abs_timestamp'] - data['abs_timestamp'].min()
                    elif time_unit == 'minutes':
                        x_data = (data['abs_timestamp'] - data['abs_timestamp'].min()) / 60.0
                    elif time_unit == 'hours':
                        x_data = (data['abs_timestamp'] - data['abs_timestamp'].min()) / 3600.0
                    x_mins.append(x_data.min())
                    x_maxs.append(x_data.max())
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
                x_mins.append(x_data.min())
                x_maxs.append(x_data.max())
            
            # Always collect y-range
            y_mins.append(data['value'].min())
            y_maxs.append(data['value'].max())
        
        # Set the limits
        if has_datetime and time_unit == 'days':
            # For datetime objects
            self.comp_ax1.set_xlim(min(x_mins), max(x_maxs))
        else:
            # For numerical data
            x_min, x_max = min(x_mins), max(x_maxs)
            x_margin = (x_max - x_min) * 0.05
            self.comp_ax1.set_xlim(x_min - x_margin, x_max + x_margin)
        
        # Set y limits
        y_min, y_max = min(y_mins), max(y_maxs)
        y_margin = (y_max - y_min) * 0.05
        
        # Apply to both axes
        self.comp_ax1.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # For the boxplot, we don't adjust the x-axis
        self.comp_ax2.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Update the canvas
        self.comparison_canvas.draw()


def main():
    """Run the application"""
    app = HexoskinWavApp()
    app.mainloop()


if __name__ == "__main__":
    main() 