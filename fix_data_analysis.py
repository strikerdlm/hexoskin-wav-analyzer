import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import scipy.stats as stats
# %matplotlib inline - uncomment this line when running in Jupyter

# Set better visualization defaults
plt.style.use('ggplot')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = [12, 8]

# Clean column names function - enhanced to handle special characters
def clean_column_name(col_name):
    if '(' in col_name and ')' in col_name:
        # Extract the part before the first opening parenthesis that contains a URL
        clean_name = col_name.split('(/api')[0].strip()
        # Further clean any remaining special characters
        clean_name = clean_name.replace('[', '').replace(']', '').replace(' ', '_').replace('/', '_')
        return clean_name
    return col_name.replace('[', '').replace(']', '').replace(' ', '_').replace('/', '_')

# Load both datasets
file_path1 = r"C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\Sol 2 (completo)\T01_Mara\record_4494.csv"
file_path2 = r"C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\Sol 3 (completo)\T01_Mara\record-4502.csv"

# Load first dataset
df1 = pd.read_csv(file_path1)
df1['source'] = 'Day 1 (Sol 2)'

# Load second dataset 
df2 = pd.read_csv(file_path2)
df2['source'] = 'Day 2 (Sol 3)'

# Print basic info about both datasets
print(f"Dataset 1: {df1.shape[0]} rows, {df1.shape[1]} columns")
print(f"Dataset 2: {df2.shape[0]} rows, {df2.shape[1]} columns")

# Create mapping dictionary for original to clean column names
orig_columns1 = list(df1.columns)
clean_columns1 = [clean_column_name(col) for col in orig_columns1]
column_mapping1 = dict(zip(orig_columns1, clean_columns1))

# Print column mapping for reference
print("\nColumn name mapping:")
for orig, clean in column_mapping1.items():
    if orig != clean and 'source' not in orig:
        print(f"- '{orig}' → '{clean}'")

# Apply clean column names
df1.columns = clean_columns1
df2.columns = [clean_column_name(col) for col in df2.columns]

# Check if column names match between datasets
common_cols = set(df1.columns).intersection(set(df2.columns))
print(f"\nCommon columns between datasets: {len(common_cols)}")
print(f"Columns only in dataset 1: {set(df1.columns) - common_cols}")
print(f"Columns only in dataset 2: {set(df2.columns) - common_cols}")

# Convert timestamps to datetime for both datasets
df1['datetime'] = pd.to_datetime(df1['time_s_1000'], unit='ms')
df2['datetime'] = pd.to_datetime(df2['time_s_1000'], unit='ms')

# Display time spans for both datasets
print("\nDataset 1 Timespan:")
print(f"Start: {df1['datetime'].min()}")
print(f"End: {df1['datetime'].max()}")
print(f"Duration: {df1['datetime'].max() - df1['datetime'].min()}")

print("\nDataset 2 Timespan:")
print(f"Start: {df2['datetime'].min()}")
print(f"End: {df2['datetime'].max()}")
print(f"Duration: {df2['datetime'].max() - df2['datetime'].min()}")

# Check for time overlap
time_gap = df2['datetime'].min() - df1['datetime'].max()
print(f"\nTime gap between datasets: {time_gap}")

# Combine datasets for full timeline analysis
combined_df = pd.concat([df1, df2], ignore_index=True)
combined_df = combined_df.sort_values('datetime')

print(f"\nCombined dataset: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
print(f"Total duration: {combined_df['datetime'].max() - combined_df['datetime'].min()}")

# Analyze sampling frequency and consistency
combined_df['time_diff'] = combined_df['datetime'].diff().dt.total_seconds()

# Statistics about sampling intervals
print("\nSampling interval statistics (seconds):")
print(combined_df['time_diff'].describe())

# Plot sampling frequency distribution
plt.figure(figsize=(12, 6))
# Remove potential outliers for better visualization (limit to 99th percentile)
max_interval = combined_df['time_diff'].quantile(0.99)
try:
    import missingno as msno
    plt.figure(figsize=(15, 10))
    msno.matrix(combined_df.sample(min(1000, len(combined_df))))
    plt.title('Missing Data Pattern (Sample)')
    plt.tight_layout()
    plt.show()
except ImportError:
    print("missingno not installed. Skip missing data visualization.")

# Make a clean list of key metrics using the exact column names that exist in the DataFrame
# First identify important metrics by partial matching
important_metrics = ['heart_rate', 'SPO2', 'breathing_rate', 'systolic_pressure', 'temperature_celcius']
key_metrics = []

# Find the actual column names that match these important metrics
for col in combined_df.columns:
    for metric in important_metrics:
        if metric in col and col not in key_metrics:
            key_metrics.append(col)
            break

print("\nIdentified key metrics in the DataFrame:")
for metric in key_metrics:
    print(f"- {metric}")

# ======= SIMPLIFIED HOURLY AGGREGATION APPROACH =======
# Create a simpler hourly aggregation that avoids the MultiIndex complexity
hourly_stats = []

# Loop through each unique hour in the dataset
for source_name in combined_df['source'].unique():
    source_data = combined_df[combined_df['source'] == source_name]
    
    # Create hourly bins
    source_data['hour_bin'] = source_data['datetime'].dt.floor('H')
    
    # Group by hour and calculate statistics
    for hour_bin, hour_data in source_data.groupby('hour_bin'):
        hour_stats = {'source': source_name, 'datetime': hour_bin}
        
        # Calculate statistics for each metric
        for metric in key_metrics:
            values = hour_data[metric].dropna()
            if len(values) > 0:
                hour_stats[f"{metric}_mean"] = values.mean()
                hour_stats[f"{metric}_std"] = values.std()
                hour_stats[f"{metric}_count"] = len(values)
                hour_stats[f"{metric}_min"] = values.min()
                hour_stats[f"{metric}_max"] = values.max()
        
        hourly_stats.append(hour_stats)

# Convert to DataFrame
hourly_data = pd.DataFrame(hourly_stats)

# Sort by time
hourly_data = hourly_data.sort_values('datetime')

# Print column names to verify structure
print("\nHourly aggregation column examples:")
for col in list(hourly_data.columns)[:10]:  # Show first 10 columns
    print(f"- {col}")

# Create function to plot hourly statistics for any metric
def plot_hourly_metric(metric_base, title=None, ylabel=None):
    """
    Plot hourly statistics for a given metric base name
    metric_base: the base part of the column name (e.g., 'heart_rate')
    """
    # Find the full column names containing this base metric
    mean_col = f"{metric_base}_mean"
    std_col = f"{metric_base}_std"
    
    if mean_col not in hourly_data.columns:
        # Try to find partial matches
        mean_cols = [col for col in hourly_data.columns if metric_base in col and col.endswith('_mean')]
        if mean_cols:
            mean_col = mean_cols[0]
            std_col = mean_col.replace('_mean', '_std')
        else:
            print(f"Error: No columns containing '{metric_base}' and ending with '_mean' found.")
            print("Available columns:", sorted([c for c in hourly_data.columns if c.endswith('_mean')]))
            return
    
    if title is None:
        title = f'Hourly {metric_base.replace("_", " ").title()} Statistics'
    
    if ylabel is None:
        ylabel = metric_base.replace("_", " ").title()
    
    plt.figure(figsize=(15, 10))
    
    for source in hourly_data['source'].unique():
        source_data = hourly_data[hourly_data['source'] == source]
        
        if len(source_data) > 0 and mean_col in source_data.columns:
            # Only plot if the column exists and has data
            plt.plot(source_data['datetime'], source_data[mean_col], 'o-', label=f"{source} Mean")
            
            # Add standard deviation ribbon if available
            if std_col in source_data.columns:
                plt.fill_between(
                    source_data['datetime'],
                    source_data[mean_col] - source_data[std_col],
                    source_data[mean_col] + source_data[std_col],
                    alpha=0.2
                )
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot statistics for key physiological metrics - using the exact column names
print("\nPlotting hourly metrics...")
for metric in key_metrics:
    print(f"Plotting {metric}...")
    plot_hourly_metric(metric)

# Analyze day/night differences (assuming day: 7AM-7PM, night: 7PM-7AM)
combined_df['hour'] = combined_df['datetime'].dt.hour
combined_df['is_night'] = (combined_df['hour'] >= 19) | (combined_df['hour'] < 7)

# Create a custom function for day/night statistics
def calculate_day_night_stats():
    day_night_results = []
    
    for source_name in combined_df['source'].unique():
        for is_night_val in [True, False]:
            # Filter data
            filtered_data = combined_df[
                (combined_df['source'] == source_name) & 
                (combined_df['is_night'] == is_night_val)
            ]
            
            # Calculate statistics for each metric
            stats_row = {
                'source': source_name,
                'is_night': is_night_val
            }
            
            for metric in key_metrics:
                values = filtered_data[metric].dropna()
                if len(values) > 0:
                    stats_row[f"{metric}_mean"] = values.mean()
                    stats_row[f"{metric}_std"] = values.std()
                    stats_row[f"{metric}_count"] = len(values)
                    stats_row[f"{metric}_min"] = values.min()
                    stats_row[f"{metric}_max"] = values.max()
            
            day_night_results.append(stats_row)
    
    return pd.DataFrame(day_night_results)

# Calculate day/night statistics
day_night_stats = calculate_day_night_stats()

print("\nDay vs Night Statistics:")
print(day_night_stats[['source', 'is_night'] + [f"{m}_mean" for m in key_metrics]])

# Statistical significance tests between day and night
for metric in key_metrics:
    print(f"\nT-test for {metric} (Day vs Night):")
    for source in combined_df['source'].unique():
        source_data = combined_df[combined_df['source'] == source]
        day_data = source_data[~source_data['is_night']][metric].dropna()
        night_data = source_data[source_data['is_night']][metric].dropna()
        
        if len(day_data) > 0 and len(night_data) > 0:
            t_stat, p_val = stats.ttest_ind(day_data, night_data, equal_var=False)
            print(f"{source}: t-statistic={t_stat:.4f}, p-value={p_val:.4f}, " +
                  f"{'Significant' if p_val < 0.05 else 'Not significant'}")

# Create hourly averages for analyzing circadian patterns
def calculate_hourly_averages():
    hourly_avg_results = []
    
    for source_name in combined_df['source'].unique():
        source_data = combined_df[combined_df['source'] == source_name]
        
        # Group by hour
        for hour_val, hour_data in source_data.groupby('hour'):
            avg_row = {
                'source': source_name,
                'hour': hour_val
            }
            
            # Calculate average for each metric
            for metric in key_metrics:
                values = hour_data[metric].dropna()
                if len(values) > 0:
                    avg_row[f"{metric}_mean"] = values.mean()
            
            hourly_avg_results.append(avg_row)
    
    return pd.DataFrame(hourly_avg_results)

# Calculate hourly averages
hourly_averages = calculate_hourly_averages()

# Plot circadian patterns
plt.figure(figsize=(15, 10))

# Find heart rate column
hr_cols = [col for col in hourly_averages.columns if 'heart_rate' in col and col.endswith('_mean')]

if hr_cols:
    hr_col = hr_cols[0]
    for source in hourly_averages['source'].unique():
        source_data = hourly_averages[hourly_averages['source'] == source]
        source_data = source_data.sort_values('hour')  # Sort by hour for proper line plot
        plt.plot(source_data['hour'], source_data[hr_col], 'o-', label=f"{source}")

    plt.title('Heart Rate by Hour of Day (Circadian Pattern)')
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('Average Heart Rate (bpm)')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.legend()
    plt.show()

print("\nTime Pattern Analysis Summary Complete!") 