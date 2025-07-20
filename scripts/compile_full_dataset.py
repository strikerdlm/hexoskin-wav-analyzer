import pandas as pd
import numpy as np
import os
from datetime import datetime
import re

# Function to clean column names
def clean_column_name(col_name):
    if '(' in col_name and ')' in col_name:
        # Extract the part before the first opening parenthesis that contains a URL
        clean_name = col_name.split('(/api')[0].strip()
        # Further clean any remaining special characters
        clean_name = clean_name.replace('[', '').replace(']', '').replace(' ', '_').replace('/', '_')
        return clean_name
    return col_name.replace('[', '').replace(']', '').replace(' ', '_').replace('/', '_')

# Function to process a single CSV file
def process_csv_file(file_path, source_label=None):
    print(f"Processing {os.path.basename(file_path)}...")
    
    try:
        # Extract Sol number from the path if no source label provided
        if source_label is None:
            sol_match = re.search(r'Sol\s+(\d+)', file_path)
            if sol_match:
                source_label = f"Sol {sol_match.group(1)}"
            else:
                source_label = os.path.basename(file_path)
        
        # Load the data
        df = pd.read_csv(file_path)
        
        # Add source identifier
        df['data_source'] = source_label
        df['original_file'] = os.path.basename(file_path)
        
        # Clean column names
        orig_columns = list(df.columns)
        clean_columns = [clean_column_name(col) for col in orig_columns]
        df.columns = clean_columns
        
        # Convert timestamp to datetime
        # Determine the timestamp column name
        time_cols = [col for col in df.columns if 'time' in col.lower() and '1000' in col]
        if time_cols:
            time_col = time_cols[0]
            df['datetime'] = pd.to_datetime(df[time_col], unit='ms')
            
            # Add additional time components for easier analysis
            df['date'] = df['datetime'].dt.date
            df['hour'] = df['datetime'].dt.hour
            df['minute'] = df['datetime'].dt.minute
            df['is_night'] = (df['hour'] >= 19) | (df['hour'] < 7)
            
            # Add day_of_recording column (integer starting from 1 for the first recording)
            df['recording_day'] = 1  # Will be updated later based on all data
        
        # Verify data was loaded correctly
        print(f"  - Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"  - Time range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return df
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# List of CSV files to process
file_paths = [
    r"C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\Sol 2 (completo)\T01_Mara\record_4494.csv",
    r"C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\Sol 3 (completo)\T01_Mara\record-4502.csv",
    r"C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\Sol 4 (completo)\T01_Mara\record_4510.csv",
    r"C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\Sol 6 (completo)\T01_Mara\record_4512.csv",
    r"C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\Sol 7 (completo)\T01_Mara\record_4582.csv",
    r"C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\Sol 10 (completo)\T01_Mara\record_4592.csv",
    r"C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\Sol 12 (completo)\T01_Mara\record_4590.csv",
    r"C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\Sol 15 (completo)\T01_Mara\record_4609.csv",
    r"C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\Sol 16 (completo)\T01_Mara\record_4610.csv"
]

# Create labels for each file (Sol 2, Sol 3, etc.)
file_labels = [f"Sol {i}" for i in [2, 3, 4, 6, 7, 10, 12, 15, 16]]

# Process each file and store in a list
print("Starting to process all CSV files...")
dataframes = []
for i, (file_path, label) in enumerate(zip(file_paths, file_labels)):
    df = process_csv_file(file_path, label)
    if df is not None:
        dataframes.append(df)

# Combine all dataframes
if dataframes:
    print("\nCombining all dataframes...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Sort by datetime
    combined_df = combined_df.sort_values('datetime')
    
    # Calculate and assign recording days (sequential number for each date)
    dates = sorted(combined_df['date'].unique())
    date_to_day = {date: i+1 for i, date in enumerate(dates)}
    combined_df['recording_day'] = combined_df['date'].map(date_to_day)
    
    # Print information about the combined dataframe
    print(f"Combined dataset: {len(combined_df)} rows, {len(combined_df.columns)} columns")
    print(f"Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")
    print(f"Total recording days: {len(dates)}")
    
    # Summary of data by source
    print("\nSummary by data source:")
    source_summary = combined_df.groupby('data_source').agg({
        'datetime': ['min', 'max', 'count']
    })
    
    # Flatten the multi-index columns
    source_summary.columns = ['_'.join(col).strip() for col in source_summary.columns.values]
    print(source_summary)
    
    # Find common metrics across all sources - focus on health metrics, not metadata
    all_columns = combined_df.columns
    health_metrics = [col for col in all_columns if col not in [
        'time_s_1000', 'datetime', 'data_source', 'original_file', 
        'date', 'hour', 'minute', 'is_night', 'recording_day'
    ]]
    
    print("\nHealth metrics in the dataset:")
    for metric in health_metrics:
        non_null_count = combined_df[metric].count()
        percent_complete = (non_null_count / len(combined_df)) * 100
        print(f"- {metric}: {non_null_count} non-null values ({percent_complete:.1f}% complete)")
    
    # Export to CSV
    output_path = "full_physiological_dataset.csv"
    print(f"\nExporting combined dataset to {output_path}...")
    combined_df.to_csv(output_path, index=False)
    print(f"Export complete! File saved to {output_path}")
    
    # Also provide a "clean" version with just the most important columns
    # Focus on the key health metrics and time information
    important_metrics = ['heart_rate', 'SPO2', 'breathing_rate', 'systolic_pressure', 'temperature']
    selected_columns = ['datetime', 'date', 'hour', 'recording_day', 'data_source']
    
    # Find columns containing the important metrics
    for metric in important_metrics:
        matching_cols = [col for col in health_metrics if metric.lower() in col.lower()]
        selected_columns.extend(matching_cols)
    
    # Create clean dataset
    clean_df = combined_df[selected_columns].copy()
    clean_output_path = "clean_physiological_dataset.csv"
    print(f"Exporting clean dataset with selected metrics to {clean_output_path}...")
    clean_df.to_csv(clean_output_path, index=False)
    print(f"Clean export complete! File saved to {clean_output_path}")
else:
    print("No data was successfully processed. Please check the file paths and formats.")

print("\nProcessing complete!") 