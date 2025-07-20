import os
import glob
import pandas as pd
import sqlite3

# Define the folder containing the CSV files
csv_folder = os.path.join(os.path.dirname(__file__), 'joined_data')

# Get list of all CSV files in the folder
csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))

if not csv_files:
    print('No CSV files found in the joined_data folder.')
    exit(1)

# List to hold dataframes
df_list = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        print(f'Error reading {file}: {e}')
        continue
    
    # Determine the time column and conversion factor
    if 'time [s/1000]' in df.columns:
        time_col = 'time [s/1000]'
        factor = 1000.0
        # Drop extra column 'user' if exists
        if 'user' in df.columns:
            df = df.drop(columns=['user'])
    elif 'time [s/256]' in df.columns:
        time_col = 'time [s/256]'
        factor = 256.0
    else:
        print(f'File {file} does not have a recognized time column. Skipping.')
        continue
    
    # Define the common columns to keep
    common_columns = ['Sol', 'source_file', time_col, 'breathing_rate [rpm]', 'minute_ventilation [mL/min]', 'sleep_position [NA]', 'activity [g]', 'heart_rate [bpm]', 'cadence [spm]']
    
    # Check if all common columns are present (if not, take intersection)
    available_columns = [col for col in common_columns if col in df.columns]
    if time_col not in available_columns:
        available_columns.append(time_col)
    
    df = df[available_columns].copy()
    
    # Convert time column to seconds
    df['time_seconds'] = df[time_col] / factor
    
    # Optional: rename the original time column to 'time_raw' for reference
    df.rename(columns={time_col: 'time_raw'}, inplace=True)
    
    # Add a subject identifier based on the file name
    subject = os.path.splitext(os.path.basename(file))[0]
    df['subject'] = subject
    
    df_list.append(df)
    print(f'Processed {file} with conversion factor {factor}')

# Concatenate all dataframes
if df_list:
    combined_df = pd.concat(df_list, ignore_index=True)
else:
    print('No dataframes to combine. Exiting.')
    exit(1)

# Save the combined dataframe to a SQLite database
db_path = os.path.join(os.path.dirname(__file__), 'merged_data.db')
conn = sqlite3.connect(db_path)
try:
    combined_df.to_sql('merged_data', conn, if_exists='replace', index=False)
    print(f'Combined data saved to {db_path} in table "merged_data".')
except Exception as e:
    print(f'Error saving to database: {e}')
finally:
    conn.close()

# Connect to the SQLite database
conn = sqlite3.connect(db_path)

# Load the table into a DataFrame
df = pd.read_sql_query("SELECT * FROM merged_data", conn)

# Close the database connection
conn.close()

# Display the first few rows
df.head()

# Export the DataFrame to CSV
df.to_csv("merged_data.csv", index=False) 