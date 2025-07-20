import os
import pandas as pd
import re
import glob
from pathlib import Path
import logging
import datetime
import sys
from collections import defaultdict
import chardet
import numpy as np

def setup_logging():
    """Set up logging configuration"""
    # Set up logging with specific absolute path for logs
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"csv_joiner_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Log file created at: {log_path}")
    return log_path

def detect_encoding(file_path):
    """
    Detect the encoding of a file using chardet
    """
    # Read the first 100KB of the file to determine encoding
    with open(file_path, 'rb') as f:
        raw_data = f.read(100000)
    
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    confidence = result['confidence']
    
    # Default to utf-8 if detection fails or has low confidence
    if not encoding or confidence < 0.7:
        encoding = 'utf-8'
    
    logging.debug(f"Detected encoding for {file_path}: {encoding} (confidence: {confidence:.2f})")
    return encoding

def standardize_header(header):
    """
    Standardize header name by removing API paths or anything in parentheses
    Example: 'heart_rate [bpm](/api/datatype/19/)' -> 'heart_rate [bpm]'
    """
    # Match the portion before any parentheses (keeping square brackets if present)
    match = re.match(r'^(.*?)(?:\(|$)', header)
    if match:
        # Remove any trailing whitespace
        return match.group(1).rstrip()
    return header

def find_subject_folders(root_dir):
    """
    Find all subject folders recursively in a given directory
    Subject folders typically match pattern T##_Name
    """
    subject_folders = []
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(root_dir):
        # Check if this directory name matches the subject pattern
        current_dir = os.path.basename(root)
        # More flexible pattern to match T##_Name or similar variations
        if re.match(r'^T\d+_\w+', current_dir) or re.match(r'^T\d+_[\w\s]+(?:\(.*\))?$', current_dir):
            subject_folders.append((root, current_dir))
    
    return subject_folders

def list_all_subject_folders(base_dir):
    """Find and list all subject folders across all Sol directories"""
    sol_dirs = []
    all_subjects = set()
    subjects_by_sol = {}
    subject_folders_map = {}  # Maps subject name to all its folder paths
    
    # Find all Sol directories
    for folder in os.listdir(base_dir):
        if re.match(r'Sol \d+ \(completo\)', folder):
            sol_dir = os.path.join(base_dir, folder)
            if os.path.isdir(sol_dir):
                sol_num = re.search(r'Sol (\d+)', folder).group(1)
                sol_dirs.append((sol_dir, sol_num))
                subjects_by_sol[sol_num] = []
    
    # Sort by Sol number
    sol_dirs.sort(key=lambda x: int(x[1]))
    logging.info(f"Found {len(sol_dirs)} Sol directories: {', '.join([f'Sol {num}' for _, num in sol_dirs])}")
    
    # Find all subject folders in each Sol directory (recursively)
    for sol_dir, sol_number in sol_dirs:
        logging.info(f"Scanning Sol {sol_number} for subject folders (including subfolders)...")
        try:
            # Find all subject folders in this Sol directory
            subject_folders = find_subject_folders(sol_dir)
            
            for folder_path, subject_folder in subject_folders:
                # Add to subjects list for this Sol
                subjects_by_sol[sol_number].append(subject_folder)
                all_subjects.add(subject_folder)
                
                # Add to the subject folders map
                if subject_folder not in subject_folders_map:
                    subject_folders_map[subject_folder] = []
                subject_folders_map[subject_folder].append((sol_number, folder_path))
                
                logging.info(f"  Found subject folder: {subject_folder} in Sol {sol_number} at {folder_path}")
                
        except Exception as e:
            logging.error(f"Error scanning Sol {sol_number} directory: {str(e)}")
    
    # Log a summary of each Sol and its subjects
    logging.info("Summary of subjects by Sol folder:")
    for sol_num, subjects in sorted(subjects_by_sol.items(), key=lambda x: int(x[0])):
        subjects.sort()  # Sort subjects alphabetically
        logging.info(f"  Sol {sol_num}: {len(subjects)} subjects - {', '.join(subjects)}")
    
    # Log overall subject summary
    all_subjects_list = sorted(list(all_subjects))
    logging.info(f"Found {len(all_subjects_list)} unique subjects across all Sol folders:")
    for subject in all_subjects_list:
        # List which Sol folders contain this subject
        sols_with_subject = [sol_num for sol_num, subjects in subjects_by_sol.items() if subject in subjects]
        logging.info(f"  {subject} found in {len(sols_with_subject)} Sol folders: {', '.join(sols_with_subject)}")
        
        # Log all folder paths for this subject
        for sol_num, folder_path in subject_folders_map[subject]:
            logging.info(f"    Sol {sol_num}: {folder_path}")
    
    return sol_dirs, all_subjects_list, subjects_by_sol, subject_folders_map

def find_csv_files(directory):
    """Find all CSV files in a directory"""
    csv_files = []
    
    # Check if directory exists
    if not os.path.exists(directory):
        return csv_files
    
    # List all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        # Check if it's a file and has .csv extension
        if os.path.isfile(file_path) and file_path.lower().endswith('.csv'):
            csv_files.append(file_path)
    
    return csv_files

def read_csv_with_fallback(file_path):
    """
    Attempt to read a CSV file with multiple approaches to handle different formats
    """
    try:
        # First detect encoding
        encoding = detect_encoding(file_path)
        
        # Try to read with detected encoding
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            if len(df) > 0:
                return df
        except Exception as e:
            logging.warning(f"Could not read {file_path} with detected encoding {encoding}: {str(e)}")

        # Try different separators
        separators = [',', ';', '\t', '|']
        for sep in separators:
            try:
                df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                if len(df) > 0:
                    logging.info(f"Successfully read {file_path} with separator '{sep}'")
                    return df
            except Exception as e:
                continue
        
        # If still not successful, try with different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        for enc in encodings:
            if enc != encoding:  # Skip the already tried encoding
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    if len(df) > 0:
                        logging.info(f"Successfully read {file_path} with encoding '{enc}'")
                        return df
                except Exception:
                    continue
        
        # Last resort: try reading with Python's built-in CSV module and convert to DataFrame
        import csv
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                headers = next(reader)
                data = list(reader)
                if headers and data:
                    df = pd.DataFrame(data, columns=headers)
                    logging.info(f"Successfully read {file_path} with CSV module")
                    return df
        except Exception as e:
            logging.error(f"All attempts to read {file_path} failed: {str(e)}")
            
        # If all above attempts fail, return an empty DataFrame with basic columns
        return pd.DataFrame(columns=['time [s/1000]', 'user'])
        
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {str(e)}")
        return pd.DataFrame(columns=['time [s/1000]', 'user'])

def analyze_headers(subject, sol_files):
    """Analyze headers across all CSV files for a subject and identify differences"""
    header_analysis = {
        'all_headers': set(),
        'all_standardized_headers': set(),
        'headers_by_sol': {},
        'standardized_headers_by_sol': {},
        'header_mapping': {},  # Maps original headers to standardized headers
        'common_headers': None,
        'header_differences': defaultdict(list)
    }
    
    # First pass to collect all headers and standardize them
    for sol_number, csv_path in sorted(sol_files.items(), key=lambda x: int(x[0])):
        try:
            # Try to read the CSV with our enhanced reader
            df = read_csv_with_fallback(csv_path)
            
            if df.empty:
                logging.warning(f"  No data found in CSV file for {subject} in Sol {sol_number}")
                header_analysis['headers_by_sol'][sol_number] = []
                header_analysis['standardized_headers_by_sol'][sol_number] = []
                continue
                
            # Get headers
            original_headers = df.columns.tolist()
            
            # Standardize headers
            standardized_headers = [standardize_header(h) for h in original_headers]
            
            # Store both original and standardized headers
            header_analysis['headers_by_sol'][sol_number] = original_headers
            header_analysis['standardized_headers_by_sol'][sol_number] = standardized_headers
            
            # Create mapping from original to standardized headers
            header_mapping = dict(zip(original_headers, standardized_headers))
            for orig, std in header_mapping.items():
                header_analysis['header_mapping'][orig] = std
            
            # Update all unique headers sets
            header_analysis['all_headers'].update(original_headers)
            header_analysis['all_standardized_headers'].update(standardized_headers)
            
            # Initialize common headers if this is the first file
            if header_analysis['common_headers'] is None:
                header_analysis['common_headers'] = set(standardized_headers)
            else:
                # Update common headers to only include headers found in all files
                header_analysis['common_headers'] &= set(standardized_headers)
            
            # Log headers and their standardized versions
            standardized_count = len(set(standardized_headers))
            original_count = len(original_headers)
            if standardized_count != original_count:
                logging.info(f"  Sol {sol_number} CSV has {original_count} original columns, standardized to {standardized_count} unique columns")
            else:
                logging.info(f"  Sol {sol_number} CSV has {original_count} columns")
                
            # Log any headers that were standardized (changed)
            for orig, std in header_mapping.items():
                if orig != std:
                    logging.info(f"    Standardized header: '{orig}' -> '{std}'")
                
        except Exception as e:
            logging.error(f"  Error reading headers from {csv_path}: {str(e)}")
            header_analysis['headers_by_sol'][sol_number] = []
            header_analysis['standardized_headers_by_sol'][sol_number] = []
    
    # Identify unique standardized headers for each Sol
    for sol_number, std_headers in header_analysis['standardized_headers_by_sol'].items():
        std_headers_set = set(std_headers)
        # Skip empty headers
        if not std_headers_set:
            continue
            
        unique_headers = std_headers_set - header_analysis['common_headers']
        
        # Record which Sols have which unique headers
        for header in unique_headers:
            header_analysis['header_differences'][header].append(sol_number)
    
    # Log header analysis
    logging.info(f"Header analysis for {subject}:")
    logging.info(f"  Total unique standardized headers across all Sol files: {len(header_analysis['all_standardized_headers'])}")
    logging.info(f"  Common standardized headers found in all Sol files: {len(header_analysis['common_headers'])}")
    
    # Log header differences if any
    if header_analysis['header_differences']:
        logging.info(f"  Header differences found across Sol files (after standardization):")
        for header, sols in header_analysis['header_differences'].items():
            sols_str = ', '.join(sorted(sols, key=int))
            logging.info(f"    Header '{header}' only found in Sol: {sols_str}")
    else:
        logging.info(f"  All Sol files have identical standardized headers")
    
    return header_analysis

def join_csv_files():
    """Main function to join CSV files for subjects across Sol folders"""
    # Base directory where Sol folders are located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Move up one directory since we're now in the csv_joiner folder
    base_dir = os.path.dirname(base_dir)
    
    # Create csv_joiner directory if it doesn't exist
    csv_joiner_dir = os.path.join(base_dir, "csv_joiner")
    os.makedirs(csv_joiner_dir, exist_ok=True)
    
    # Output directory for joined files
    output_dir = os.path.join(base_dir, "joined_data")
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Base directory: {base_dir}")
    logging.info(f"CSV joiner directory: {csv_joiner_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    # List all subject folders
    sol_dirs, all_subjects, subjects_by_sol, subject_folders_map = list_all_subject_folders(base_dir)
    
    # Dictionary to store subject data: {subject_name: {sol_number: csv_file_path}}
    subjects_data = {}
    
    # Find all CSV files for each subject
    for subject, folder_paths in subject_folders_map.items():
        if subject not in subjects_data:
            subjects_data[subject] = {}
            
        for sol_number, folder_path in folder_paths:
            logging.info(f"Scanning for CSV files for {subject} in Sol {sol_number}: {folder_path}")
            
            # Find all CSV files in the subject folder
            csv_files = find_csv_files(folder_path)
            
            if csv_files:
                # Use all CSV files instead of just the first one
                # If multiple CSV files are found, we'll combine them
                if len(csv_files) > 1:
                    filenames = [os.path.basename(f) for f in csv_files]
                    logging.info(f"  Found multiple CSV files for {subject} in Sol {sol_number}: {', '.join(filenames)}")
                    
                    # Store all CSV files for this subject in this Sol
                    subjects_data[subject][sol_number] = csv_files
                    logging.info(f"  Will process all {len(csv_files)} CSV files for {subject} in Sol {sol_number}")
                else:
                    # Store the single CSV file found
                    subjects_data[subject][sol_number] = [csv_files[0]]
                    logging.info(f"  Found CSV for {subject} in Sol {sol_number}: {os.path.basename(csv_files[0])}")
            else:
                logging.warning(f"  No CSV files found for {subject} in Sol {sol_number}")
    
    # Check for subjects that were found in folders but have no CSV files
    missing_csv_subjects = set(all_subjects) - set(s for s, files in subjects_data.items() if files)
    if missing_csv_subjects:
        logging.warning(f"Found {len(missing_csv_subjects)} subjects with no CSV files: {', '.join(sorted(list(missing_csv_subjects)))}")
    
    # Get total count of subjects and their data points
    subject_count = len([s for s, files in subjects_data.items() if files])
    total_files = sum(sum(len(files) for files in sol_dict.values()) for sol_dict in subjects_data.values())
    logging.info(f"Found {subject_count} unique subjects with CSV files, {total_files} CSV files total")
    
    # Log a summary of subjects and their Sol data
    for subject, sol_files in subjects_data.items():
        if not sol_files:
            continue
            
        sol_nums = sorted(sol_files.keys(), key=int)
        logging.info(f"Subject {subject} has CSV data in Sol: {', '.join(sol_nums)}")
        
        # Check if any Sol folders for this subject are missing CSV files
        sols_with_subject = [sol_num for sol_num, subjects in subjects_by_sol.items() if subject in subjects]
        sols_missing_csv = set(sols_with_subject) - set(sol_nums)
        if sols_missing_csv:
            logging.warning(f"  Subject {subject} exists in Sol {', '.join(sorted(sols_missing_csv, key=int))} but has no CSV files there")
    
    # Join CSV files for each subject
    joined_count = 0
    for subject, sol_files_dict in subjects_data.items():
        if not sol_files_dict:
            continue
        
        # Flatten sol_files_dict to get a list of all CSV files
        flattened_sol_files = {}
        for sol_number, file_list in sol_files_dict.items():
            # For each file, add it with a unique key (sol_number, index)
            for i, file_path in enumerate(file_list):
                flattened_sol_files[(sol_number, i)] = file_path
                
        logging.info(f"Processing {subject} with data from {len(sol_files_dict)} Sol folders")
        
        # Create a dictionary with just one file per sol for header analysis
        sol_files_for_analysis = {sol: files[0] for sol, files in sol_files_dict.items()}
        
        # Analyze headers across all SOL files for this subject
        header_analysis = analyze_headers(subject, sol_files_for_analysis)
        all_standardized_headers = header_analysis['all_standardized_headers']
        
        all_dfs = []
        
        # Process each CSV file
        for (sol_number, file_idx), csv_path in sorted(flattened_sol_files.items(), key=lambda x: (int(x[0][0]), x[0][1])):
            try:
                # Read CSV file using our enhanced reader
                df = read_csv_with_fallback(csv_path)
                
                if df.empty:
                    logging.warning(f"  Skipping empty file: {csv_path}")
                    continue
                
                # Standardize the DataFrame column names
                column_mapping = {col: standardize_header(col) for col in df.columns}
                df = df.rename(columns=column_mapping)
                
                # Add Sol number and source file as columns
                df['Sol'] = sol_number
                df['source_file'] = os.path.basename(csv_path)
                
                # Check for missing columns and add them if necessary
                missing_columns = []
                for col in all_standardized_headers:
                    if col not in df.columns and col not in ['Sol', 'source_file']:
                        df[col] = np.nan  # Use NaN instead of None
                        missing_columns.append(col)
                
                if missing_columns:
                    logging.warning(f"  Added {len(missing_columns)} missing columns to Sol {sol_number} data: {', '.join(missing_columns)}")
                
                # If 'user' column doesn't exist, add it with the subject name
                if 'user' not in df.columns:
                    df['user'] = subject
                
                # Skip if there's no data after all processing
                if df.empty:
                    logging.warning(f"  No data in {csv_path} after processing")
                    continue
                
                all_dfs.append(df)
                logging.info(f"  Added Sol {sol_number} data with {len(df)} rows and {len(df.columns)} columns")
            except Exception as e:
                logging.error(f"  Error processing {csv_path}: {str(e)}")
        
        if not all_dfs:
            logging.warning(f"No valid data found for {subject}")
            continue
        
        # Combine all DataFrames
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Reorder columns to ensure consistent layout, with 'Sol' and 'user' as the first columns
        columns = ['Sol', 'user', 'source_file'] + [col for col in combined_df.columns if col not in ['Sol', 'user', 'source_file']]
        combined_df = combined_df[columns]
        
        # Get some stats for logging
        total_rows = len(combined_df)
        logging.info(f"  Combined data for {subject} has {total_rows} rows and {len(combined_df.columns)} columns")
        
        # Save to output file
        output_file = os.path.join(output_dir, f"{subject}.csv")
        combined_df.to_csv(output_file, index=False)
        logging.info(f"  Saved joined data for {subject} to {output_file}")
        joined_count += 1
    
    logging.info(f"CSV joining process completed. Joined data for {joined_count} subjects out of {len(all_subjects)} total subjects.")
    return output_dir

def main():
    """Main entry point for the application"""
    print("Starting CSV Joiner for Valquiria Research Data")
    print("=" * 50)
    
    log_path = setup_logging()
    
    try:
        output_dir = join_csv_files()
        print("\nSuccessfully joined CSV files!")
        print(f"Results saved to: {output_dir}")
        print(f"Log file saved to: {log_path}")
        print("\nIMPORTANT: Detailed logs are available in the logs folder at:")
        print(f"  {os.path.dirname(log_path)}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
        print(f"\nAn error occurred during processing.")
        print(f"Log file with error details: {log_path}")
        sys.exit(1)

if __name__ == "__main__":
    main() 