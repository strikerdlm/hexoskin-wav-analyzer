# CSV Joiner for Valquiria Research Data

This application joins CSV files for the same subjects across different "Sol" folders in the Valquiria research data. It is designed to handle the specific structure of the Valquiria project data.

## Features

- **Complete Subject Listing**: Lists all subject folders across all Sol directories (with recursive subfolder search)
- **Flexible File Matching**: Finds all CSV files in subject folders regardless of filename
- **Header Standardization**: Automatically cleans up header names by removing API paths in parentheses
- **Detailed Logging**: Creates comprehensive log files with details about the data structure and processing
- **Advanced Header Analysis**: Analyzes and reports differences in CSV headers across Sol folders
- **Consistent Data Merging**: Ensures headers match when joining different CSV files, even when the original files have different structures
- **Sol Identification**: Adds a "Sol" column to identify the source of each row
- **Error Handling**: Robust error handling with detailed log messages

## Requirements

- Python 3.6 or later
- pandas library (specified in requirements.txt)

## Installation

1. Ensure Python is installed on your system
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the script from the parent directory (where the Sol folders are located):
   ```bash
   python run_csv_joiner.py
   ```

2. The script will:
   - Recursively scan all "Sol X (completo)" folders and subfolders
   - Find subject folders matching pattern T##_Name anywhere in the directory structure
   - List all subject folders and create a detailed inventory in the log file
   - Find all CSV files in each subject folder (regardless of filename)
   - Standardize header names by removing API paths and unnecessary information
   - Analyze header differences between CSV files from different Sol folders
   - Join the CSV data across all Sol folders
   - Add a "Sol" column to identify the source of each row
   - Save the joined data in the "joined_data" folder

3. Check the "joined_data" directory for the combined CSV files
4. Review the detailed log files in the logs directory at:
   ```
   C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\csv_joiner\logs
   ```

## CSV File Handling

The application is designed to be flexible in how it handles CSV files:

- **Any CSV File**: Finds all files with `.csv` extension in each subject folder
- **Multiple File Handling**: If a subject folder contains multiple CSV files, the first one is used (with a log entry noting this)
- **Recursive Search**: Finds subject folders in any subfolder, not just at the top level
- **Complete Coverage**: Ensures all subject folders are checked for CSV files, regardless of filename convention

## Header Standardization

The app standardizes column headers across different CSV files:

- **Removing API Paths**: Automatically removes parts like `(/api/datatype/19/)` from headers
- **Example**: `heart_rate [bpm](/api/datatype/19/)` becomes `heart_rate [bpm]`
- **Consistent Mapping**: Ensures data from different files is mapped to the same standardized headers
- **All Headers Preserved**: Even after standardization, all unique headers are preserved in the final CSV

## Handling Different CSV Structures

A key feature of this application is its ability to handle CSV files with different structures (headers) across Sol folders:

1. **Header Analysis**: The app analyzes all headers across all Sol folders for each subject
2. **Standardization**: It standardizes headers by removing unnecessary parts (like API paths)
3. **Difference Detection**: It identifies which headers exist in which Sol folders
4. **Complete Merging**: All columns from all CSVs are included in the final joined file
5. **Null Filling**: When a column doesn't exist in a particular Sol's CSV, it's filled with NULL values
6. **Detailed Logging**: The log file includes detailed information about header differences

This ensures that all data is preserved when joining files, even when the CSV structure varies between Sol folders.

## Output Files

- **Joined CSV Files**: Located in the "joined_data" folder in the data directory
- **Log Files**: Located at `C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\csv_joiner\logs` with timestamps (e.g., "csv_joiner_20230615_120145.log")

## Log Details

The log file includes:
- Complete list of all Sol directories found
- Inventory of all subject folders in each Sol directory (including in subfolders)
- Summary of which subjects exist in which Sol folders
- Details of all CSV files found, including cases where multiple CSVs exist in one folder
- Header standardization information (before and after standardization)
- Detailed analysis of header differences between CSV files
- Warning messages for subjects that exist but have no CSV files
- Details about the CSV joining process
- Statistics about the joined data

## Example

For a subject like "T02_Laura" that has data in Sol 2, 3, 4, etc., the script will:
1. Find all folders named "T02_Laura" recursively in all Sol directories
2. Find all CSV files in these folders
3. Standardize header names by removing API paths and unnecessary information
4. Analyze header differences between the CSV files
5. Join them into a single CSV file called "T02_Laura.csv", preserving all columns
6. Add a "Sol" column with values like "2", "3", "4", etc.
7. Log detailed information about the process 