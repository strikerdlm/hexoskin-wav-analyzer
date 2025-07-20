"""
Data Loading Utility for Valquiria Analysis
Provides functions to load and prepare data for analysis.
"""

import pandas as pd
import sqlite3
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_csv_data():
    """Load all CSV files from the joined_data directory."""
    data_dir = Path(__file__).parent.parent
    csv_files = list(data_dir.glob("*.csv"))
    
    dataframes = {}
    for file in csv_files:
        try:
            df = pd.read_csv(file, low_memory=False)
            dataframes[file.stem] = df
            print(f"✓ Loaded {file.name}: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"✗ Error loading {file.name}: {e}")
    
    return dataframes

def load_database_data(db_path="merged_data.db"):
    """Load data from SQLite database."""
    data_dir = Path(__file__).parent.parent
    db_file = data_dir / db_path
    
    if not db_file.exists():
        print(f"✗ Database file {db_file} not found")
        return None
    
    try:
        conn = sqlite3.connect(db_file)
        
        # Check available tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("✗ No tables found in database")
            conn.close()
            return None
        
        # Load the first table (assuming it's the main data table)
        table_name = tables[0][0]
        print(f"✓ Loading table: {table_name}")
        
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        
        print(f"✓ Loaded database: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        print(f"✗ Error loading database: {e}")
        return None

def get_data_summary(df):
    """Get a summary of the dataset."""
    if df is None:
        return "No data available"
    
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "missing_values": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum(),
        "column_names": list(df.columns),
        "data_types": df.dtypes.value_counts().to_dict()
    }
    
    return summary

def print_data_summary(df, title="Dataset Summary"):
    """Print a formatted summary of the dataset."""
    if df is None:
        print(f"{title}: No data available")
        return
    
    summary = get_data_summary(df)
    
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Rows: {summary['rows']:,}")
    print(f"Columns: {summary['columns']}")
    print(f"Memory Usage: {summary['memory_usage_mb']} MB")
    print(f"Missing Values: {summary['missing_values']:,}")
    print(f"Duplicate Rows: {summary['duplicates']:,}")
    
    print(f"\nData Types:")
    for dtype, count in summary['data_types'].items():
        print(f"  {dtype}: {count} columns")
    
    print(f"\nColumns:")
    for i, col in enumerate(summary['column_names'], 1):
        print(f"  {i:2d}. {col}")
    
    print(f"{'='*50}")

def load_sample_data(sample_size=10000, random_state=42):
    """Load a sample of data for quick testing."""
    df = load_database_data()
    if df is not None:
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=random_state)
        print(f"✓ Loaded sample: {len(sample_df)} rows")
        return sample_df
    return None

def check_data_quality(df):
    """Perform basic data quality checks."""
    if df is None:
        print("No data to check")
        return
    
    print("\n" + "="*50)
    print("DATA QUALITY CHECK")
    print("="*50)
    
    # Check for missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    print("\nMissing Values:")
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]:,} ({missing_pct[col]:.1f}%)")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates:,} ({duplicates/len(df)*100:.1f}%)")
    
    # Check data types
    print(f"\nData Types:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"  {dtype}: {count} columns")
    
    # Check for potential issues
    print(f"\nPotential Issues:")
    
    # Check for infinite values
    inf_count = df.isin([float('inf'), float('-inf')]).sum().sum()
    if inf_count > 0:
        print(f"  Infinite values: {inf_count}")
    
    # Check for very large numbers (potential errors)
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].max() > 1e6:
            print(f"  Large values in {col}: max = {df[col].max():.2e}")
    
    print("="*50)

def get_subject_data(df, subject_col='subject', subject_id=None):
    """Get data for a specific subject."""
    if df is None:
        return None
    
    if subject_id is None:
        # Return list of available subjects
        subjects = df[subject_col].unique()
        print(f"Available subjects: {list(subjects)}")
        return subjects
    
    subject_data = df[df[subject_col] == subject_id]
    if len(subject_data) == 0:
        print(f"Subject {subject_id} not found")
        return None
    
    print(f"✓ Loaded data for {subject_id}: {len(subject_data)} rows")
    return subject_data

if __name__ == "__main__":
    # Test data loading
    print("Testing data loading utilities...")
    
    # Test CSV loading
    print("\n1. Testing CSV loading:")
    csv_data = load_csv_data()
    print(f"Loaded {len(csv_data)} CSV files")
    
    # Test database loading
    print("\n2. Testing database loading:")
    db_data = load_database_data()
    
    if db_data is not None:
        print_data_summary(db_data, "Database Data Summary")
        check_data_quality(db_data)
        
        # Test subject data
        print("\n3. Testing subject data extraction:")
        subjects = get_subject_data(db_data)
        if subjects is not None and len(subjects) > 0:
            sample_subject = subjects[0]
            subject_data = get_subject_data(db_data, subject_id=sample_subject)
            if subject_data is not None:
                print_data_summary(subject_data, f"Subject {sample_subject} Data")
    
    print("\n✓ Data loading utilities test completed") 