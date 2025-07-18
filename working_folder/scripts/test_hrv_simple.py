#!/usr/bin/env python3
"""
Simple test to check data structure and generate HRV tables
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def check_database():
    """Check the database structure"""
    
    try:
        conn = sqlite3.connect('merged_data.db')
        cursor = conn.cursor()
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"Database tables: {tables}")
        
        if tables:
            # Get first table info
            table_name = tables[0][0]
            print(f"Using table: {table_name}")
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print(f"Columns: {[col[1] for col in columns]}")
            
            # Get sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            sample = cursor.fetchall()
            print(f"Sample rows: {len(sample)}")
            
            # Load data
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            conn.close()
            
            print(f"Data loaded: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            return df
            
    except Exception as e:
        print(f"Database error: {e}")
        
    conn.close()
    return None

def create_sample_hrv_table():
    """Create sample HRV table to demonstrate the structure"""
    
    # Sample data (this is what the output would look like)
    sample_data = {
        'Subject': ['T01_Mara', 'T01_Mara', 'T02_Laura', 'T02_Laura', 'T03_Nancy'],
        'Sol': ['Sol2', 'Sol3', 'Sol2', 'Sol3', 'Sol2'],
        'n_beats': [1200, 1150, 1300, 1250, 1100],
        'mean_hr_bpm': [68.5, 72.1, 65.2, 69.8, 71.3],
        'std_hr_bpm': [8.2, 9.1, 7.5, 8.8, 9.5],
        'mean_rr_ms': [876.8, 833.5, 920.2, 859.1, 842.1],
        'std_rr_ms': [105.2, 112.8, 98.7, 108.9, 115.4],
        'rmssd': [45.2, 52.1, 38.9, 47.8, 51.2],
        'pnn50': [12.8, 18.5, 8.9, 15.2, 19.1],
        'sdnn': [98.4, 105.7, 89.2, 102.3, 108.9],
        'lf_power': [1250, 1580, 980, 1420, 1650],
        'hf_power': [890, 1120, 720, 1050, 1180],
        'lf_hf_ratio': [1.4, 1.41, 1.36, 1.35, 1.4],
        'sd1': [32.1, 36.9, 27.5, 33.8, 36.2],
        'sd2': [128.7, 138.4, 116.8, 134.1, 142.5]
    }
    
    return pd.DataFrame(sample_data)

def main():
    """Main function"""
    
    print("="*60)
    print("HRV TABLE STRUCTURE TEST")
    print("="*60)
    
    # Check database
    df = check_database()
    
    if df is not None:
        print("\n" + "="*40)
        print("ACTUAL DATA STRUCTURE")
        print("="*40)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for heart rate column
        hr_columns = [col for col in df.columns if 'heart' in col.lower() or 'hr' in col.lower()]
        print(f"Heart rate columns: {hr_columns}")
        
        # Check for subject/Sol columns
        subject_cols = [col for col in df.columns if 'subject' in col.lower()]
        sol_cols = [col for col in df.columns if 'sol' in col.lower()]
        print(f"Subject columns: {subject_cols}")
        print(f"Sol columns: {sol_cols}")
        
        # Show sample data
        print("\nFirst 5 rows:")
        print(df.head())
        
    else:
        print("Could not load database data")
    
    # Create sample HRV table
    print("\n" + "="*40)
    print("SAMPLE HRV TABLE STRUCTURE")
    print("="*40)
    
    sample_hrv = create_sample_hrv_table()
    print("This is what your HRV metrics table would look like:")
    print(sample_hrv.to_string(index=False))
    
    # Save sample
    sample_hrv.to_csv('sample_hrv_table.csv', index=False)
    print(f"\nSample HRV table saved to: sample_hrv_table.csv")
    
    print("\n" + "="*40)
    print("HRV METRICS EXPLANATION")
    print("="*40)
    
    explanations = {
        'n_beats': 'Number of heartbeats in the segment',
        'mean_hr_bpm': 'Average heart rate (beats per minute)',
        'std_hr_bpm': 'Standard deviation of heart rate',
        'mean_rr_ms': 'Average RR interval (milliseconds)',
        'std_rr_ms': 'Standard deviation of RR intervals',
        'rmssd': 'Root mean square of successive differences',
        'pnn50': 'Percentage of successive RR intervals > 50ms',
        'sdnn': 'Standard deviation of all RR intervals',
        'lf_power': 'Low frequency power (0.04-0.15 Hz)',
        'hf_power': 'High frequency power (0.15-0.4 Hz)',
        'lf_hf_ratio': 'Ratio of LF to HF power',
        'sd1': 'Poincaré plot SD1 (short-term variability)',
        'sd2': 'Poincaré plot SD2 (long-term variability)'
    }
    
    for metric, explanation in explanations.items():
        print(f"{metric:12}: {explanation}")

if __name__ == "__main__":
    main() 