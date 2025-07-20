#!/usr/bin/env python
"""
Run CSV Joiner

Main script to run the CSV joiner application.
This script joins CSV files for the same subjects across different "Sol" folders.
"""

import os
import sys
from csv_joiner.csv_joiner import main

if __name__ == "__main__":
    print("CSV Joiner for Valquiria Research Data")
    print("======================================")
    print("This application will join CSV files for the same subjects across different Sol folders.")
    print("It will create a 'joined_data' folder with the joined CSV files.")
    print("Detailed logs will be saved in the 'logs' folder.")
    print("")
    
    try:
        main()
        print("\nCSV joining completed successfully!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1) 