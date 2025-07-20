#!/usr/bin/env python
"""
Example script to demonstrate how to use the CSV joiner
"""

import os
import sys
from csv_joiner import main

if __name__ == "__main__":
    print("CSV Joiner Example")
    print("=================")
    print("This example demonstrates how to use the CSV joiner.")
    print("The script will:")
    print("1. Find all Sol folders")
    print("2. List all subject folders")
    print("3. Join CSV files for each subject")
    print("4. Save the joined data in the 'joined_data' folder")
    print("")
    print("Press Enter to continue...")
    input()
    
    # Run the CSV joiner
    main()
    
    print("\nExample completed.")
    print("Check the 'joined_data' folder for the joined CSV files.")
    print("Check the 'logs' folder for the detailed log file.") 