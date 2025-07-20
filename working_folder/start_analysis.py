#!/usr/bin/env python3
"""
Jupyter Startup Script for Valquiria Analysis
"""
import subprocess
import sys
import os
from pathlib import Path

def start_jupyter():
    """Start Jupyter notebook with proper configuration"""
    # Apply compatibility patches
    exec(open('comprehensive_jupyter_fix.py').read())
    
    # Start Jupyter
    print("Starting Jupyter Notebook...")
    print("The notebook will open in your browser automatically.")
    print("Press Ctrl+C to stop the server.")
    
    cmd = ["jupyter", "notebook", "--no-browser", "--port=8888"]
    subprocess.run(cmd)

if __name__ == "__main__":
    start_jupyter()
