#!/usr/bin/env python3
"""
Hexoskin WAV Analyzer - Launcher Script
Created by Diego Malpica, MD
"""

import os
import sys
from hexoskin_wav_loader import main

if __name__ == "__main__":
    # Add the current directory to the path to ensure imports work correctly
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Launch the application
    main() 