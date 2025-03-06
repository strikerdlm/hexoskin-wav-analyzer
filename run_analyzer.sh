#!/bin/bash

echo "Hexoskin WAV File Analyzer Setup"
echo "==============================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found! Please install Python 3.6 or higher."
    echo "Visit https://www.python.org/downloads/"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment."
        exit 1
    fi
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run the application
echo
echo "Starting Hexoskin WAV File Analyzer..."
echo
python hexoskin_wav_loader.py

# Deactivate virtual environment when done
deactivate 