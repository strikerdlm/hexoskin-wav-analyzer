@echo off
REM Start Jupyter Notebook for Valquiria Analysis
REM This script starts Jupyter with the correct configuration

echo ========================================
echo Starting Jupyter for Valquiria Analysis
echo ========================================

REM Change to the joined_data directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if Jupyter is installed
jupyter --version >nul 2>&1
if errorlevel 1 (
    echo Installing Jupyter requirements...
    pip install -r requirements_jupyter.txt
    if errorlevel 1 (
        echo Error: Failed to install Jupyter requirements
        pause
        exit /b 1
    )
)

REM Start Jupyter Notebook
echo Starting Jupyter Notebook...
echo.
echo The notebook will open in your default browser
echo If it doesn't open automatically, go to: http://localhost:8888
echo.
echo Press Ctrl+C to stop the server
echo.

jupyter notebook --notebook-dir="%~dp0" --port=8888 --no-browser

pause 