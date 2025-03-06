@echo off
echo Hexoskin WAV File Analyzer Setup
echo ===============================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found! Please install Python 3.6 or higher.
    echo Visit https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM Activate virtual environment and install dependencies
echo Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Run the application
echo.
echo Starting Hexoskin WAV File Analyzer...
echo.
python hexoskin_wav_loader.py

REM Deactivate virtual environment when done
call venv\Scripts\deactivate.bat
pause 