# Start Jupyter Notebook for Valquiria Analysis
# This script starts Jupyter with the correct configuration

Write-Host "========================================" -ForegroundColor Green
Write-Host "Starting Jupyter for Valquiria Analysis" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Change to the script directory
Set-Location $PSScriptRoot

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Error: Python is not installed or not in PATH" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if Jupyter is installed
try {
    $jupyterVersion = jupyter --version 2>&1
    Write-Host "✓ Jupyter found: $jupyterVersion" -ForegroundColor Green
} catch {
    Write-Host "Installing Jupyter requirements..." -ForegroundColor Yellow
    try {
        pip install -r requirements_jupyter.txt
        Write-Host "✓ Jupyter requirements installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "✗ Error: Failed to install Jupyter requirements" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Start Jupyter Notebook
Write-Host "Starting Jupyter Notebook..." -ForegroundColor Green
Write-Host ""
Write-Host "The notebook will open in your default browser" -ForegroundColor Cyan
Write-Host "If it doesn't open automatically, go to: http://localhost:8888" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

try {
    jupyter notebook --notebook-dir="$PSScriptRoot" --port=8888 --no-browser
} catch {
    Write-Host "✗ Error starting Jupyter: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
} 