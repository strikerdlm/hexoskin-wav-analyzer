@echo off
echo Starting Valquiria Jupyter Analysis Environment
echo =============================================
python comprehensive_jupyter_fix.py
if %ERRORLEVEL% EQU 0 (
    echo Environment setup complete. Starting Jupyter...
    jupyter notebook --port=8888
) else (
    echo Setup failed. Please check the error messages above.
    pause
)
