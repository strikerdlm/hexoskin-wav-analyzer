# PowerShell script to set up GitHub repository for Hexoskin WAV Analyzer
# Created for @strikerdlm

# Create .gitignore file
$gitignoreContent = @"
# Python bytecode
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/

# Virtual environments
venv/
env/
ENV/

# IDE specific files
.idea/
.vscode/
*.swp
*.swo
.cursor/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# Data files
*.csv
*.xlsx
*.wav
Data backup/
Old plots/

# Explicitly exclude Sol folders
Sol*/
Sol\ */
/Sol\ *
/Sol*/

# Exclude specific data files
Mara_full.csv
Cronograma*.xlsx
analyze_physiological_data.py
compile_full_dataset.py
fix_data_analysis.py
1.ipynb

# Logs
*.log

# OS specific
.DS_Store
Thumbs.db
"@

# Write .gitignore file
$gitignoreContent | Out-File -FilePath ".gitignore" -Encoding utf8

Write-Host "Created .gitignore file" -ForegroundColor Green

# Initialize Git repository
git init
Write-Host "Initialized Git repository" -ForegroundColor Green

# Add only software-related files
$filesToAdd = @(
    ".gitignore",
    "README.md",
    "INSTALL.md",
    "hexoskin_wav_loader.py",
    "requirements.txt",
    "run.py",
    "run_analyzer.bat",
    "run_analyzer.sh",
    "setup.py",
    "GITHUB_SETUP.md",
    "hexoskin_wav_example.py",
    "setup_github_repo.ps1"
)

foreach ($file in $filesToAdd) {
    if (Test-Path $file) {
        git add $file
        Write-Host "Added $file to Git" -ForegroundColor Green
    } else {
        Write-Host "Warning: $file not found, skipping" -ForegroundColor Yellow
    }
}

# Show status
Write-Host "`nCurrent Git status:" -ForegroundColor Cyan
git status

# Commit instructions
Write-Host "`n`nTo complete the setup, run these commands:" -ForegroundColor Magenta
Write-Host "1. Commit your files:" -ForegroundColor White
Write-Host "   git commit -m ""Initial commit of Hexoskin WAV Analyzer""" -ForegroundColor Yellow
Write-Host "2. Add your GitHub repository as remote:" -ForegroundColor White
Write-Host "   git remote add origin https://github.com/strikerdlm/hexoskin-wav-analyzer.git" -ForegroundColor Yellow
Write-Host "3. Push to GitHub:" -ForegroundColor White
Write-Host "   git push -u origin main" -ForegroundColor Yellow
Write-Host "`nNote: If your default branch is 'master' instead of 'main', use 'git push -u origin master'" -ForegroundColor Cyan 