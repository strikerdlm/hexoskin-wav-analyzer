@echo off
echo Pushing Hexoskin WAV Analyzer to GitHub...
echo.

echo Step 1: Checking Git installation...
where git >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Git is not installed or not in PATH.
    echo Please install Git from https://git-scm.com/downloads
    pause
    exit /b 1
)

echo Step 2: Verifying repository status...
git status

echo.
echo Step 3: Setting up remote repository...
git remote set-url origin https://github.com/strikerdlm/hexoskin-wav-analyzer.git

echo.
echo Step 4: Pushing to GitHub...
echo Note: You may be prompted for your GitHub credentials.
echo If password authentication fails, use a personal access token instead.
echo.
git push -u origin master

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Push failed. Please try these troubleshooting steps:
    echo.
    echo 1. Make sure you've created the repository on GitHub:
    echo    - Go to https://github.com/new
    echo    - Name: hexoskin-wav-analyzer
    echo    - Description: An advanced application for analyzing physiological data from Hexoskin WAV files
    echo    - Public repository
    echo    - Do NOT initialize with README, .gitignore, or license
    echo.
    echo 2. If you're having authentication issues:
    echo    - Create a personal access token at https://github.com/settings/tokens
    echo    - Use the token instead of your password
    echo.
    echo 3. If the branch name is an issue, try:
    echo    git branch -m main
    echo    git push -u origin main
    echo.
) else (
    echo.
    echo Success! Your code has been pushed to GitHub.
    echo Visit https://github.com/strikerdlm/hexoskin-wav-analyzer to see your repository.
)

pause 