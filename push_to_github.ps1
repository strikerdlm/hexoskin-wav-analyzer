# PowerShell script to push Hexoskin WAV Analyzer to GitHub
Write-Host "Pushing Hexoskin WAV Analyzer to GitHub..." -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Git installation
Write-Host "Step 1: Checking Git installation..." -ForegroundColor Green
try {
    $gitVersion = git --version
    Write-Host "Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Git is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Git from https://git-scm.com/downloads" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 2: Verify repository status
Write-Host "`nStep 2: Verifying repository status..." -ForegroundColor Green
git status

# Step 3: Set up remote repository
Write-Host "`nStep 3: Setting up remote repository..." -ForegroundColor Green
git remote set-url origin https://github.com/strikerdlm/hexoskin-wav-analyzer.git
Write-Host "Remote URL set to: https://github.com/strikerdlm/hexoskin-wav-analyzer.git" -ForegroundColor Green

# Step 4: Push to GitHub
Write-Host "`nStep 4: Pushing to GitHub..." -ForegroundColor Green
Write-Host "Note: You may be prompted for your GitHub credentials." -ForegroundColor Yellow
Write-Host "If password authentication fails, use a personal access token instead." -ForegroundColor Yellow
Write-Host ""

try {
    git push -u origin master
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nSuccess! Your code has been pushed to GitHub." -ForegroundColor Green
        Write-Host "Visit https://github.com/strikerdlm/hexoskin-wav-analyzer to see your repository." -ForegroundColor Cyan
    } else {
        throw "Git push failed with exit code $LASTEXITCODE"
    }
} catch {
    Write-Host "`nPush failed. Please try these troubleshooting steps:" -ForegroundColor Red
    Write-Host "`n1. Make sure you've created the repository on GitHub:" -ForegroundColor Yellow
    Write-Host "   - Go to https://github.com/new" -ForegroundColor White
    Write-Host "   - Name: hexoskin-wav-analyzer" -ForegroundColor White
    Write-Host "   - Description: An advanced application for analyzing physiological data from Hexoskin WAV files" -ForegroundColor White
    Write-Host "   - Public repository" -ForegroundColor White
    Write-Host "   - Do NOT initialize with README, .gitignore, or license" -ForegroundColor White
    
    Write-Host "`n2. If you're having authentication issues:" -ForegroundColor Yellow
    Write-Host "   - Create a personal access token at https://github.com/settings/tokens" -ForegroundColor White
    Write-Host "   - Use the token instead of your password" -ForegroundColor White
    
    Write-Host "`n3. If the branch name is an issue, try:" -ForegroundColor Yellow
    Write-Host "   git branch -m main" -ForegroundColor White
    Write-Host "   git push -u origin main" -ForegroundColor White
}

Read-Host "`nPress Enter to exit" 