# Simple Guide to Push Hexoskin WAV Analyzer to GitHub

This guide provides the simplest steps to push your project to GitHub.

## Step 1: Create the Repository on GitHub

1. Go to [GitHub New Repository Page](https://github.com/new)
2. Enter repository name: `hexoskin-wav-analyzer`
3. Add description: `An advanced application for analyzing physiological data from Hexoskin WAV files`
4. Choose "Public" visibility
5. Do NOT initialize with README, .gitignore, or license
6. Click "Create repository"

## Step 2: Push Your Code

### Option 1: Using the Batch File (Recommended for Windows)

1. Double-click the `push_to_github.bat` file in your project folder
2. Follow the prompts
3. Enter your GitHub credentials when asked

### Option 2: Using the PowerShell Script

1. Right-click `push_to_github.ps1` and select "Run with PowerShell"
2. Follow the prompts
3. Enter your GitHub credentials when asked

### Option 3: Manual Commands

If the scripts don't work, open Command Prompt as Administrator and run:

```
cd C:\Users\User\OneDrive\FAC\Research\Valquiria\Data
git push -u origin master
```

## Troubleshooting

### Authentication Issues

GitHub no longer accepts password authentication through the command line. You need to use a personal access token:

1. Go to [GitHub Personal Access Tokens](https://github.com/settings/tokens)
2. Click "Generate new token"
3. Give it a name like "Hexoskin WAV Analyzer"
4. Select the "repo" scope
5. Click "Generate token"
6. Copy the token
7. Use this token instead of your password when prompted

### Branch Name Issues

If you get an error about the branch name, try:

```
git branch -m main
git push -u origin main
```

### Repository Already Exists

If you get an error about the repository already existing, you might need to force push:

```
git push -f origin master
```

## Verification

After pushing, go to `https://github.com/strikerdlm/hexoskin-wav-analyzer` to verify that all your files have been uploaded correctly. 