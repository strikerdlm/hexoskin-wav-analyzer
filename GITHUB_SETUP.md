# GitHub Setup Guide for Hexoskin WAV Analyzer

This guide will help you push your Hexoskin WAV Analyzer project to GitHub.

## Prerequisites

1. You already have a GitHub account: [@strikerdlm](https://github.com/strikerdlm)
2. [Install Git](https://git-scm.com/downloads) on your computer if not already installed
3. [Configure Git](https://docs.github.com/en/get-started/quickstart/set-up-git) with your username and email

## Step 1: Create a New Repository on GitHub

1. Log in to your GitHub account ([@strikerdlm](https://github.com/strikerdlm))
2. Click the "+" icon in the top-right corner and select "New repository"
3. Name your repository: "hexoskin-wav-analyzer"
4. Add a description: "An advanced application for analyzing physiological data from Hexoskin WAV files"
5. Choose "Public" visibility (or "Private" if you prefer)
6. Do NOT initialize the repository with a README, .gitignore, or license
7. Click "Create repository"

## Step 2: Initialize Your Local Repository

Open PowerShell and navigate to your project directory:

```bash
cd "C:\Users\User\OneDrive\FAC\Research\Valquiria\Data"
```

Initialize a Git repository:

```bash
git init
```

## Step 3: Create a Better .gitignore File

First, ensure your .gitignore file is properly set up to exclude data folders:

```bash
# Create or update .gitignore file
@"
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

# Logs
*.log

# OS specific
.DS_Store
Thumbs.db
"@ | Out-File -FilePath .gitignore -Encoding utf8
```

## Step 4: Add Only the Software-Related Files

Add ONLY the files related to the software (excluding Sol folders and data files):

```bash
# Add only the software-related files
git add .gitignore
git add README.md
git add INSTALL.md
git add hexoskin_wav_loader.py
git add requirements.txt
git add run.py
git add run_analyzer.bat
git add run_analyzer.sh
git add setup.py
git add GITHUB_SETUP.md
git add hexoskin_wav_example.py
```

Verify what will be committed (this should NOT show any Sol folders):

```bash
git status
```

Commit the files:

```bash
git commit -m "Initial commit of Hexoskin WAV Analyzer"
```

## Step 5: Connect to GitHub and Push

Connect your local repository to your GitHub repository:

```bash
git remote add origin https://github.com/strikerdlm/hexoskin-wav-analyzer.git
```

Push your code to GitHub:

```bash
git push -u origin main
```

Note: If your default branch is named "master" instead of "main", use:

```bash
git push -u origin master
```

## Step 6: Verify Your Repository

1. Go to `https://github.com/strikerdlm/hexoskin-wav-analyzer`
2. Confirm that all your software files have been uploaded correctly
3. Verify that NO Sol folders or data files were uploaded

## Important Notes

- This setup will ONLY push the software-related files to GitHub
- All Sol folders and data files will be excluded
- If you need to add more software files later, use `git add filename` for each file
- Always check `git status` before committing to ensure no data files are included

## Additional Tips

### Adding More Files Later

If you need to add more files or make changes:

```bash
git add new_file.py
git commit -m "Add new feature"
git push
```

### Creating Releases

When you reach a stable version:

1. Go to your repository on GitHub
2. Click on "Releases" on the right side
3. Click "Create a new release"
4. Add a tag (e.g., "v0.0.1"), title, and description
5. Click "Publish release"

### Setting Up GitHub Pages (Optional)

If you want to create a project website:

1. Go to your repository settings
2. Scroll down to "GitHub Pages"
3. Select the branch you want to use (usually "main")
4. Choose a theme
5. Your site will be available at `https://YOUR_USERNAME.github.io/hexoskin-wav-analyzer` 