# Manual GitHub Setup for Hexoskin WAV Analyzer

Since we're experiencing some issues with the automated setup, here's a manual guide to push your project to GitHub.

## Step 1: Create a New Repository on GitHub

1. Go to [GitHub](https://github.com) and log in to your account (@strikerdlm)
2. Click the "+" icon in the top-right corner and select "New repository"
3. Name your repository: "hexoskin-wav-analyzer"
4. Add a description: "An advanced application for analyzing physiological data from Hexoskin WAV files"
5. Choose "Public" visibility (or "Private" if you prefer)
6. Do NOT initialize the repository with a README, .gitignore, or license
7. Click "Create repository"

## Step 2: Manual Setup on Your Computer

1. Open Windows Explorer and navigate to your project folder:
   ```
   C:\Users\User\OneDrive\FAC\Research\Valquiria\Data
   ```

2. Create a new text file named `.gitignore` (make sure it has no file extension, just `.gitignore`)
   - Right-click > New > Text Document
   - Name it `.gitignore.` (with a dot at the end)
   - Windows will automatically remove the `.txt` extension

3. Open the `.gitignore` file with a text editor and paste the following content:

```
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
```

## Step 3: Open Command Prompt as Administrator

1. Press Windows key
2. Type "cmd"
3. Right-click on "Command Prompt" and select "Run as administrator"

## Step 4: Navigate to Your Project Directory

```
cd C:\Users\User\OneDrive\FAC\Research\Valquiria\Data
```

## Step 5: Initialize Git Repository

```
git init
```

## Step 6: Add Only the Software Files

Add each file individually:

```
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
git add GITHUB_MANUAL_SETUP.md
git add hexoskin_wav_example.py
```

## Step 7: Check What Will Be Committed

```
git status
```

Make sure NO Sol folders or data files are listed as "to be committed".

## Step 8: Commit Your Files

```
git commit -m "Initial commit of Hexoskin WAV Analyzer"
```

## Step 9: Connect to GitHub

```
git remote add origin https://github.com/strikerdlm/hexoskin-wav-analyzer.git
```

## Step 10: Push to GitHub

```
git push -u origin main
```

If your default branch is named "master" instead of "main", use:

```
git push -u origin master
```

## Step 11: Verify on GitHub

1. Go to `https://github.com/strikerdlm/hexoskin-wav-analyzer`
2. Confirm that all your software files have been uploaded correctly
3. Verify that NO Sol folders or data files were uploaded

## Troubleshooting

If you encounter any issues:

1. **Authentication problems**: You might need to authenticate with GitHub. Follow the prompts or use a personal access token.

2. **Branch name issues**: If you get an error about the branch name, try:
   ```
   git branch -m main
   git push -u origin main
   ```

3. **File not found errors**: Make sure you're in the correct directory and the files exist.

4. **Permission denied**: Make sure you're running Command Prompt as Administrator. 