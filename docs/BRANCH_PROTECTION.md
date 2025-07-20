# Branch Protection Rules

This document provides instructions for setting up branch protection rules for the Hexoskin WAV Analyzer repository on GitHub. These rules help maintain code quality and ensure proper review processes.

## Why Branch Protection?

Branch protection rules:
- Prevent direct pushes to important branches
- Require code reviews before merging
- Ensure CI checks pass before merging
- Maintain a clean and stable codebase
- Enforce best practices for collaborative development

## Setting Up Branch Protection Rules

### For the Main Branch

1. Go to your repository on GitHub: https://github.com/strikerdlm/hexoskin-wav-analyzer
2. Click on "Settings" in the top navigation bar
3. In the left sidebar, click on "Branches"
4. Under "Branch protection rules", click "Add rule"
5. In the "Branch name pattern" field, enter `main`
6. Configure the following settings:

#### Required Settings

- [x] **Require a pull request before merging**
  - [x] Require approvals (1)
  - [x] Dismiss stale pull request approvals when new commits are pushed
  - [x] Require review from Code Owners

- [x] **Require status checks to pass before merging**
  - [x] Require branches to be up to date before merging
  - Status checks to require:
    - [x] CI (from GitHub Actions)

- [x] **Require conversation resolution before merging**

#### Optional Settings (Recommended)

- [x] **Do not allow bypassing the above settings**
- [x] **Restrict who can push to matching branches**
  - Add yourself and any trusted collaborators

7. Click "Create" or "Save changes"

### For Release Branches

If you use release branches (e.g., `release/*`), consider adding similar protection rules:

1. Follow steps 1-4 above
2. In the "Branch name pattern" field, enter `release/*`
3. Configure similar settings as for the main branch
4. Click "Create" or "Save changes"

## Working with Protected Branches

When branches are protected, the workflow changes slightly:

1. **Create a feature branch** for your work
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit them
   ```bash
   git add .
   git commit -m "Implement your feature"
   ```

3. **Push your feature branch** to GitHub
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a pull request** on GitHub
   - Go to your repository
   - Click "Compare & pull request"
   - Fill in the details
   - Click "Create pull request"

5. **Wait for reviews and CI checks** to pass

6. **Merge the pull request** once approved

## Troubleshooting

- **Cannot push to protected branch**: Create a pull request instead
- **CI checks failing**: Fix the issues and push new commits to your feature branch
- **Stale reviews**: Ask for new reviews after pushing changes

## Best Practices

- Never force push to protected branches
- Keep pull requests focused on a single feature or fix
- Address all review comments before merging
- Keep your feature branches up to date with the main branch 