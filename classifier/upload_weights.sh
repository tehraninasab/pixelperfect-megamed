#!/bin/bash

# Initialize Git LFS
git lfs install

# Track common large file types
git lfs track "*.pth"
git lfs track "*.csv"


# Commit LFS configuration
git add .gitattributes
git commit -m "Configure Git LFS for large weights and CSV files"

# Push to remote
git push 

echo "Git LFS is now configured for your repository!"
echo "You can now add large files normally with 'git add' and 'git commit'"