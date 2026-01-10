#!/bin/bash
# ============================================
# Setup Git Branches
# Creates feature, dev, main branch structure
# ============================================

PROJECT_DIR="$HOME/llm-mail-trainer"
cd "$PROJECT_DIR" || exit 1

echo "🔧 Setting up branch structure..."

# Ensure we're on main
git checkout main 2>/dev/null || git checkout -b main

# Create dev branch from main
git branch -D dev 2>/dev/null
git checkout -b dev
echo "✅ Created 'dev' branch"

# Create feature branch from dev
git branch -D feature/improvements 2>/dev/null
git checkout -b feature/improvements
echo "✅ Created 'feature/improvements' branch"

# Push all branches to remote
echo ""
echo "🚀 Pushing branches to HuggingFace..."
git push hf main --force 2>/dev/null
git push hf dev --force 2>/dev/null
git push hf feature/improvements --force 2>/dev/null

echo ""
echo "📋 Branch structure:"
git branch -a

echo ""
echo "✅ Branch setup complete!"
echo ""
echo "Branch workflow:"
echo "  feature/* → dev → main"
echo ""
echo "Commands:"
echo "  git checkout feature/improvements  # Work on features"
echo "  git checkout dev                   # Merge features here"
echo "  git checkout main                  # Production code"
