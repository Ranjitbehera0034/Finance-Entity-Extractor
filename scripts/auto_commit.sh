#!/bin/bash
# ============================================
# Git Auto-Commit Scheduler
# Commits changes daily to maintain activity
# ============================================

# Configuration
PROJECT_DIR="$HOME/llm-mail-trainer"
LOG_DIR="$PROJECT_DIR/scheduler_logs"
LOG_FILE="$LOG_DIR/commits.log"
REMOTE="hf"  # HuggingFace remote

# Create log directory
mkdir -p "$LOG_DIR"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to get random commit message
get_commit_message() {
    local messages=(
        "chore: Update project files"
        "docs: Minor documentation updates"
        "refactor: Code cleanup"
        "style: Format improvements"
        "chore: Regular maintenance"
        "docs: Improve comments"
        "chore: Sync changes"
        "refactor: Minor optimizations"
    )
    local idx=$((RANDOM % ${#messages[@]}))
    echo "${messages[$idx]}"
}

# Main function
main() {
    log "=== Starting auto-commit scheduler ==="
    
    cd "$PROJECT_DIR" || {
        log "ERROR: Cannot access $PROJECT_DIR"
        exit 1
    }
    
    # Check for changes
    if git diff --quiet && git diff --cached --quiet; then
        log "No changes to commit"
        
        # Optional: Create a small update to ensure commit
        # Uncomment if you want to force daily commits
        # echo "# Last updated: $(date)" >> .project-meta
        # git add .project-meta
    else
        log "Changes detected, preparing commit..."
    fi
    
    # Stage all changes
    git add -A
    
    # Check again after staging
    if git diff --cached --quiet; then
        log "Nothing staged to commit"
        exit 0
    fi
    
    # Get current branch
    BRANCH=$(git branch --show-current)
    log "Current branch: $BRANCH"
    
    # Commit with random message
    COMMIT_MSG=$(get_commit_message)
    git commit -m "$COMMIT_MSG" >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        log "✅ Committed: $COMMIT_MSG"
        
        # Push to remote
        git push "$REMOTE" "$BRANCH" >> "$LOG_FILE" 2>&1
        
        if [ $? -eq 0 ]; then
            log "✅ Pushed to $REMOTE/$BRANCH"
        else
            log "❌ Push failed"
        fi
    else
        log "❌ Commit failed"
    fi
    
    log "=== Scheduler complete ==="
}

# Run main function
main
