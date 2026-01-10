#!/bin/bash
# ============================================
# Install Auto-Commit Scheduler (cron job)
# Runs daily at 10 PM IST
# ============================================

SCRIPT_PATH="$HOME/llm-mail-trainer/scripts/auto_commit.sh"

# Make script executable
chmod +x "$SCRIPT_PATH"
chmod +x "$HOME/llm-mail-trainer/scripts/setup_branches.sh"

echo "🔧 Setting up daily auto-commit scheduler..."

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "auto_commit.sh"; then
    echo "⚠️  Cron job already exists. Removing old one..."
    crontab -l | grep -v "auto_commit.sh" | crontab -
fi

# Add new cron job (10 PM IST = 4:30 PM UTC)
# Format: minute hour day month weekday command
(crontab -l 2>/dev/null; echo "30 16 * * * $SCRIPT_PATH") | crontab -

echo "✅ Cron job installed!"
echo ""
echo "📋 Current cron jobs:"
crontab -l
echo ""
echo "The script will run daily at 10:00 PM IST (4:30 PM UTC)"
echo ""
echo "To manually run the scheduler:"
echo "  bash $SCRIPT_PATH"
echo ""
echo "To remove the scheduler:"
echo "  crontab -e  # and delete the line"
echo ""
echo "To check logs:"
echo "  cat ~/llm-mail-trainer/scheduler_logs/commits.log"
