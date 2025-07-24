#!/bin/bash

# Learning Dashboard Launcher
echo "🚀 Opening AI Learning Dashboard..."

# Try different methods to open the dashboard
if command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open tutorials/LEARNING_DASHBOARD.html
elif command -v open &> /dev/null; then
    # macOS
    open tutorials/LEARNING_DASHBOARD.html
elif command -v start &> /dev/null; then
    # Windows
    start tutorials/LEARNING_DASHBOARD.html
else
    echo "📖 Please manually open: tutorials/LEARNING_DASHBOARD.html"
    echo "💡 Or use: code tutorials/ to open in VS Code"
fi

echo "✅ Dashboard should now be open in your browser!"
echo "�� Happy learning!" 