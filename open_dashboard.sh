#!/bin/bash

# Learning Dashboard Launcher
echo "🚀 Starting AI Learning Dashboard..."

# Check if Flask app is running
if ! curl -s http://localhost:5000 > /dev/null; then
    echo "⚡ Starting Flask server..."
    python app.py &
    FLASK_PID=$!
    echo "Flask server started with PID: $FLASK_PID"
    
    # Wait for server to start
    sleep 3
fi

# Open dashboard in browser
echo "🌐 Opening dashboard in browser..."

# Try different methods to open the dashboard
if command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open http://localhost:5000/dashboard
elif command -v open &> /dev/null; then
    # macOS
    open http://localhost:5000/dashboard
elif command -v start &> /dev/null; then
    # Windows
    start http://localhost:5000/dashboard
else
    echo "📖 Please manually open: http://localhost:5000/dashboard"
fi

echo "✅ Dashboard should now be open in your browser!"
echo "📚 Happy learning!"
echo "💡 Use Ctrl+C to stop the Flask server when done" 