#!/bin/bash

echo "🚀 Starting Tutorial Dashboard with Markdown Viewer..."

# Check if we're in the tutorials directory
if [ ! -f "dashboard_server.py" ]; then
    echo "❌ Please run this script from the tutorials/ directory"
    echo "💡 Try: cd tutorials && ./start_dashboard.sh"
    exit 1
fi

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment for dashboard..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📚 Installing dashboard requirements..."
pip install -r requirements_dashboard.txt

# Start the dashboard server
echo ""
echo "🎯 Tutorial Dashboard Features:"
echo "   • Beautiful markdown rendering"
echo "   • Table of contents navigation"
echo "   • Code syntax highlighting"
echo "   • Copy-to-clipboard for code blocks"
echo "   • Mobile responsive design"
echo ""
echo "🌐 Dashboard will open at: http://localhost:8080"
echo "⚠️  This runs separately from your main project (port 5000)"
echo ""

# Start the server
python dashboard_server.py 