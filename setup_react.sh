#!/bin/bash

echo "🚀 Setting up React Frontend + Flask Backend"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "📦 Installing Flask dependencies..."
pip install -r requirements.txt

echo "📦 Installing React dependencies..."
cd frontend
npm install
cd ..

echo "✅ Setup complete!"
echo ""
echo "🎯 To start the application:"
echo ""
echo "Terminal 1 (Flask Backend):"
echo "  python app.py"
echo ""
echo "Terminal 2 (React Frontend):"
echo "  cd frontend && npm start"
echo ""
echo "🌐 Your sites will be available at:"
echo "  Flask (HTML): http://localhost:5000"
echo "  React: http://localhost:3000"
echo ""
echo "📝 Note: Make sure your Flask backend is running before starting the React app!" 