# Tutorial Dashboard Setup

## 🎯 What This Is

A **separate** Flask application that provides a beautiful web interface for reading your tutorials with:

- ✨ **Gorgeous markdown rendering** with syntax highlighting
- 📑 **Table of contents** navigation 
- 🎨 **Professional styling** for code, achievements, and connections
- 📱 **Mobile responsive** design
- 🔗 **Easy tutorial switching**
- 📋 **Copy-to-clipboard** for code blocks

## 🚀 Quick Start

### Option 1: One-Command Launch
```bash
cd tutorials
./start_dashboard.sh
```

### Option 2: Manual Setup
```bash
cd tutorials

# Create virtual environment (first time only)
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements_dashboard.txt

# Start dashboard
python dashboard_server.py
```

## 🌐 Access Your Dashboard

- **Main Dashboard**: http://localhost:8080
- **Individual Tutorials**: http://localhost:8080/tutorial/react
- **Your Main Project**: Still runs on http://localhost:5000 (unchanged!)

## 📚 Available Tutorials

- `react` - React: Modern Development
- `prerequisites` - Prerequisites: Modern JavaScript  
- `llm-fundamentals` - LLM Fundamentals + RAG
- `llm-agents` - LLM Agents + Multi-Agent Systems
- `computer-vision` - Computer Vision + IoT
- `tinyml` - TinyML + Edge AI
- `tinyml-advanced` - Advanced TinyML Optimization
- `study-guide` - How to Study These Tutorials
- `roadmap` - Project Roadmap

## 🔧 Technical Details

- **Completely separate** from your main `app.py`
- **Port 8080** (vs your project's port 5000)
- **Own virtual environment** in `tutorials/venv/`
- **Zero interference** with your actual project

## 🎨 Features

### Markdown Enhancements
- **YOUR code** references are highlighted in purple
- **✅ Achievements** get special green styling
- **🔗 Connections** are highlighted in orange
- **🎯 Goals** are styled in purple gradients

### Navigation
- **Sidebar** with table of contents
- **Quick tutorial switching**
- **Mobile responsive** layout

### Developer Experience
- **Copy buttons** on all code blocks
- **Syntax highlighting** for all languages
- **Print-friendly** styling
- **Smooth scrolling** navigation

## 🚀 Next Steps

1. **Start the dashboard**: `cd tutorials && ./start_dashboard.sh`
2. **Open browser**: Go to http://localhost:8080
3. **Explore tutorials**: Click any tutorial to see the beautiful rendering
4. **Compare**: Notice how much better this is than reading raw markdown!

Your main project remains completely untouched and continues to work exactly as before. 🎉 