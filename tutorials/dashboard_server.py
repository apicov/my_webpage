#!/usr/bin/env python3
"""
Tutorial Dashboard Server
A separate Flask application for viewing tutorials with markdown rendering.
This does NOT interfere with your main project's app.py.
"""

from flask import Flask, render_template, request, jsonify
import markdown
from markupsafe import Markup
import re
import os
from pathlib import Path
import json
import time

app = Flask(__name__, template_folder='templates')

# Progress tracking file
PROGRESS_FILE = 'learning_progress.json'

def load_progress():
    """Load user progress from JSON file."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        'completed_tutorials': [],
        'current_tutorial': None,
        'reading_time': {},
        'notes': {},
        'achievements': [],
        'streak_days': 0,
        'last_activity': None,
        'total_time_spent': 0
    }

def save_progress(progress_data):
    """Save user progress to JSON file."""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f, indent=2)
        return True
    except:
        return False

def calculate_progress_stats(progress_data):
    """Calculate various progress statistics."""
    # Exclude meta-learning tutorials from progress count
    excluded_tutorials = {'study-guide', 'roadmap'}
    
    # Count only technical tutorials
    technical_tutorials = {k: v for k, v in TUTORIALS.items() if k not in excluded_tutorials}
    total_tutorials = len(technical_tutorials)
    
    # Count completed technical tutorials only
    completed_technical = [t for t in progress_data['completed_tutorials'] if t not in excluded_tutorials]
    completed = len(completed_technical)
    
    completion_rate = (completed / total_tutorials * 100) if total_tutorials > 0 else 0
    
    # Calculate estimated total time (excluding meta-learning tutorials)
    total_estimated_minutes = 0
    for tutorial_id, tutorial in TUTORIALS.items():
        if tutorial_id in excluded_tutorials:
            continue  # Skip meta-learning tutorials
            
        duration = tutorial['duration']
        # Extract time from duration string, handling ranges (e.g., "2-3 hours" -> 2, "45-60 minutes" -> 45)
        if 'hour' in duration:
            hours = duration.split()[0].split('-')[0]  # Take first number from range
            total_estimated_minutes += int(hours) * 60
        elif 'minute' in duration:
            minutes = duration.split()[0].split('-')[0]  # Take first number from range
            total_estimated_minutes += int(minutes)
    
    return {
        'total_tutorials': total_tutorials,
        'completed': completed,
        'remaining': total_tutorials - completed,
        'completion_rate': round(completion_rate, 1),
        'total_estimated_minutes': total_estimated_minutes,
        'time_spent_minutes': progress_data.get('total_time_spent', 0)
    }

# Tutorial configuration
TUTORIALS = {
    'codebase-overview': {
        'title': 'Understanding Your Codebase',
        'file': 'CODEBASE_OVERVIEW.md',
        'description': 'Complete architecture guide to YOUR React + Flask system',
        'level': 'Foundation',
        'duration': '45-60 minutes'
    },
    'prerequisites': {
        'title': 'Prerequisites: Modern JavaScript',
        'file': 'PREREQUISITES_TUTORIAL.md',
        'description': 'Master modern JavaScript through YOUR actual code',
        'level': 'Foundation',
        'duration': '2-3 hours'
    },
    'react': {
        'title': 'React: Modern Development',
        'file': 'REACT_TUTORIAL.md', 
        'description': 'Transform YOUR ChatInterface.js with React 18 + TypeScript',
        'level': 'Foundation',
        'duration': '4-6 hours'
    },
    'llm-fundamentals': {
        'title': 'LLM Fundamentals + RAG',
        'file': 'LLM_FUNDAMENTALS_KERAS3_TUTORIAL.md',
        'description': 'Enhance YOUR Assistant with advanced AI capabilities',
        'level': 'Intermediate',
        'duration': '6-8 hours'
    },
    'llm-agents': {
        'title': 'LLM Agents + Multi-Agent Systems',
        'file': 'LLM_AGENTS_KERAS3_TUTORIAL.md',
        'description': 'Add autonomous capabilities to YOUR Assistant',
        'level': 'Advanced',
        'duration': '6-8 hours'
    },
    'computer-vision': {
        'title': 'Computer Vision + IoT',
        'file': 'IOT_WEBCAM_TUTORIAL.md',
        'description': 'Add camera controls and vision AI to YOUR chat',
        'level': 'Intermediate',
        'duration': '5-7 hours'
    },
    'tinyml': {
        'title': 'TinyML + Edge AI',
        'file': 'TINYML_TUTORIAL.md',
        'description': 'Control edge AI devices through YOUR platform',
        'level': 'Advanced',
        'duration': '5-7 hours'
    },
    'tinyml-advanced': {
        'title': 'Advanced TinyML Optimization',
        'file': 'TINYML_ADVANCED_TUTORIAL.md',
        'description': 'Enterprise-grade optimization for YOUR platform',
        'level': 'Expert',
        'duration': '4-6 hours'
    },
    'study-guide': {
        'title': 'How to Study These Tutorials',
        'file': 'HOW_TO_STUDY_TUTORIALS.md',
        'description': 'Integrated learning strategy for maximum effectiveness',
        'level': 'Meta-Learning',
        'duration': '1 hour'
    },
    'roadmap': {
        'title': 'Project Roadmap',
        'file': 'PROJECT_ROADMAP.md',
        'description': 'Complete platform evolution strategy',
        'level': 'Planning',
        'duration': '30 minutes'
    }
}

def preprocess_code_blocks(content):
    """Preprocess markdown content to ensure proper language classes for Prism.js"""
    import re
    
    # Pattern to match fenced code blocks
    pattern = r'```(\w*)\n(.*?)```'
    
    def replace_code_block(match):
        lang = match.group(1) or 'text'
        code = match.group(2)
        return f'```{lang}\n{code}```'
    
    return re.sub(pattern, replace_code_block, content, flags=re.DOTALL)

def process_markdown_content(content):
    """Process markdown content with custom enhancements."""
    # Pre-process content to ensure proper language tags
    content = preprocess_code_blocks(content)
    
    # Configure markdown with extensions
    md = markdown.Markdown(extensions=[
        'fenced_code', 
        'tables',
        'toc',
        'attr_list'
    ], extension_configs={
        'toc': {
            'permalink': True,
            'title': 'Table of Contents'
        }
    })
    
    # Convert markdown to HTML
    html_content = md.convert(content)
    
    # Add custom styling for tutorial-specific elements
    html_content = enhance_tutorial_styling(html_content)
    
    # Get TOC if available
    toc = getattr(md, 'toc', '')
    
    return Markup(html_content), toc

def enhance_tutorial_styling(html_content):
    """Add custom CSS classes and styling to HTML content."""
    
    # First, fix code blocks for Prism.js syntax highlighting
    html_content = fix_code_block_classes(html_content)
    
    # Style code blocks with YOUR project references
    html_content = re.sub(
        r'YOUR ([^<\s]+)',
        r'<span class="your-code">YOUR \1</span>',
        html_content
    )
    
    # Style achievements and checkmarks
    html_content = re.sub(
        r'✅ ([^<\n]+)',
        r'<div class="achievement">✅ <span class="achievement-text">\1</span></div>',
        html_content
    )
    
    # Style tutorial connections
    html_content = re.sub(
        r'🔗 ([^<\n]+)',
        r'<div class="connection">🔗 <span class="connection-text">\1</span></div>',
        html_content
    )
    
    # Style goals and targets
    html_content = re.sub(
        r'🎯 ([^<\n]+)',
        r'<div class="goal">🎯 <span class="goal-text">\1</span></div>',
        html_content
    )
    
    return html_content

def fix_code_block_classes(html_content):
    """Fix code block classes for proper Prism.js syntax highlighting"""
    import re
    
    # Pattern to find code blocks and ensure they have language classes
    def replace_code_block(match):
        existing_class = match.group(1) if match.group(1) else ''
        
        # Extract language from existing class or set default
        lang_match = re.search(r'language-(\w+)', existing_class)
        if lang_match:
            lang = lang_match.group(1)
        else:
            # Try to extract from other class formats
            if 'python' in existing_class.lower():
                lang = 'python'
            elif 'javascript' in existing_class.lower() or 'js' in existing_class.lower():
                lang = 'javascript'
            elif 'bash' in existing_class.lower() or 'shell' in existing_class.lower():
                lang = 'bash'
            elif 'json' in existing_class.lower():
                lang = 'json'
            elif 'html' in existing_class.lower():
                lang = 'html'
            elif 'css' in existing_class.lower():
                lang = 'css'
            else:
                lang = 'text'
        
        return f'<pre><code class="language-{lang}">'
    
    # Replace all code blocks to ensure proper language classes
    html_content = re.sub(
        r'<pre><code(?: class="([^"]*)")?[^>]*>',
        replace_code_block,
        html_content
    )
    
    return html_content

# Routes
@app.route('/')
def dashboard():
    """Main tutorial dashboard with navigation"""
    progress_data = load_progress()
    progress_stats = calculate_progress_stats(progress_data)
    
    # Separate technical tutorials from meta-learning resources
    excluded_tutorials = {'study-guide', 'roadmap'}
    technical_tutorials = {k: v for k, v in TUTORIALS.items() if k not in excluded_tutorials}
    meta_tutorials = {k: v for k, v in TUTORIALS.items() if k in excluded_tutorials}
    
    return render_template('dashboard.html', 
                         tutorials=technical_tutorials, 
                         meta_tutorials=meta_tutorials,
                         personal_info={'name': 'Developer'}, 
                         progress_stats=progress_stats)

@app.route('/tutorial/<tutorial_id>')
def tutorial_viewer(tutorial_id):
    """View a specific tutorial with markdown rendering"""
    if tutorial_id not in TUTORIALS:
        return "Tutorial not found", 404
    
    tutorial_info = TUTORIALS[tutorial_id]
    tutorial_path = f"{tutorial_info['file']}"
    
    try:
        with open(tutorial_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Process markdown to HTML
        html_content, toc = process_markdown_content(markdown_content)
        
        return render_template('tutorial_viewer.html', 
                             tutorial=tutorial_info,
                             tutorial_id=tutorial_id,
                             content=html_content,
                             toc=toc,
                             tutorials=TUTORIALS)
    
    except FileNotFoundError:
        return f"Tutorial file not found: {tutorial_info['file']}", 404
    except Exception as e:
        return f"Error loading tutorial: {str(e)}", 500

@app.route('/api/tutorials')
def api_tutorials():
    """API endpoint to get tutorials list"""
    return jsonify(TUTORIALS)

@app.route('/api/progress', methods=['GET'])
def get_progress():
    """Get current learning progress"""
    progress_data = load_progress()
    progress_stats = calculate_progress_stats(progress_data)
    return jsonify({
        'progress': progress_data,
        'stats': progress_stats
    })

@app.route('/api/progress/complete', methods=['POST'])
def mark_tutorial_complete():
    """Mark a tutorial as completed"""
    data = request.get_json()
    tutorial_id = data.get('tutorial_id')
    time_spent = data.get('time_spent', 0)  # in minutes
    
    if tutorial_id not in TUTORIALS:
        return jsonify({'error': 'Invalid tutorial ID'}), 400
    
    progress_data = load_progress()
    
    # Add to completed if not already there
    if tutorial_id not in progress_data['completed_tutorials']:
        progress_data['completed_tutorials'].append(tutorial_id)
    
    # Update reading time
    progress_data['reading_time'][tutorial_id] = time_spent
    progress_data['total_time_spent'] += time_spent
    progress_data['last_activity'] = time.time()
    
    # Check for achievements
    completed_count = len(progress_data['completed_tutorials'])
    new_achievements = []
    
    if completed_count == 1 and 'first_tutorial' not in progress_data['achievements']:
        new_achievements.append('first_tutorial')
    elif completed_count == 3 and 'third_tutorial' not in progress_data['achievements']:
        new_achievements.append('third_tutorial')
    elif completed_count == len(TUTORIALS) and 'all_complete' not in progress_data['achievements']:
        new_achievements.append('all_complete')
    
    progress_data['achievements'].extend(new_achievements)
    
    if save_progress(progress_data):
        return jsonify({
            'success': True,
            'new_achievements': new_achievements,
            'stats': calculate_progress_stats(progress_data)
        })
    else:
        return jsonify({'error': 'Failed to save progress'}), 500

@app.route('/api/progress/start', methods=['POST'])
def start_tutorial():
    """Mark tutorial as started/currently reading"""
    data = request.get_json()
    tutorial_id = data.get('tutorial_id')
    
    if tutorial_id not in TUTORIALS:
        return jsonify({'error': 'Invalid tutorial ID'}), 400
    
    progress_data = load_progress()
    progress_data['current_tutorial'] = tutorial_id
    progress_data['last_activity'] = time.time()
    
    if save_progress(progress_data):
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Failed to save progress'}), 500

@app.route('/api/progress/note', methods=['POST'])
def save_note():
    """Save a note for a tutorial"""
    data = request.get_json()
    tutorial_id = data.get('tutorial_id')
    note = data.get('note', '')
    
    if tutorial_id not in TUTORIALS:
        return jsonify({'error': 'Invalid tutorial ID'}), 400
    
    progress_data = load_progress()
    progress_data['notes'][tutorial_id] = note
    progress_data['last_activity'] = time.time()
    
    if save_progress(progress_data):
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Failed to save progress'}), 500

if __name__ == '__main__':
    print("🚀 Starting Tutorial Dashboard Server...")
    print("📚 Dashboard will be available at: http://localhost:8080")
    print("🔗 Access individual tutorials at: http://localhost:8080/tutorial/<tutorial-name>")
    print("⚠️  This is separate from your main project (which runs on port 5000)")
    app.run(debug=True, port=8080, host='0.0.0.0') 